"""
Congestion Prediction Service — NetSentinel AI Phase 2
======================================================
Loads the trained H=5 congestion prediction model artifact and serves
replayed predictions and historical context from the simulated bottleneck dataset.

SCIENTIFIC DISCLAIMER:
----------------------
All predictions, telemetry, and metrics provided by this service are derived
from a SIMULATED bottleneck-link model (100 Mbps). They are explicitly labeled
as simulated in every API response.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from ml.congestion.evaluate_multi_horizon import build_multi_horizon_dataset
from ml.congestion.features import CONGESTION_THRESHOLD

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/model2_congestion/h5/model.joblib")
DEFAULT_DATASET_PATH = Path("data/processed/simulated_congestion_1min.csv")
DEFAULT_AS_OF_TIMESTAMP = "2024-11-07 15:00:00"  # Midst of Thursday surge in Test split
DEFAULT_HORIZON_MINUTES = 5
LINK_CAPACITY_MBPS = 100.0
MIN_LOOKBACK_MINUTES = 30


class CongestionService:
    """Service for serving H=5 congestion prediction forecasts and status."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        data_path: str | Path = DEFAULT_DATASET_PATH,
    ):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self._model_bundle: Optional[Dict[str, Any]] = None
        self._df_clean: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[List[str]] = None
        self._model_load_attempted = False

        self._load_model_safe()
        self._load_data_safe()

    def _load_model_safe(self) -> None:
        """Attempt to load model artifact without throwing unhandled exceptions."""
        self._model_load_attempted = True
        if not self.model_path.exists():
            logger.warning("Congestion model artifact not found at %s", self.model_path)
            self._model_bundle = None
            return

        try:
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self._model_bundle = bundle
                self._feature_cols = bundle.get("feature_cols", [])
                logger.info(
                    "Loaded congestion model %s (horizon=%s) from %s",
                    bundle.get("champion_name"),
                    bundle.get("horizon_minutes"),
                    self.model_path,
                )
            else:
                logger.warning("Model bundle at %s is malformed.", self.model_path)
                self._model_bundle = None
        except Exception as exc:
            logger.error("Failed to load congestion model from %s: %s", self.model_path, exc)
            self._model_bundle = None

    def _load_data_safe(self) -> None:
        """Load and prepare replay dataset."""
        if not self.data_path.exists():
            logger.warning("Simulated congestion dataset not found at %s", self.data_path)
            self._df_clean = None
            return

        try:
            raw_df = pd.read_csv(self.data_path)
            clean_df, feature_cols, _, _ = build_multi_horizon_dataset(
                raw_df, horizons=[DEFAULT_HORIZON_MINUTES], threshold=CONGESTION_THRESHOLD
            )
            self._df_clean = clean_df
            if not self._feature_cols:
                self._feature_cols = feature_cols
            logger.info("Loaded %d clean aligned rows for congestion replay.", len(clean_df))
        except Exception as exc:
            logger.error("Failed to load and prepare congestion replay dataset: %s", exc)
            self._df_clean = None

    @property
    def is_model_available(self) -> bool:
        """Check if model artifact is loaded and ready."""
        return self._model_bundle is not None and "model" in self._model_bundle

    @property
    def is_data_available(self) -> bool:
        """Check if replay dataset is loaded."""
        return self._df_clean is not None and not self._df_clean.empty

    def get_status(self) -> Dict[str, Any]:
        """Return operational readiness and configuration status."""
        model_name = self._model_bundle.get("champion_name") if self.is_model_available else None
        horizon = self._model_bundle.get("horizon_minutes", DEFAULT_HORIZON_MINUTES) if self.is_model_available else DEFAULT_HORIZON_MINUTES
        features_count = len(self._feature_cols) if (self.is_model_available and self._feature_cols) else 0

        return {
            "available": self.is_model_available,
            "status": "ready" if self.is_model_available else "model_unavailable",
            "model_name": model_name,
            "horizon_minutes": horizon,
            "threshold": CONGESTION_THRESHOLD,
            "capacity_mbps": LINK_CAPACITY_MBPS,
            "simulated": True,
            "data_source": "SIMULATED_BOTTLENECK_LINK",
            "dataset_loaded": self.is_data_available,
            "total_samples": len(self._df_clean) if self.is_data_available else 0,
            "features_count": features_count,
            "default_as_of": DEFAULT_AS_OF_TIMESTAMP,
        }

    def _resolve_as_of_index(self, as_of: Optional[str | int]) -> int:
        """
        Validate and resolve as_of parameter (index int or timestamp string).
        Raises ValueError for invalid, non-existent, or out-of-bounds parameters.
        """
        if not self.is_data_available or self._df_clean is None:
            raise RuntimeError("Congestion replay dataset is unavailable.")

        df = self._df_clean
        n = len(df)

        if as_of is None or str(as_of).strip() == "":
            # Default to DEFAULT_AS_OF_TIMESTAMP inside the Test partition
            matches = df.index[df["timestamp"] == DEFAULT_AS_OF_TIMESTAMP].tolist()
            if matches:
                return matches[0]
            # Fallback to 85% into dataset
            return int(n * 0.85)

        # 1. Try integer index
        try:
            idx = int(as_of)
            if idx < 0 or idx >= n:
                raise ValueError(f"as_of index {idx} out of bounds [0, {n - 1}].")
            if idx < MIN_LOOKBACK_MINUTES:
                raise ValueError(
                    f"as_of index {idx} has insufficient lookback history (must be >= {MIN_LOOKBACK_MINUTES})."
                )
            return idx
        except (ValueError, TypeError) as err:
            if isinstance(err, ValueError) and ("out of bounds" in str(err) or "insufficient" in str(err)):
                raise

        # 2. Try timestamp string
        as_of_str = str(as_of).strip()
        matches = df.index[df["timestamp"] == as_of_str].tolist()
        if matches:
            idx = matches[0]
            if idx < MIN_LOOKBACK_MINUTES:
                raise ValueError(
                    f"Timestamp '{as_of_str}' at index {idx} has insufficient lookback history (must be >= {MIN_LOOKBACK_MINUTES})."
                )
            return idx

        # Try flexible datetime matching
        try:
            target_dt = pd.to_datetime(as_of_str)
            df_dt = pd.to_datetime(df["timestamp"])
            # Exact match
            dt_matches = df.index[df_dt == target_dt].tolist()
            if dt_matches:
                idx = dt_matches[0]
                if idx < MIN_LOOKBACK_MINUTES:
                    raise ValueError(
                        f"Timestamp '{as_of_str}' at index {idx} has insufficient lookback history."
                    )
                return idx
            # Nearest match within 5 minutes
            diffs = (df_dt - target_dt).abs()
            min_diff = diffs.min()
            if min_diff <= pd.Timedelta(minutes=5):
                idx = int(diffs.idxmin())
                if idx < MIN_LOOKBACK_MINUTES:
                    raise ValueError(
                        f"Timestamp '{as_of_str}' at index {idx} has insufficient lookback history."
                    )
                return idx
            raise ValueError(f"Timestamp '{as_of_str}' is outside available dataset range.")
        except Exception as exc:
            raise ValueError(f"Invalid as_of timestamp or index '{as_of}': {exc}") from exc

    def get_forecast(self, as_of: Optional[str | int] = None) -> Dict[str, Any]:
        """
        Compute forecast and return recent observations for the given as_of point.
        """
        if not self.is_model_available:
            raise RuntimeError("Congestion prediction model is currently unavailable.")

        if not self.is_data_available or self._df_clean is None:
            raise RuntimeError("Congestion replay dataset is currently unavailable.")

        idx = self._resolve_as_of_index(as_of)
        df = self._df_clean

        row = df.iloc[idx]
        current_ts = str(row["timestamp"])
        current_u = float(row["utilization"])

        # 1. Recent observed utilization (up to 30 trailing minutes ending at idx)
        start_idx = max(0, idx - 29)
        recent_slice = df.iloc[start_idx : idx + 1]
        recent_observed = [
            {
                "timestamp": str(r["timestamp"]),
                "utilization": round(float(r["utilization"]), 4),
                "utilization_pct": round(float(r["utilization"]) * 100, 2),
                "queue_delay_ms": round(float(r.get("queue_delay_ms", 2.5)), 2),
                "flow_count": int(r.get("flow_count", 0)),
                "packets_per_sec": round(float(r.get("packets_per_sec", float(r["utilization"]) * 1000)), 1),
                "bytes_per_sec": round(float(r.get("bytes_per_sec", float(r["utilization"]) * 12_500_000)), 1),
            }
            for _, r in recent_slice.iterrows()
        ]

        # 2. Extract feature vector at forecast time t
        feature_cols = self._feature_cols or []
        features_dict = {col: float(row[col]) for col in feature_cols if col in row}
        X_input = np.array([[features_dict[col] for col in feature_cols]])

        # 3. Model prediction
        model = self._model_bundle["model"]
        pred_raw = float(model.predict(X_input)[0])
        pred_utilization = round(float(np.clip(pred_raw, 0.0, 1.0)), 4)

        # Forecast timestamp is t + 5 minutes
        current_dt = pd.to_datetime(current_ts)
        forecast_dt = current_dt + pd.Timedelta(minutes=DEFAULT_HORIZON_MINUTES)
        forecast_ts = forecast_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 4. Baselines
        persistence_val = round(current_u, 4)
        moving_avg_val = round(float(row.get("utilization_roll_mean_5", current_u)), 4)

        # 5. Alert flag
        alert = bool(pred_utilization >= CONGESTION_THRESHOLD)

        return {
            "simulated": True,
            "data_source": "SIMULATED_BOTTLENECK_LINK",
            "model_name": self._model_bundle.get("champion_name", "Ridge"),
            "model_version": self._model_bundle.get("version", "1.0.0-h5"),
            "horizon_minutes": DEFAULT_HORIZON_MINUTES,
            "threshold": CONGESTION_THRESHOLD,
            "capacity_mbps": LINK_CAPACITY_MBPS,
            "as_of": current_ts,
            "as_of_index": idx,
            "as_of_timestamp": current_ts,
            "current_utilization": persistence_val,
            "forecast_timestamp": forecast_ts,
            "forecast_utilization": pred_utilization,
            "persistence_baseline": persistence_val,
            "moving_average_baseline": moving_avg_val,
            "alert": alert,
            "forecast": {
                "target_timestamp": forecast_ts,
                "predicted_utilization": pred_utilization,
                "predicted_utilization_pct": round(pred_utilization * 100, 2),
                "alert": alert,
                "persistence_baseline": persistence_val,
                "persistence_baseline_pct": round(persistence_val * 100, 2),
                "moving_average_5m": moving_avg_val,
                "moving_average_5m_pct": round(moving_avg_val * 100, 2),
            },
            "recent_observed": recent_observed,
        }


# Global singleton instance for use by Flask routes
_service_instance: Optional[CongestionService] = None


def get_congestion_service() -> CongestionService:
    """Return singleton CongestionService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CongestionService()
    return _service_instance
