"""
Time-Series Feature Engineering & Target Preparation — NetSentinel AI Phase 2
=============================================================================
Transforms raw/simulated 1-minute network traffic telemetry into temporal
features for predicting next-step bottleneck link utilization and congestion alerts.

LEAKAGE-FREE DESIGN PRINCIPLES:
-------------------------------
1. Information at time t uses ONLY observations at or prior to time t:
   - Current utilization: U_t
   - Utilization lags: U_{t-1}, U_{t-2}, U_{t-3}
   - Rolling statistics over past 5 and 15 minutes (closed='right', strictly <= t)
   - Velocity: U_t - U_{t-1}
   - Acceleration: (U_t - U_{t-1}) - (U_{t-1} - U_{t-2})
   - Auxiliary features at time t: flow_count, udp_share (when present)
2. Targets are strictly FUTURE observations (used as training/evaluation labels only):
   - Regression target: target_utilization_next = U_{t+1} (shift -1)
   - Classification target: congestion_next = 1 if U_{t+1} >= 0.80 else 0
3. Target columns are NEVER included in the input feature set (X).
4. Records with incomplete history (initial rolling window) or missing future target
   (last record) are cleanly dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names
COL_TIMESTAMP = "timestamp"
COL_UTILIZATION = "utilization"
COL_FLOW_COUNT = "flow_count"
COL_UDP_SHARE = "udp_share"

# Target column names
TARGET_REGRESSION = "target_utilization_next"
TARGET_CLASSIFICATION = "congestion_next"
CONGESTION_THRESHOLD = 0.80

# Base required columns
REQUIRED_COLUMNS = [COL_TIMESTAMP, COL_UTILIZATION]


def create_features(
    df: pd.DataFrame,
    drop_na: bool = True,
    congestion_threshold: float = CONGESTION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate past/current temporal features and future prediction targets.

    Parameters
    ----------
    df : pd.DataFrame
        Input telemetry containing at least 'timestamp' and 'utilization'.
    drop_na : bool, default True
        Whether to drop rows with NaN values resulting from rolling windows/lags
        and the final shifted target row.
    congestion_threshold : float, default 0.80
        Threshold on future utilization for binary congestion alert.

    Returns
    -------
    df_features : pd.DataFrame
        Transformed DataFrame with engineered features and target columns.
    feature_cols : list[str]
        List of input feature column names (strictly past/current only).
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    # Validate required columns
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_required}")

    # Copy and ensure chronological ordering
    data = df.copy()
    data[COL_TIMESTAMP] = pd.to_datetime(data[COL_TIMESTAMP])
    data = data.sort_values(COL_TIMESTAMP).reset_index(drop=True)

    # -------------------------------------------------------------
    # 1. Past / Current Temporal Features (Available at or before t)
    # -------------------------------------------------------------
    u = data[COL_UTILIZATION].astype(float)

    # Current value at time t
    data["utilization_current"] = u

    # Autoregressive lags (strictly prior time steps)
    data["utilization_lag_1"] = u.shift(1)
    data["utilization_lag_2"] = u.shift(2)
    data["utilization_lag_3"] = u.shift(3)

    # Rolling window metrics over past 5 and 15 minutes (closed='right' includes t)
    # min_periods ensures we only compute when full window is present
    data["utilization_roll_mean_5"] = u.rolling(window=5, min_periods=5).mean()
    data["utilization_roll_std_5"] = u.rolling(window=5, min_periods=5).std()
    data["utilization_roll_mean_15"] = u.rolling(window=15, min_periods=15).mean()
    data["utilization_roll_std_15"] = u.rolling(window=15, min_periods=15).std()

    # Rate of change: Velocity and Acceleration
    data["utilization_velocity"] = u.diff(1)
    data["utilization_acceleration"] = data["utilization_velocity"].diff(1)

    # Feature column tracking
    feature_cols = [
        "utilization_current",
        "utilization_lag_1",
        "utilization_lag_2",
        "utilization_lag_3",
        "utilization_roll_mean_5",
        "utilization_roll_std_5",
        "utilization_roll_mean_15",
        "utilization_roll_std_15",
        "utilization_velocity",
        "utilization_acceleration",
    ]

    # Optional traffic features if available in input
    if COL_FLOW_COUNT in data.columns:
        data[COL_FLOW_COUNT] = data[COL_FLOW_COUNT].astype(float)
        feature_cols.append(COL_FLOW_COUNT)

    if COL_UDP_SHARE in data.columns:
        data[COL_UDP_SHARE] = data[COL_UDP_SHARE].astype(float)
        feature_cols.append(COL_UDP_SHARE)

    # -------------------------------------------------------------
    # 2. Future Target Variables (Used ONLY as ground-truth labels)
    # -------------------------------------------------------------
    # Next-minute utilization at t+1 (strictly future observation)
    data[TARGET_REGRESSION] = u.shift(-1)

    # Next-minute binary congestion alert (1 if U_{t+1} >= threshold else 0)
    data[TARGET_CLASSIFICATION] = (data[TARGET_REGRESSION] >= congestion_threshold).astype(int)

    # If drop_na is True, drop rows where any feature or target is NaN
    # (first 14 rows due to 15-min rolling window; last row due to shift -1)
    if drop_na:
        all_relevant = feature_cols + [TARGET_REGRESSION, TARGET_CLASSIFICATION]
        before = len(data)
        data = data.dropna(subset=all_relevant).reset_index(drop=True)
        dropped = before - len(data)
        logger.debug("create_features: dropped %d rows with boundary NaNs.", dropped)

    return data, feature_cols


def load_and_prepare_features(
    csv_path: str | Path,
    drop_na: bool = True,
    congestion_threshold: float = CONGESTION_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV from disk and create features."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    return create_features(df, drop_na=drop_na, congestion_threshold=congestion_threshold)
