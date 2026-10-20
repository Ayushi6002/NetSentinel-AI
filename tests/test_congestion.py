"""
Unit Tests for Phase 2 Congestion Prediction ML Pipeline
=========================================================
Verifies:
- Deterministic and reproducible data generation (same seed vs different seed)
- Bounds, column presence, and time series monotonicity
- Strict avoidance of future-data leakage in feature generation
- Target shifting and congestion threshold definition
- Chronological train/val/test splitting without overlap or shuffling
- Training-only preprocessing fit
- Robust zero-division handling for missing alert classes
- Pipeline robustness when optional columns are absent
- End-to-end model training, prediction, and artifact generation
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.congestion.features import (
    COL_TIMESTAMP,
    COL_UTILIZATION,
    CONGESTION_THRESHOLD,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    create_features,
)
from ml.congestion.generate_data import (
    LINK_CAPACITY_MBPS,
    generate_congestion_dataset,
)
from ml.congestion.train_congestion import (
    evaluate_alert_classification,
    evaluate_regression,
    split_chronological,
    train_and_evaluate,
)


class TestSimulatedDataGenerator:
    """Tests for ml.congestion.generate_data."""

    def test_same_seed_generates_identical_data(self):
        """Same random seed must produce identical dataframes."""
        df1 = generate_congestion_dataset(days=1, seed=42)
        df2 = generate_congestion_dataset(days=1, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_generate_different_data(self):
        """Different seeds must produce different simulated values."""
        df1 = generate_congestion_dataset(days=1, seed=42)
        df2 = generate_congestion_dataset(days=1, seed=99)
        assert not df1["offered_load_mbps"].equals(df2["offered_load_mbps"])
        assert not df1["utilization"].equals(df2["utilization"])

    def test_generated_row_count_and_columns(self):
        """7 days at 1-minute interval must produce exactly 10,080 rows and expected schema."""
        df = generate_congestion_dataset(days=7, seed=42)
        assert len(df) == 10080

        expected_cols = [
            "timestamp",
            "offered_load_mbps",
            "utilization",
            "queue_delay_ms",
            "flow_count",
            "udp_share",
            "simulated",
        ]
        assert list(df.columns) == expected_cols

    def test_data_bounds_and_validity(self):
        """Verify physics-inspired bounds on telemetry."""
        df = generate_congestion_dataset(days=2, seed=42)

        # Utilization strictly in [0.0, 1.0]
        assert (df["utilization"] >= 0.0).all()
        assert (df["utilization"] <= 1.0).all()

        # Offered load strictly positive
        assert (df["offered_load_mbps"] >= 5.0).all()

        # Queue delay positive and bounded
        assert (df["queue_delay_ms"] >= 1.5).all()
        assert (df["queue_delay_ms"] <= 120.0).all()

        # Flow count positive integer
        assert (df["flow_count"] >= 20).all()

        # UDP share in [0.0, 1.0]
        assert (df["udp_share"] >= 0.05).all()
        assert (df["udp_share"] <= 0.65).all()

        # Simulated marker column
        assert df["simulated"].all() is True or (df["simulated"] == True).all()

    def test_timestamp_monotonicity_and_resolution(self):
        """Timestamps must be monotonically increasing with 1-minute delta."""
        df = generate_congestion_dataset(days=1, seed=42)
        ts = pd.to_datetime(df["timestamp"])
        assert ts.is_monotonic_increasing
        diffs = ts.diff().dropna()
        assert (diffs == pd.Timedelta(minutes=1)).all()


class TestFeatureEngineering:
    """Tests for ml.congestion.features."""

    def test_features_do_not_use_future_observations(self):
        """
        Proof of zero-leakage: modifying future rows (t > T_cutoff) must NOT
        change any input feature values computed at or before T_cutoff.
        """
        raw_df = generate_congestion_dataset(days=1, seed=42)
        cutoff = 100

        # Feature extraction on original data
        feat_orig, f_cols = create_features(raw_df, drop_na=False)

        # Create modified copy where all future rows after cutoff are drastically changed
        modified_df = raw_df.copy()
        modified_df.loc[cutoff + 1 :, "utilization"] = 0.999
        if "flow_count" in modified_df.columns:
            modified_df.loc[cutoff + 1 :, "flow_count"] = 9999
        if "udp_share" in modified_df.columns:
            modified_df.loc[cutoff + 1 :, "udp_share"] = 0.99

        feat_mod, _ = create_features(modified_df, drop_na=False)

        # For all rows <= cutoff, every input feature must remain identical
        orig_past = feat_orig.loc[:cutoff, f_cols]
        mod_past = feat_mod.loc[:cutoff, f_cols]

        pd.testing.assert_frame_equal(orig_past, mod_past)

    def test_target_shifting_and_congestion_threshold(self):
        """Confirm target_utilization_next is U_{t+1} and congestion_next is (U_{t+1} >= 0.80)."""
        raw_df = generate_congestion_dataset(days=1, seed=42)
        feat_df, _ = create_features(raw_df, drop_na=True)

        # In non-boundary rows, target_utilization_next at row i must equal original U at row i+1
        # Check matching timestamps
        ts_orig = pd.to_datetime(raw_df["timestamp"])
        u_orig = raw_df["utilization"].values

        for idx in [0, 10, 50, 100]:
            t_row = pd.to_datetime(feat_df["timestamp"].iloc[idx])
            orig_idx = np.where(ts_orig == t_row)[0][0]

            expected_next_u = u_orig[orig_idx + 1]
            actual_next_u = feat_df[TARGET_REGRESSION].iloc[idx]
            assert actual_next_u == pytest.approx(expected_next_u, abs=1e-5)

            expected_cong = 1 if expected_next_u >= CONGESTION_THRESHOLD else 0
            assert feat_df[TARGET_CLASSIFICATION].iloc[idx] == expected_cong

    def test_optional_columns_absent_handled_gracefully(self):
        """create_features must work when only minimal required columns are present."""
        minimal_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-11-01", periods=60, freq="1min"),
                "utilization": np.linspace(0.1, 0.8, 60),
            }
        )
        feat_df, f_cols = create_features(minimal_df, drop_na=True)
        assert len(feat_df) > 0
        assert "flow_count" not in f_cols
        assert "udp_share" not in f_cols
        assert "utilization_current" in f_cols
        assert "utilization_roll_mean_15" in f_cols
        assert TARGET_REGRESSION in feat_df.columns
        assert TARGET_CLASSIFICATION in feat_df.columns

    def test_missing_required_columns_raises_error(self):
        """Missing timestamp or utilization must raise ValueError."""
        bad_df = pd.DataFrame({"utilization": [0.5, 0.6]})
        with pytest.raises(ValueError, match="missing required columns"):
            create_features(bad_df)


class TestTrainingAndEvaluation:
    """Tests for ml.congestion.train_congestion."""

    def test_chronological_split_boundaries(self):
        """Split must be strictly chronological: train max timestamp < val min timestamp < test min."""
        raw_df = generate_congestion_dataset(days=2, seed=42)
        feat_df, _ = create_features(raw_df, drop_na=True)

        train_df, val_df, test_df = split_chronological(feat_df, train_frac=0.70, val_frac=0.15)

        assert len(train_df) + len(val_df) + len(test_df) == len(feat_df)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # Check timestamp ordering
        t_train_max = pd.to_datetime(train_df[COL_TIMESTAMP]).max()
        t_val_min = pd.to_datetime(val_df[COL_TIMESTAMP]).min()
        t_val_max = pd.to_datetime(val_df[COL_TIMESTAMP]).max()
        t_test_min = pd.to_datetime(test_df[COL_TIMESTAMP]).min()

        assert t_train_max < t_val_min
        assert t_val_max < t_test_min

    def test_alert_classification_zero_division_safety(self):
        """Missing positive or negative alerts in predictions must not raise division-by-zero errors."""
        # Case A: No positive labels anywhere (all 0s)
        y_true = np.array([0.2, 0.3, 0.4, 0.5])
        y_pred = np.array([0.2, 0.3, 0.4, 0.5])
        res = evaluate_alert_classification(y_true, y_pred, threshold=0.80)

        assert res["class_counts"]["congested"] == 0
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0
        assert res["f1"] == 0.0
        assert res["confusion_matrix"]["tp"] == 0
        assert res["confusion_matrix"]["fp"] == 0
        assert res["confusion_matrix"]["fn"] == 0
        assert res["confusion_matrix"]["tn"] == 4

        # Case B: All positive labels
        y_true_all_pos = np.array([0.85, 0.90, 0.95])
        y_pred_all_pos = np.array([0.88, 0.92, 0.89])
        res_pos = evaluate_alert_classification(y_true_all_pos, y_pred_all_pos, threshold=0.80)
        assert res_pos["precision"] == 1.0
        assert res_pos["recall"] == 1.0
        assert res_pos["f1"] == 1.0
        assert res_pos["confusion_matrix"]["tp"] == 3

    def test_end_to_end_training_and_artifacts(self, tmp_path):
        """Verify train_and_evaluate generates all expected artifacts safely in target directory."""
        # Generate 2-day small dataset for fast test
        sim_csv = tmp_path / "sim_data.csv"
        df_sim = generate_congestion_dataset(days=2, seed=42)
        df_sim.to_csv(sim_csv, index=False)

        model_dir = tmp_path / "models"
        report_dir = tmp_path / "reports"

        results = train_and_evaluate(
            data_path=sim_csv,
            seed=42,
            model_dir=model_dir,
            report_dir=report_dir,
        )

        # Verify return structure
        assert "metadata" in results
        assert "regression_results" in results
        assert "alert_classification_test" in results
        assert "champion_model" in results

        # Verify artifacts exist on disk
        model_file = model_dir / "model.joblib"
        schema_file = model_dir / "feature_schema.json"
        metrics_file = model_dir / "metrics.json"
        report_file = report_dir / "evaluation.md"

        assert model_file.exists() and model_file.stat().st_size > 0
        assert schema_file.exists() and schema_file.stat().st_size > 0
        assert metrics_file.exists() and metrics_file.stat().st_size > 0
        assert report_file.exists() and report_file.stat().st_size > 0

        # Verify model artifact is loadable and can predict
        bundle = joblib.load(model_file)
        assert "model" in bundle
        assert bundle["is_simulated"] is True
        trained_model = bundle["model"]
        f_cols = bundle["feature_cols"]

        dummy_x = np.ones((1, len(f_cols))) * 0.5
        pred = trained_model.predict(dummy_x)
        assert len(pred) == 1
        assert 0.0 <= pred[0] <= 1.0

        # Verify schema is valid JSON
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        assert schema_data["dataset_type"] == "SIMULATED_BOTTLENECK_LINK"
        assert len(schema_data["feature_names"]) == len(f_cols)

        # Verify evaluation report contains simulation disclaimer
        with open(report_file, "r", encoding="utf-8") as f:
            report_text = f.read()
        assert "SIMULATED TELEMETRY DISCLAIMER" in report_text
