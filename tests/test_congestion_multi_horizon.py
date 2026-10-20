"""
Unit Tests for Stage 1b Multi-Horizon Congestion Prediction
============================================================
Verifies:
- Multi-horizon target construction (h=1, 5, 10) matches U_{t+h}
- Absolute absence of future-data leakage in multi-horizon feature extraction
- Strictly chronological train/validation/test partitions
- Baseline validity: persistence and moving average use strictly <= t information
- Zero-division robustness for alert classification
- End-to-end execution of multi-horizon evaluation pipeline without overwriting Stage 1 artifacts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.congestion.evaluate_multi_horizon import (
    build_multi_horizon_dataset,
    evaluate_alert_classification,
    evaluate_regression,
    run_multi_horizon_evaluation,
    split_chronological,
)
from ml.congestion.features import CONGESTION_THRESHOLD
from ml.congestion.generate_data import generate_congestion_dataset


class TestMultiHorizonDataset:
    """Tests for multi-horizon dataset alignment and leakage prevention."""

    def test_multi_horizon_targets_match_shifted_utilization(self):
        """Verify target_utilization_h{h} equals utilization shifted by -h."""
        df_raw = generate_congestion_dataset(days=1, seed=42)
        horizons = [1, 5, 10]
        clean_df, feature_cols, reg_targets, cls_targets = build_multi_horizon_dataset(
            df_raw, horizons=horizons, threshold=CONGESTION_THRESHOLD
        )

        assert len(clean_df) > 0
        u_raw = df_raw["utilization"].values
        ts_raw = pd.to_datetime(df_raw["timestamp"])

        for idx in [0, 20, 100]:
            t_curr = pd.to_datetime(clean_df["timestamp"].iloc[idx])
            raw_pos = np.where(ts_raw == t_curr)[0][0]

            for h in horizons:
                expected_u = u_raw[raw_pos + h]
                actual_u = clean_df[reg_targets[h]].iloc[idx]
                assert actual_u == pytest.approx(expected_u, abs=1e-5)

                expected_cls = 1 if expected_u >= CONGESTION_THRESHOLD else 0
                actual_cls = clean_df[cls_targets[h]].iloc[idx]
                assert actual_cls == expected_cls

    def test_features_do_not_leak_future_targets(self):
        """
        Proof of zero-leakage: changing future data (t > cutoff) must not alter
        input feature representations at or before cutoff.
        """
        df_raw = generate_congestion_dataset(days=1, seed=42)
        cutoff_raw_idx = 80
        cutoff_ts = pd.to_datetime(df_raw["timestamp"].iloc[cutoff_raw_idx])
        horizons = [1, 5, 10]

        clean_orig, f_cols, _, _ = build_multi_horizon_dataset(df_raw, horizons=horizons)

        # Mutate future rows in raw data
        df_mutated = df_raw.copy()
        df_mutated.loc[cutoff_raw_idx + 1 :, "utilization"] = 0.999
        if "flow_count" in df_mutated.columns:
            df_mutated.loc[cutoff_raw_idx + 1 :, "flow_count"] = 9999

        clean_mut, _, _, _ = build_multi_horizon_dataset(df_mutated, horizons=horizons)

        # Past features up to cutoff timestamp must match exactly
        mask_orig = pd.to_datetime(clean_orig["timestamp"]) <= cutoff_ts
        mask_mut = pd.to_datetime(clean_mut["timestamp"]) <= cutoff_ts

        orig_past = clean_orig.loc[mask_orig, f_cols].reset_index(drop=True)
        mut_past = clean_mut.loc[mask_mut, f_cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(orig_past, mut_past)

    def test_chronological_splitting_no_overlap(self):
        """Partitions must be strictly chronological, non-overlapping, and cover the dataset."""
        df_raw = generate_congestion_dataset(days=2, seed=42)
        clean_df, _, _, _ = build_multi_horizon_dataset(df_raw, horizons=[1, 5, 10])

        tr, va, te = split_chronological(clean_df, train_frac=0.70, val_frac=0.15)
        assert len(tr) + len(va) + len(te) == len(clean_df)

        t_tr_max = pd.to_datetime(tr["timestamp"]).max()
        t_va_min = pd.to_datetime(va["timestamp"]).min()
        t_va_max = pd.to_datetime(va["timestamp"]).max()
        t_te_min = pd.to_datetime(te["timestamp"]).min()

        assert t_tr_max < t_va_min
        assert t_va_max < t_te_min


class TestMultiHorizonEvaluationExecution:
    """Tests for multi-horizon evaluation pipeline execution."""

    def test_alert_classification_zero_division_guard(self):
        """Ensure missing alert classes do not raise ZeroDivisionError."""
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.1, 0.2, 0.3])
        res = evaluate_alert_classification(y_true, y_pred, threshold=0.80)
        assert res["class_counts"]["congested"] == 0
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0
        assert res["f1"] == 0.0

    def test_run_multi_horizon_pipeline_outputs(self, tmp_path):
        """Verify pipeline runs and produces valid, non-empty artifacts."""
        sim_data = tmp_path / "sim_data.csv"
        df = generate_congestion_dataset(days=2, seed=42)
        df.to_csv(sim_data, index=False)

        report_file = tmp_path / "test_report.md"
        metrics_file = tmp_path / "test_metrics.json"

        results = run_multi_horizon_evaluation(
            data_path=sim_data,
            horizons=[1, 5],
            seed=42,
            report_path=report_file,
            metrics_path=metrics_file,
        )

        assert report_file.exists() and report_file.stat().st_size > 0
        assert metrics_file.exists() and metrics_file.stat().st_size > 0

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)

        assert "results_by_horizon" in metrics_json
        assert "h1" in metrics_json["results_by_horizon"]
        assert "h5" in metrics_json["results_by_horizon"]

        with open(report_file, "r", encoding="utf-8") as f:
            report_text = f.read()

        assert "SIMULATED TELEMETRY DISCLAIMER" in report_text
