"""
Train and Save Final H=5 Congestion Prediction Model — NetSentinel AI
=====================================================================
Saves final h=5 model artifacts under models/model2_congestion/h5/
without overwriting existing h=1 artifacts.
Candidate model is selected strictly based on minimum VALIDATION MAE.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.congestion.evaluate_multi_horizon import (
    build_multi_horizon_dataset,
    evaluate_alert_classification,
    evaluate_regression,
    split_chronological,
)
from ml.congestion.features import CONGESTION_THRESHOLD

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_DATA_PATH = Path("data/processed/simulated_congestion_1min.csv")
TARGET_H5_DIR = Path("models/model2_congestion/h5")


def train_and_save_h5(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = TARGET_H5_DIR,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Train h=5 model and save artifacts to target directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    raw_df = pd.read_csv(data_file)
    clean_df, feature_cols, reg_targets, cls_targets = build_multi_horizon_dataset(
        raw_df, horizons=[5], threshold=CONGESTION_THRESHOLD
    )

    train_df, val_df, test_df = split_chronological(clean_df, train_frac=0.70, val_frac=0.15)

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    target_col = reg_targets[5]
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # 1. Baseline evaluations
    curr_idx = feature_cols.index("utilization_current")
    ma_idx = feature_cols.index("utilization_roll_mean_5")

    y_pred_persist_train = X_train[:, curr_idx]
    y_pred_persist_val = X_val[:, curr_idx]
    y_pred_persist_test = X_test[:, curr_idx]

    y_pred_ma_train = X_train[:, ma_idx]
    y_pred_ma_val = X_val[:, ma_idx]
    y_pred_ma_test = X_test[:, ma_idx]

    # 2. Tune Ridge Regression on Validation MAE
    best_ridge_alpha = 1.0
    best_ridge_val_mae = float("inf")
    best_ridge_pipe = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=seed)),
            ]
        )
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        if val_mae < best_ridge_val_mae:
            best_ridge_val_mae = val_mae
            best_ridge_alpha = alpha
            best_ridge_pipe = pipe

    # 3. Tune Random Forest on Validation MAE
    best_rf_depth = 6
    best_rf_val_mae = float("inf")
    best_rf_model = None

    for depth in [6, 8, 12]:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        val_pred = rf.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        if val_mae < best_rf_val_mae:
            best_rf_val_mae = val_mae
            best_rf_depth = depth
            best_rf_model = rf

    # 4. Champion selection based on Validation MAE
    if best_rf_val_mae < best_ridge_val_mae:
        champ_name = "RandomForest"
        champ_model = best_rf_model
        champ_val_mae = best_rf_val_mae
    else:
        champ_name = "Ridge"
        champ_model = best_ridge_pipe
        champ_val_mae = best_ridge_val_mae

    logger.info("Selected h=5 Champion based on Validation MAE: %s (Val MAE=%.5f)", champ_name, champ_val_mae)

    # 5. Evaluate on test set
    y_pred_test = champ_model.predict(X_test)
    test_reg_metrics = evaluate_regression(y_test, y_pred_test)
    test_alert_metrics = evaluate_alert_classification(y_test, y_pred_test, threshold=CONGESTION_THRESHOLD)

    persist_test_reg = evaluate_regression(y_test, y_pred_persist_test)
    persist_test_alert = evaluate_alert_classification(y_test, y_pred_persist_test, threshold=CONGESTION_THRESHOLD)

    ma_test_reg = evaluate_regression(y_test, y_pred_ma_test)
    ma_test_alert = evaluate_alert_classification(y_test, y_pred_ma_test, threshold=CONGESTION_THRESHOLD)

    metrics_payload = {
        "metadata": {
            "model_name": champ_name,
            "horizon_minutes": 5,
            "simulated": True,
            "data_source": "SIMULATED_BOTTLENECK_LINK",
            "link_capacity_mbps": 100.0,
            "congestion_threshold": CONGESTION_THRESHOLD,
            "seed": seed,
            "partition_sizes": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
            },
            "features": feature_cols,
        },
        "validation_selection": {
            "ridge_val_mae": round(float(best_ridge_val_mae), 5),
            "ridge_best_alpha": best_ridge_alpha,
            "rf_val_mae": round(float(best_rf_val_mae), 5),
            "rf_best_depth": best_rf_depth,
            "selected_model": champ_name,
        },
        "test_performance": {
            "champion": {
                "name": champ_name,
                "regression": test_reg_metrics,
                "alert": test_alert_metrics,
            },
            "persistence": {
                "regression": persist_test_reg,
                "alert": persist_test_alert,
            },
            "moving_average_5min": {
                "regression": ma_test_reg,
                "alert": ma_test_alert,
            },
        },
    }

    # 6. Save model bundle
    model_bundle = {
        "model": champ_model,
        "champion_name": champ_name,
        "horizon_minutes": 5,
        "feature_cols": feature_cols,
        "congestion_threshold": CONGESTION_THRESHOLD,
        "link_capacity_mbps": 100.0,
        "simulated": True,
        "version": "1.0.0-h5",
    }
    model_path = out_dir / "model.joblib"
    joblib.dump(model_bundle, model_path)

    # 7. Save feature schema
    schema_payload = {
        "dataset_type": "SIMULATED_BOTTLENECK_LINK",
        "horizon_minutes": 5,
        "simulated": True,
        "link_capacity_mbps": 100.0,
        "feature_names": feature_cols,
        "feature_count": len(feature_cols),
        "target_regression": "target_utilization_h5",
        "target_classification": "congestion_h5",
        "congestion_threshold": CONGESTION_THRESHOLD,
        "feature_descriptions": {
            "utilization_current": "Bottleneck link utilization at current time t [0.0, 1.0]",
            "utilization_lag_1": "Utilization 1 minute prior (t-1)",
            "utilization_lag_2": "Utilization 2 minutes prior (t-2)",
            "utilization_lag_3": "Utilization 3 minutes prior (t-3)",
            "utilization_roll_mean_5": "Rolling mean utilization over trailing 5 minutes",
            "utilization_roll_std_5": "Rolling std of utilization over trailing 5 minutes",
            "utilization_roll_mean_15": "Rolling mean utilization over trailing 15 minutes",
            "utilization_roll_std_15": "Rolling std of utilization over trailing 15 minutes",
            "utilization_velocity": "First-order difference U_t - U_{t-1} (surge velocity)",
            "utilization_acceleration": "Second-order difference Delta U_t - Delta U_{t-1} (surge acceleration)",
            "flow_count": "Active concurrent connection count at time t (optional)",
            "udp_share": "Fraction of offered load using UDP protocol at time t (optional)",
        },
    }
    schema_path = out_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2)

    # 8. Save metrics.json
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


def main() -> None:
    results = train_and_save_h5()
    print("Successfully trained and saved H=5 model artifacts to models/model2_congestion/h5/")
    print(f"Selected Champion Model: {results['metadata']['model_name']}")
    print(f"Test MAE: {results['test_performance']['champion']['regression']['mae']:.5f}")
    print(f"Test RMSE: {results['test_performance']['champion']['regression']['rmse']:.5f}")
    print(f"Test Alert F1: {results['test_performance']['champion']['alert']['f1']:.4f}")


if __name__ == "__main__":
    main()
