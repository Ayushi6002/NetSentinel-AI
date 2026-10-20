"""
Model Training & Evaluation Pipeline — NetSentinel AI Phase 2
=============================================================
Trains and evaluates predictive models for bottleneck link utilization and
congestion alerts using strictly chronological train/validation/test splits.

SCIENTIFIC & ETHICAL DISCLAIMER:
--------------------------------
All models, features, and evaluation metrics in this pipeline are derived from
SIMULATED TELEMETRY. They establish algorithmic feasibility, feature stability,
and pipeline correctness, but do NOT prove real-world network congestion prediction accuracy.

LEAKAGE-FREE CHRONOLOGICAL SPLIT:
---------------------------------
1. Chronological Partition:
   - First 70%: Training (historical baseline)
   - Next 15%: Validation (model selection / hyperparameter tuning)
   - Final 15%: Test (untouched, unbiased out-of-sample evaluation)
2. No shuffling or random K-fold splits are permitted in time-series evaluation.
3. Feature preprocessing (e.g. StandardScaler for Ridge) is fit exclusively on the training
   partition and applied to validation and test sets via scikit-learn Pipeline.
4. Model selection is performed strictly on validation metrics; test data is evaluated once.

EVALUATED APPROACHES:
---------------------
1. Persistence Baseline: Predicts U_{t+1} = U_t (current utilization).
2. Moving-Average Baseline: Predicts U_{t+1} = 5-minute rolling mean of past utilization.
3. Ridge Regression: L2-regularized linear model with scaled features (tuned on validation).
4. Random Forest Regressor: Non-linear tree ensemble capturing non-linear burst dynamics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.congestion.features import (
    COL_TIMESTAMP,
    COL_UTILIZATION,
    CONGESTION_THRESHOLD,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    load_and_prepare_features,
)

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_MODEL_DIR = Path("models/model2_congestion")
DEFAULT_REPORT_DIR = Path("reports/congestion")


def split_chronological(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series DataFrame into strictly chronological train, val, and test partitions.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate MAE and RMSE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": round(mae, 5), "rmse": round(rmse, 5)}


def evaluate_alert_classification(
    y_true_util: np.ndarray,
    y_pred_util: np.ndarray,
    threshold: float = CONGESTION_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluate binary congestion alert classification based on thresholding utilization.
    Safely handles edge cases where a class is absent in y_true or y_pred.
    """
    y_true_bin = (y_true_util >= threshold).astype(int)
    y_pred_bin = (y_pred_util >= threshold).astype(int)

    # Class counts
    pos_count = int(np.sum(y_true_bin == 1))
    neg_count = int(np.sum(y_true_bin == 0))
    total_count = len(y_true_bin)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # labels=[0, 1] ensures 2x2 matrix even if one class is missing in test predictions
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Precision, Recall, F1 with explicit zero_division handling
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0.0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0.0))
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0.0))

    return {
        "threshold": threshold,
        "class_counts": {
            "total": total_count,
            "non_congested": neg_count,
            "congested": pos_count,
            "congested_pct": round(pos_count / total_count * 100, 2) if total_count > 0 else 0.0,
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def train_and_evaluate(
    data_path: str | Path = "data/processed/simulated_congestion_1min.csv",
    seed: int = DEFAULT_SEED,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> Dict[str, Any]:
    """
    Execute full training, tuning, and evaluation workflow.
    """
    model_dir_path = Path(model_dir)
    report_dir_path = Path(report_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    report_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data and engineer features
    logger.info("Loading data from %s and engineering features...", data_path)
    df, feature_cols = load_and_prepare_features(data_path)

    # 2. Chronological split
    train_df, val_df, test_df = split_chronological(df, train_frac=0.70, val_frac=0.15)
    logger.info(
        "Chronological split sizes: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(df),
    )

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_REGRESSION].values

    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_REGRESSION].values

    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_REGRESSION].values

    # -------------------------------------------------------------
    # 3. Model 1: Persistence Baseline (y_hat_{t+1} = U_t)
    # -------------------------------------------------------------
    # In feature_cols, 'utilization_current' is index 0
    curr_idx = feature_cols.index("utilization_current")
    y_pred_persist_train = X_train[:, curr_idx]
    y_pred_persist_val = X_val[:, curr_idx]
    y_pred_persist_test = X_test[:, curr_idx]

    # -------------------------------------------------------------
    # 4. Model 2: Moving Average Baseline (y_hat_{t+1} = 5-min rolling mean)
    # -------------------------------------------------------------
    ma_idx = feature_cols.index("utilization_roll_mean_5")
    y_pred_ma_train = X_train[:, ma_idx]
    y_pred_ma_val = X_val[:, ma_idx]
    y_pred_ma_test = X_test[:, ma_idx]

    # -------------------------------------------------------------
    # 5. Model 3: Ridge Regression with Hyperparameter Tuning
    # -------------------------------------------------------------
    best_ridge_alpha = 1.0
    best_ridge_val_mae = float("inf")
    best_ridge_pipeline = None

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
            best_ridge_pipeline = pipe

    logger.info("Best Ridge alpha: %s (Val MAE: %.5f)", best_ridge_alpha, best_ridge_val_mae)
    y_pred_ridge_train = best_ridge_pipeline.predict(X_train)
    y_pred_ridge_val = best_ridge_pipeline.predict(X_val)
    y_pred_ridge_test = best_ridge_pipeline.predict(X_test)

    # -------------------------------------------------------------
    # 6. Model 4: Random Forest Regressor with Validation Tuning
    # -------------------------------------------------------------
    best_rf_depth = 10
    best_rf_val_mae = float("inf")
    best_rf_model = None

    for max_depth in [6, 10, 14]:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        val_pred = rf.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        if val_mae < best_rf_val_mae:
            best_rf_val_mae = val_mae
            best_rf_depth = max_depth
            best_rf_model = rf

    logger.info("Best RF max_depth: %s (Val MAE: %.5f)", best_rf_depth, best_rf_val_mae)
    y_pred_rf_train = best_rf_model.predict(X_train)
    y_pred_rf_val = best_rf_model.predict(X_val)
    y_pred_rf_test = best_rf_model.predict(X_test)

    # Select overall champion model based on validation MAE
    if best_rf_val_mae < best_ridge_val_mae:
        champion_name = "RandomForest"
        champion_model = best_rf_model
        y_pred_champion_test = y_pred_rf_test
    else:
        champion_name = "Ridge"
        champion_model = best_ridge_pipeline
        y_pred_champion_test = y_pred_ridge_test

    logger.info("Selected Champion model based on Validation MAE: %s", champion_name)

    # -------------------------------------------------------------
    # 7. Collect Comprehensive Metrics
    # -------------------------------------------------------------
    metrics_summary: Dict[str, Any] = {
        "metadata": {
            "dataset_origin": "SIMULATED",
            "link_capacity_mbps": 100.0,
            "interval": "1 minute",
            "seed": seed,
            "total_samples": len(df),
            "split_counts": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
            },
            "congestion_threshold": CONGESTION_THRESHOLD,
            "feature_count": len(feature_cols),
            "features": feature_cols,
        },
        "class_balance": {
            "train": {
                "total": len(train_df),
                "congested": int((train_df[TARGET_REGRESSION] >= CONGESTION_THRESHOLD).sum()),
                "congested_pct": round(
                    float((train_df[TARGET_REGRESSION] >= CONGESTION_THRESHOLD).mean() * 100), 2
                ),
            },
            "validation": {
                "total": len(val_df),
                "congested": int((val_df[TARGET_REGRESSION] >= CONGESTION_THRESHOLD).sum()),
                "congested_pct": round(
                    float((val_df[TARGET_REGRESSION] >= CONGESTION_THRESHOLD).mean() * 100), 2
                ),
            },
            "test": {
                "total": len(test_df),
                "congested": int((test_df[TARGET_REGRESSION] >= CONGESTION_THRESHOLD).sum()),
                "congested_pct": round(
                    float((test_df[TARGET_REGRESSION] >= CONGESTION_THRESHOLD).mean() * 100), 2
                ),
            },
        },
        "regression_results": {
            "persistence": {
                "train": evaluate_regression(y_train, y_pred_persist_train),
                "validation": evaluate_regression(y_val, y_pred_persist_val),
                "test": evaluate_regression(y_test, y_pred_persist_test),
            },
            "moving_average_5min": {
                "train": evaluate_regression(y_train, y_pred_ma_train),
                "validation": evaluate_regression(y_val, y_pred_ma_val),
                "test": evaluate_regression(y_test, y_pred_ma_test),
            },
            "ridge": {
                "best_alpha": best_ridge_alpha,
                "train": evaluate_regression(y_train, y_pred_ridge_train),
                "validation": evaluate_regression(y_val, y_pred_ridge_val),
                "test": evaluate_regression(y_test, y_pred_ridge_test),
            },
            "random_forest": {
                "best_max_depth": best_rf_depth,
                "train": evaluate_regression(y_train, y_pred_rf_train),
                "validation": evaluate_regression(y_val, y_pred_rf_val),
                "test": evaluate_regression(y_test, y_pred_rf_test),
            },
        },
        "alert_classification_test": {
            "persistence": evaluate_alert_classification(y_test, y_pred_persist_test),
            "moving_average_5min": evaluate_alert_classification(y_test, y_pred_ma_test),
            "ridge": evaluate_alert_classification(y_test, y_pred_ridge_test),
            "random_forest": evaluate_alert_classification(y_test, y_pred_rf_test),
        },
        "champion_model": {
            "name": champion_name,
            "test_regression": evaluate_regression(y_test, y_pred_champion_test),
            "test_alert": evaluate_alert_classification(y_test, y_pred_champion_test),
        },
    }

    # Extract feature importances if RF is available
    if hasattr(best_rf_model, "feature_importances_"):
        rf_importances = {
            col: round(float(imp), 4)
            for col, imp in zip(feature_cols, best_rf_model.feature_importances_)
        }
        # Sort descending
        rf_importances = dict(sorted(rf_importances.items(), key=lambda item: item[1], reverse=True))
        metrics_summary["rf_feature_importances"] = rf_importances

    # -------------------------------------------------------------
    # 8. Save Artifacts Safely
    # -------------------------------------------------------------
    # Save model artifact bundle
    model_artifact_path = model_dir_path / "model.joblib"
    model_bundle = {
        "model": champion_model,
        "champion_name": champion_name,
        "feature_cols": feature_cols,
        "congestion_threshold": CONGESTION_THRESHOLD,
        "link_capacity_mbps": 100.0,
        "seed": seed,
        "is_simulated": True,
    }
    joblib.dump(model_bundle, model_artifact_path)
    logger.info("Saved model bundle to %s", model_artifact_path)

    # Save feature schema
    schema_path = model_dir_path / "feature_schema.json"
    schema_payload = {
        "dataset_type": "SIMULATED_BOTTLENECK_LINK",
        "link_capacity_mbps": 100.0,
        "interval": "1 minute",
        "feature_names": feature_cols,
        "feature_count": len(feature_cols),
        "target_regression": TARGET_REGRESSION,
        "target_classification": TARGET_CLASSIFICATION,
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
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2)
    logger.info("Saved feature schema to %s", schema_path)

    # Save metrics JSON
    metrics_path = model_dir_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved metrics summary to %s", metrics_path)

    # -------------------------------------------------------------
    # 9. Generate Markdown Evaluation Report
    # -------------------------------------------------------------
    report_path = report_dir_path / "evaluation.md"
    generate_evaluation_report(metrics_summary, report_path)
    logger.info("Saved evaluation report to %s", report_path)

    return metrics_summary


def generate_evaluation_report(metrics: Dict[str, Any], output_path: Path) -> None:
    """Generate comprehensive evaluation report in Markdown format."""
    meta = metrics["metadata"]
    cb = metrics["class_balance"]
    reg = metrics["regression_results"]
    alert = metrics["alert_classification_test"]
    champ = metrics["champion_model"]

    report_lines = [
        "# NetSentinel AI — Phase 2 Congestion Prediction Evaluation Report",
        "",
        "> [!IMPORTANT]",
        "> **SIMULATED TELEMETRY DISCLAIMER**: All datasets, models, and results presented in this",
        "> report are based on **simulated network bottleneck telemetry**. They establish algorithmic",
        "> correctness and pipeline feasibility for the NetSentinel AI platform, but do not represent",
        "> measured physical hardware router behavior or real-world traffic captures.",
        "",
        "## 1. Experimental Setup & Methodology",
        f"- **Simulated Link Capacity:** {meta['link_capacity_mbps']} Mbps",
        f"- **Sampling Resolution:** {meta['interval']}",
        f"- **Random Seed:** {meta['seed']} (deterministic and reproducible)",
        f"- **Total Samples:** {meta['total_samples']} records",
        f"- **Partition Strategy:** Chronological split (no shuffling, zero leakage)",
        f"  - **Train Partition (70%):** {meta['split_counts']['train']} rows",
        f"  - **Validation Partition (15%):** {meta['split_counts']['validation']} rows (used for model tuning)",
        f"  - **Test Partition (15%):** {meta['split_counts']['test']} rows (held-out out-of-sample evaluation)",
        f"- **Congestion Alert Threshold:** Utilization $\\ge {meta['congestion_threshold']:.2f}$ (80% link capacity)",
        "",
        "## 2. Partition Class Balance (Congestion Event Distribution)",
        "| Partition | Total Intervals | Congested Intervals ($\\ge 0.80$) | Congestion Prevalance (%) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Train** | {cb['train']['total']} | {cb['train']['congested']} | {cb['train']['congested_pct']}% |",
        f"| **Validation** | {cb['validation']['total']} | {cb['validation']['congested']} | {cb['validation']['congested_pct']}% |",
        f"| **Test** | {cb['test']['total']} | {cb['test']['congested']} | {cb['test']['congested_pct']}% |",
        "",
        "## 3. Regression Performance Comparison",
        "Predicting continuous utilization $U_{t+1}$ 1 minute ahead:",
        "",
        "| Model Architecture | Train MAE | Train RMSE | Val MAE | Val RMSE | Test MAE | Test RMSE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Persistence Baseline ($U_{{t}}$)** | {reg['persistence']['train']['mae']} | {reg['persistence']['train']['rmse']} | {reg['persistence']['validation']['mae']} | {reg['persistence']['validation']['rmse']} | {reg['persistence']['test']['mae']} | {reg['persistence']['test']['rmse']} |",
        f"| **5-min Moving Average** | {reg['moving_average_5min']['train']['mae']} | {reg['moving_average_5min']['train']['rmse']} | {reg['moving_average_5min']['validation']['mae']} | {reg['moving_average_5min']['validation']['rmse']} | {reg['moving_average_5min']['test']['mae']} | {reg['moving_average_5min']['test']['rmse']} |",
        f"| **Ridge Regression ($\\alpha={reg['ridge']['best_alpha']}$)** | {reg['ridge']['train']['mae']} | {reg['ridge']['train']['rmse']} | {reg['ridge']['validation']['mae']} | {reg['ridge']['validation']['rmse']} | {reg['ridge']['test']['mae']} | {reg['ridge']['test']['rmse']} |",
        f"| **Random Forest (depth={reg['random_forest']['best_max_depth']})** | {reg['random_forest']['train']['mae']} | {reg['random_forest']['train']['rmse']} | {reg['random_forest']['validation']['mae']} | {reg['random_forest']['validation']['rmse']} | {reg['random_forest']['test']['mae']} | {reg['random_forest']['test']['rmse']} |",
        "",
        f"**Selected Champion Model:** `{champ['name']}` (selected based on minimum Validation MAE).",
        "",
        "## 4. Congestion Alert Classification (Threshold = 0.80)",
        "Evaluating binary congestion alert warning ($\\hat{U}_{t+1} \\ge 0.80$) on the untouched Test partition:",
        "",
        "| Model Architecture | Test Precision | Test Recall | Test F1-Score | Confusion Matrix [TN, FP, FN, TP] |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Persistence Baseline** | {alert['persistence']['precision']} | {alert['persistence']['recall']} | {alert['persistence']['f1']} | {list(alert['persistence']['confusion_matrix'].values())} |",
        f"| **5-min Moving Average** | {alert['moving_average_5min']['precision']} | {alert['moving_average_5min']['recall']} | {alert['moving_average_5min']['f1']} | {list(alert['moving_average_5min']['confusion_matrix'].values())} |",
        f"| **Ridge Regression** | {alert['ridge']['precision']} | {alert['ridge']['recall']} | {alert['ridge']['f1']} | {list(alert['ridge']['confusion_matrix'].values())} |",
        f"| **Random Forest** | {alert['random_forest']['precision']} | {alert['random_forest']['recall']} | {alert['random_forest']['f1']} | {list(alert['random_forest']['confusion_matrix'].values())} |",
        "",
        "### Test Confusion Matrix Breakdown (Champion Model):",
        f"- **True Negatives (TN):** {champ['test_alert']['confusion_matrix']['tn']} (Correctly predicted normal traffic)",
        f"- **False Positives (FP):** {champ['test_alert']['confusion_matrix']['fp']} (False alarm during normal traffic)",
        f"- **False Negatives (FN):** {champ['test_alert']['confusion_matrix']['fn']} (Missed congestion event)",
        f"- **True Positives (TP):** {champ['test_alert']['confusion_matrix']['tp']} (Successfully anticipated congestion event)",
        "",
    ]

    if "rf_feature_importances" in metrics:
        report_lines.extend(
            [
                "## 5. Random Forest Feature Importances",
                "Relative contribution of past/current features to predicting next-minute utilization:",
                "",
                "| Feature Name | Description | Importance |",
                "| :--- | :--- | :---: |",
            ]
        )
        for feat, imp in metrics["rf_feature_importances"].items():
            desc = meta["features"]
            report_lines.append(f"| `{feat}` | {feat.replace('_', ' ').capitalize()} | {imp:.4f} |")
        report_lines.append("")

    report_lines.extend(
        [
            "## 6. Assumptions and Limitations of Simulated Data",
            "1. **Idealized Bottleneck Model:** Telemetry models a single gateway link with fixed 100 Mbps capacity. Multi-path routing, BGP re-routing, and ECMP effects are omitted.",
            "2. **Macro-Level Resolution:** 1-minute aggregation captures structural queue trends and capacity exhaustion, but does not capture millisecond micro-bursts that cause instantaneous hardware buffer drops.",
            "3. **Zero-Division Handling:** For any scenario where positive alerts are absent in testing, metrics are guarded using `zero_division=0.0` to avoid statistical misrepresentation.",
            "4. **No Real-World Claims:** This pipeline demonstrates end-to-end integration feasibility. Production deployment will require live hardware telemetry ingestion.",
            "",
            "---",
            "*Report auto-generated by `ml.congestion.train_congestion`.*",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate congestion prediction models.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/simulated_congestion_1min.csv",
        help="Path to preprocessed 1-minute congestion CSV.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(DEFAULT_REPORT_DIR),
        help="Directory to save evaluation reports.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    results = train_and_evaluate(
        data_path=args.data,
        seed=args.seed,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
    )

    print("\n================ CONGESTION PREDICTION EVALUATION RESULTS ================")
    print(f"Data: {args.data} (SIMULATED)")
    print(f"Total Samples: {results['metadata']['total_samples']}")
    print(
        f"Splits: Train={results['metadata']['split_counts']['train']}, "
        f"Val={results['metadata']['split_counts']['validation']}, "
        f"Test={results['metadata']['split_counts']['test']}"
    )
    print("\n--- Regression Results (Test Set) ---")
    reg = results["regression_results"]
    print(f"Persistence Baseline:  MAE={reg['persistence']['test']['mae']:.5f}, RMSE={reg['persistence']['test']['rmse']:.5f}")
    print(f"5-min Moving Average:  MAE={reg['moving_average_5min']['test']['mae']:.5f}, RMSE={reg['moving_average_5min']['test']['rmse']:.5f}")
    print(f"Ridge Regression:      MAE={reg['ridge']['test']['mae']:.5f}, RMSE={reg['ridge']['test']['rmse']:.5f}")
    print(f"Random Forest:         MAE={reg['random_forest']['test']['mae']:.5f}, RMSE={reg['random_forest']['test']['rmse']:.5f}")

    print("\n--- Alert Classification (Test Set, Threshold=0.80) ---")
    alert = results["alert_classification_test"]
    for m_name in ["persistence", "moving_average_5min", "ridge", "random_forest"]:
        a = alert[m_name]
        cm = a["confusion_matrix"]
        print(
            f"{m_name.replace('_', ' ').capitalize():<22}: "
            f"Precision={a['precision']:.4f}, Recall={a['recall']:.4f}, F1={a['f1']:.4f} "
            f"[TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}]"
        )
    print("==========================================================================")


if __name__ == "__main__":
    main()
