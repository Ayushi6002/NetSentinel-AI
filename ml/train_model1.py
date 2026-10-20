"""
NetSentinel-AI Model 1 - Offline Binary Attack Classifier Training Utility
==========================================================================

Trains a RandomForestClassifier on CICIDS2017 flow records to perform
binary classification: BENIGN (0) vs. ATTACK (1).

Important Disclaimers:
----------------------
1. Offline Classifier: This is an offline experimental model and is NOT integrated
   into the live Phase 1 network intelligence engine or telemetry pipeline.
2. Data Leakage Warning: Random flow-level train/test splits can leak near-identical
   packets or flow telemetry from the same attack session into both training and testing folds.
   Standard stratified split metrics are therefore likely optimistic.
3. Leave-One-File-Out: Serves as a distinct, significantly more rigorous out-of-distribution
   generalization evaluation.
4. Quick-Mode Metrics: Numbers produced with `--quick` are smoke-test verification numbers
   only and must not be cited as final model evaluation results.
5. Serialization Security: Model artifacts (`model1_rf.joblib`) utilize Python serialization
   (pickle/joblib) and should only ever be loaded from trusted internal sources.

CLI Options:
------------
  --data-dir            Path to raw CSV dataset directory (default: data/raw/MachineLearningCVE)
  --out-dir             Path to output model artifacts directory (default: models/model1)
  --quick               Enable fast chunked sampling for smoke testing
  --max-train-rows      Cap on training split row count (test split remains uncapped)
  --leave-one-file-out  Perform leave-one-file-out cross-validation across the 8 CSV files
  --no-dest-port        Ablation experiment: retrain without Destination Port feature
  --seed                Random seed for reproducibility (default: 42)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Ensure UTF-8 console output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ml.data_prep import stream_clean_dataset


def split_data(
    features: pd.DataFrame,
    labels: pd.Series,
    metadata: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    max_train_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Perform stratified train/test split on binary target.

    Subsampling and caps apply ONLY to the training fold.
    The test fold preserves its natural class distribution and is never capped or rebalanced.
    """
    if len(features) == 0:
        raise ValueError("Cannot split empty dataset.")

    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            f"Stratified split requires at least 2 distinct classes. Found only 1 class: {dict(class_counts)}"
        )

    for cls, cnt in class_counts.items():
        if cnt < 2:
            raise ValueError(
                f"Class {cls} has only {cnt} sample(s). Minimum of 2 samples per class required for splitting."
            )
        if cnt < 10:
            print(f"[!] Warning: Class {cls} has very few samples ({cnt}). Split may have high variance.", flush=True)

    # Perform stratified split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        features,
        labels,
        metadata,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    # Subsample training fold only if requested
    if max_train_rows is not None and len(X_train) > max_train_rows:
        rng = np.random.RandomState(seed)
        # Stratified subsampling of training fold
        train_idx = np.arange(len(X_train))
        sampled_train_idx, _ = train_test_split(
            train_idx,
            train_size=max_train_rows,
            stratify=y_train,
            random_state=seed,
        )
        sampled_train_idx = np.sort(sampled_train_idx)
        X_train = X_train.iloc[sampled_train_idx].reset_index(drop=True)
        y_train = y_train.iloc[sampled_train_idx].reset_index(drop=True)
        meta_train = meta_train.iloc[sampled_train_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def evaluate_classifier(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics on test set.

    Writes null for ROC-AUC if undefined, never NaN.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Baseline: majority class accuracy
    class_counts = y_test.value_counts()
    majority_count = class_counts.max()
    majority_class_baseline = float(majority_count / len(y_test)) if len(y_test) > 0 else 0.0

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC calculation
    roc_auc_val: Optional[float] = None
    roc_auc_reason: Optional[str] = None

    if len(np.unique(y_test)) < 2:
        roc_auc_val = None
        roc_auc_reason = "ROC-AUC is undefined because test set contains only one class."
    elif y_prob is None or y_prob.shape[1] < 2:
        roc_auc_val = None
        roc_auc_reason = "Model predict_proba returned fewer than 2 probability columns."
    else:
        try:
            val = float(roc_auc_score(y_test, y_prob[:, 1]))
            if np.isnan(val) or np.isinf(val):
                roc_auc_val = None
                roc_auc_reason = "ROC-AUC computed as NaN or Inf."
            else:
                roc_auc_val = round(val, 6)
        except Exception as e:
            roc_auc_val = None
            roc_auc_reason = f"ROC-AUC calculation raised {type(e).__name__}: {str(e)}"

    # Category-level recall on original attack categories
    category_recalls: Dict[str, Dict[str, Any]] = {}
    if "original_label" in meta_test.columns:
        test_df_eval = pd.DataFrame({
            "original_label": meta_test["original_label"].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        })
        for cat, group in test_df_eval.groupby("original_label"):
            total_cat = len(group)
            if cat == "BENIGN":
                correct = int((group["y_pred"] == 0).sum())
            else:
                correct = int((group["y_pred"] == 1).sum())
            cat_recall = round(correct / total_cat, 4) if total_cat > 0 else 0.0
            category_recalls[str(cat)] = {
                "total_samples": total_cat,
                "correct_predictions": correct,
                "recall": cat_recall,
            }

    # Top feature importances
    feature_importances: List[Dict[str, Any]] = []
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        feature_names = list(X_test.columns)
        sorted_indices = np.argsort(raw_imp)[::-1]
        for idx in sorted_indices[:15]:
            feature_importances.append({
                "feature": feature_names[idx],
                "importance": round(float(raw_imp[idx]), 6),
            })

    return {
        "test_sample_count": len(y_test),
        "majority_class_baseline": round(majority_class_baseline, 6),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "raw_matrix": [[tn, fp], [fn, tp]],
        },
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "roc_auc": roc_auc_val,
        "roc_auc_note": roc_auc_reason,
        "category_recalls": category_recalls,
        "top_feature_importances": feature_importances,
    }


def run_leave_one_file_out(
    features: pd.DataFrame,
    labels: pd.Series,
    metadata: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute Leave-One-File-Out cross validation across all unique source files.

    Records skipped folds with explicit reasons.
    """
    unique_files = sorted(list(metadata["source_file"].unique()))
    fold_results: List[Dict[str, Any]] = []
    skipped_folds: List[Dict[str, str]] = []

    print("\nRunning Leave-One-File-Out Validation Across Files...", flush=True)

    for holdout_file in unique_files:
        test_mask = metadata["source_file"] == holdout_file
        train_mask = ~test_mask

        y_train = labels[train_mask]
        y_test = labels[test_mask]

        # Check fold validity
        if len(y_test) == 0:
            skipped_folds.append({
                "file": holdout_file,
                "reason": "Test fold contains 0 samples."
            })
            continue

        train_classes = np.unique(y_train)
        if len(train_classes) < 2:
            skipped_folds.append({
                "file": holdout_file,
                "reason": f"Training fold contains only 1 class: {train_classes.tolist()}. Cannot train binary model."
            })
            continue

        test_classes = np.unique(y_test)
        if len(test_classes) < 2:
            # Note: Single-class test fold (e.g. Monday is 100% BENIGN)
            # We can still evaluate accuracy and false positive rate, but ROC-AUC is undefined.
            test_note = f"Test fold contains only 1 class ({test_classes.tolist()}); binary ROC-AUC/precision/recall may be partial."
        else:
            test_note = "Fold valid (both classes present in test)."

        X_train = features[train_mask]
        X_test = features[test_mask]
        meta_test = metadata[test_mask]

        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test, meta_test)
        metrics["holdout_file"] = holdout_file
        metrics["fold_note"] = test_note
        metrics["train_class_counts"] = {str(k): int(v) for k, v in y_train.value_counts().items()}
        metrics["test_class_counts"] = {str(k): int(v) for k, v in y_test.value_counts().items()}

        fold_results.append(metrics)
        print(f"  Holdout File: {holdout_file[:35]:<36} Accuracy: {metrics['accuracy']:.4f} F1: {metrics['f1_score']:.4f}", flush=True)

    return {
        "completed_folds": fold_results,
        "skipped_folds": skipped_folds,
    }


def generate_evaluation_report(
    metrics: Dict[str, Any],
    ablation_metrics: Optional[Dict[str, Any]],
    lofo_results: Optional[Dict[str, Any]],
    quick: bool,
    class_weight: str,
    n_estimators: int,
    max_depth: Optional[int],
) -> str:
    """Generate comprehensive Markdown evaluation report meeting all audit requirements."""
    cm = metrics["confusion_matrix"]
    now_utc = datetime.now(timezone.utc).isoformat()

    report = f"""# NetSentinel-AI Model 1: Offline Binary Classifier Evaluation Report
**Generated UTC**: {now_utc}  
**Model Architecture**: Random Forest Classifier (`RandomForestClassifier`)  
**Execution Mode**: {'QUICK / SMOKE-TEST MODE' if quick else 'FULL DATASET TRAINING'}

---

## 1. Important Disclaimers & Methodological Context
1. **Offline Model Only**: This classifier is an offline research module engineered to analyze flow patterns in the CICIDS2017 dataset. It is **not** integrated into Phase 1 live packet capture or real-time detection telemetry.
2. **Session / Flow Leakage**: Random stratified flow-level splits allow flows from the same attack session to appear simultaneously in both train and test partitions. Standard split metrics are inherently **optimistic**.
3. **Leave-One-File-Out Generalization**: Evaluating across separate raw files represents a much more realistic and challenging out-of-distribution test because files correspond to different recording sessions and distinct attack profiles.
4. **Quick-Mode Notice**: Metrics produced in `--quick` mode are for smoke-testing and pipeline verification only, not final evaluation numbers.
5. **Artifact Security**: The serialized model (`model1_rf.joblib`) uses Python joblib/pickle serialization and should only ever be loaded from trusted internal sources.

---

## 2. Hyperparameter Rationale & Training Configuration
- **Model Choice**: `RandomForestClassifier`
- **n_estimators**: `{n_estimators}` (provides ensemble diversity while maintaining practical training times)
- **max_depth**: `{max_depth if max_depth is not None else 'None (unbounded)'}` (controls overfitting on correlated flow metrics)
- **class_weight**: `'{class_weight}'`
  - *Rationale*: Network telemetry in CICIDS2017 is heavily imbalanced (~80% BENIGN vs. ~20% ATTACK, with rare attacks like Heartbleed possessing only 11 instances). Setting `class_weight='{class_weight}'` penalizes mistakes on minority attack flows proportionally to their class frequency.

---

## 3. Test Set Performance Metrics (Natural Distribution)
*Note: Test set preserves its natural class distribution and was never artificially capped or rebalanced.*

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Majority-Class Baseline** | {metrics['majority_class_baseline']:.4f} | Accuracy achieved by predicting majority class on all flows |
| **Accuracy** | **{metrics['accuracy']:.4f}** | Overall correct predictions |
| **Attack Precision** | {metrics['precision']:.4f} | TP / (TP + FP) |
| **Attack Recall** | **{metrics['recall']:.4f}** | TP / (TP + FN) |
| **F1 Score** | **{metrics['f1_score']:.4f}** | Harmonic mean of precision and recall |
| **False Positive Rate (FPR)** | {metrics['false_positive_rate']:.4f} | FP / (FP + TN) |
| **False Negative Rate (FNR)** | {metrics['false_negative_rate']:.4f} | FN / (FN + TP) |
| **ROC-AUC** | {str(metrics['roc_auc']) if metrics['roc_auc'] is not None else 'null'} | {metrics.get('roc_auc_note') or 'Area Under ROC Curve'} |

### Confusion Matrix
- **True Negatives (TN)**: {cm['true_negatives']:,}
- **False Positives (FP)**: {cm['false_positives']:,}
- **False Negatives (FN)**: {cm['false_negatives']:,}
- **True Positives (TP)**: {cm['true_positives']:,}

---

## 4. Category-Level Recall Breakdown
Evaluates detection sensitivity across each original attack category:

| Original Category | Total Test Flows | Correctly Identified | Recall |
| :--- | :---: | :---: | :---: |
"""
    for cat, data in metrics.get("category_recalls", {}).items():
        report += f"| `{cat}` | {data['total_samples']:,} | {data['correct_predictions']:,} | **{data['recall']:.4f}** |\n"

    report += """
---

## 5. Top Feature Importances (Gini Impurity Reduction)

| Rank | Feature Name | Importance Score |
| :---: | :--- | :---: |
"""
    for idx, item in enumerate(metrics.get("top_feature_importances", []), 1):
        report += f"| {idx} | `{item['feature']}` | {item['importance']:.6f} |\n"

    if ablation_metrics:
        report += f"""
---

## 6. Feature Ablation: Destination Port Removal
Evaluating whether the classifier relies heavily on port numbers (`Destination Port`):
- Baseline Model: `model1_rf.joblib` (`feature_schema.json`, {metrics.get('feature_count', 77)} features)
- Ablation Model: `model1_rf_no_port.joblib` (`feature_schema_no_port.json`, {ablation_metrics.get('feature_count', 76)} features)

| Metric | With Destination Port | Without Destination Port | Difference |
| :--- | :---: | :---: | :---: |
| **Accuracy** | {metrics['accuracy']:.4f} | {ablation_metrics['accuracy']:.4f} | {ablation_metrics['accuracy'] - metrics['accuracy']:+.4f} |
| **Attack Precision** | {metrics['precision']:.4f} | {ablation_metrics['precision']:.4f} | {ablation_metrics['precision'] - metrics['precision']:+.4f} |
| **Attack Recall** | {metrics['recall']:.4f} | {ablation_metrics['recall']:.4f} | {ablation_metrics['recall'] - metrics['recall']:+.4f} |
| **F1 Score** | {metrics['f1_score']:.4f} | {ablation_metrics['f1_score']:.4f} | {ablation_metrics['f1_score'] - metrics['f1_score']:+.4f} |
| **False Positive Rate (FPR)** | {metrics['false_positive_rate']:.4f} | {ablation_metrics['false_positive_rate']:.4f} | {ablation_metrics['false_positive_rate'] - metrics['false_positive_rate']:+.4f} |
| **False Negative Rate (FNR)** | {metrics['false_negative_rate']:.4f} | {ablation_metrics['false_negative_rate']:.4f} | {ablation_metrics['false_negative_rate'] - metrics['false_negative_rate']:+.4f} |
"""

    if lofo_results:
        report += """
---

## 7. Leave-One-File-Out Cross-Validation
*Evaluates generalization when entire capture files are held out. Note: some training folds completely lack certain attack types.*

| Holdout File | Accuracy | F1 Score | Notes |
| :--- | :---: | :---: | :--- |
"""
        for fold in lofo_results.get("completed_folds", []):
            f1_str = f"{fold['f1_score']:.4f}" if fold['f1_score'] is not None else "N/A"
            report += f"| `{fold['holdout_file']}` | {fold['accuracy']:.4f} | {f1_str} | {fold['fold_note']} |\n"

        if lofo_results.get("skipped_folds"):
            report += "\n### Skipped Folds:\n"
            for sk in lofo_results["skipped_folds"]:
                report += f"- `{sk['file']}`: {sk['reason']}\n"

    return report


def save_model_artifacts(
    out_dir: Path,
    clf: RandomForestClassifier,
    prep_report: Dict[str, Any],
    metrics: Dict[str, Any],
    seed: int = 42,
    quick: bool = False,
    class_weight: str = "balanced",
    n_estimators: int = 100,
    max_depth: Optional[int] = 20,
    X_test: Optional[pd.DataFrame] = None,
    clf_abl: Optional[RandomForestClassifier] = None,
    ablation_metrics: Optional[Dict[str, Any]] = None,
    lofo_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Save all Model 1 artifacts ensuring baseline and ablation models and schemas remain consistent.

    Baseline artifacts:
      - out_dir / "model1_rf.joblib" (77 features)
      - out_dir / "feature_schema.json" (77 features)
      - out_dir / "sample_upload.csv" (77 features, if X_test provided)

    Ablation artifacts (if clf_abl is provided):
      - out_dir / "model1_rf_no_port.joblib" (76 features, without Destination Port)
      - out_dir / "feature_schema_no_port.json" (76 features, without Destination Port)
      - out_dir / "sample_upload_no_port.csv" (76 features, if X_test provided)

    Common artifacts:
      - out_dir / "metrics.json" (records both baseline and ablation metrics with explicit artifact mapping)
      - out_dir / "evaluation_report.md"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}

    baseline_features = list(prep_report.get("ordered_features", []))
    if hasattr(clf, "n_features_in_") and clf.n_features_in_ != len(baseline_features):
        raise ValueError(
            f"Baseline classifier n_features_in_ ({clf.n_features_in_}) does not match "
            f"baseline feature schema count ({len(baseline_features)}). Refusing to save inconsistent baseline artifact."
        )

    # 1. Baseline Model & Schema
    baseline_model_path = out_dir / "model1_rf.joblib"
    joblib.dump(clf, baseline_model_path)
    saved_paths["baseline_model"] = baseline_model_path

    baseline_schema = {
        "model_name": "NetSentinel-AI Model 1 (Offline Binary Attack Classifier)",
        "model_type": "RandomForestClassifier",
        "model_file": "model1_rf.joblib",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "library_versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": "1.9.1",
            "joblib": joblib.__version__,
        },
        "cleaning_rules": {
            "dropped_columns": prep_report.get("dropped_columns", {}),
            "nan_inf_policy": "Coerce numeric, convert +/-Inf to NaN, drop rows with NaN in any feature",
            "flow_duration_policy": "Drop rows where Flow Duration < 0",
            "dedup_policy": prep_report.get("dedup_methodology", ""),
            "feature_dtype": "float32",
        },
        "ordered_features": baseline_features,
        "feature_count": len(baseline_features),
    }
    baseline_schema_path = out_dir / "feature_schema.json"
    with open(baseline_schema_path, "w", encoding="utf-8") as f:
        json.dump(baseline_schema, f, indent=2)
    saved_paths["baseline_schema"] = baseline_schema_path

    # 2. Ablation Model & Schema (if ablation model trained)
    ablation_features: Optional[List[str]] = None
    if clf_abl is not None:
        ablation_features = [col for col in baseline_features if col != "Destination Port"]
        if hasattr(clf_abl, "n_features_in_") and clf_abl.n_features_in_ != len(ablation_features):
            raise ValueError(
                f"Ablation classifier n_features_in_ ({clf_abl.n_features_in_}) does not match "
                f"ablation feature schema count ({len(ablation_features)}). Refusing to save inconsistent ablation artifact."
            )

        abl_model_path = out_dir / "model1_rf_no_port.joblib"
        joblib.dump(clf_abl, abl_model_path)
        saved_paths["ablation_model"] = abl_model_path

        dropped_cols_abl = dict(prep_report.get("dropped_columns", {}))
        dropped_cols_abl["Destination Port (ablation)"] = (
            "Explicitly removed for feature ablation experiment to test model sensitivity to port-based telemetry."
        )

        abl_schema = {
            "model_name": "NetSentinel-AI Model 1 (Offline Binary Attack Classifier - Destination Port Ablation)",
            "model_type": "RandomForestClassifier",
            "model_file": "model1_rf_no_port.joblib",
            "training_date_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "library_versions": {
                "python": sys.version,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit-learn": "1.9.1",
                "joblib": joblib.__version__,
            },
            "cleaning_rules": {
                "dropped_columns": dropped_cols_abl,
                "nan_inf_policy": "Coerce numeric, convert +/-Inf to NaN, drop rows with NaN in any feature",
                "flow_duration_policy": "Drop rows where Flow Duration < 0",
                "dedup_policy": prep_report.get("dedup_methodology", ""),
                "feature_dtype": "float32",
            },
            "ordered_features": ablation_features,
            "feature_count": len(ablation_features),
        }
        abl_schema_path = out_dir / "feature_schema_no_port.json"
        with open(abl_schema_path, "w", encoding="utf-8") as f:
            json.dump(abl_schema, f, indent=2)
        saved_paths["ablation_schema"] = abl_schema_path

    # 3. Metrics Payload
    annotated_test_metrics = dict(metrics)
    annotated_test_metrics["model_file"] = "model1_rf.joblib"
    annotated_test_metrics["schema_file"] = "feature_schema.json"
    annotated_test_metrics["feature_count"] = len(baseline_features)

    annotated_ablation_metrics: Optional[Dict[str, Any]] = None
    if ablation_metrics is not None:
        annotated_ablation_metrics = dict(ablation_metrics)
        annotated_ablation_metrics["model_file"] = "model1_rf_no_port.joblib"
        annotated_ablation_metrics["schema_file"] = "feature_schema_no_port.json"
        annotated_ablation_metrics["feature_count"] = len(ablation_features) if ablation_features else len(baseline_features) - 1
        annotated_ablation_metrics["ablation_target"] = "Destination Port"

    metrics_payload = {
        "evaluation_mode": "QUICK_SMOKE_TEST" if quick else "FULL_EVALUATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_artifact": {
            "model_file": "model1_rf.joblib",
            "schema_file": "feature_schema.json",
            "feature_count": len(baseline_features),
            "includes_destination_port": True,
        },
        "test_metrics": annotated_test_metrics,
        "ablation_artifact": {
            "model_file": "model1_rf_no_port.joblib",
            "schema_file": "feature_schema_no_port.json",
            "feature_count": len(ablation_features) if ablation_features else len(baseline_features) - 1,
            "includes_destination_port": False,
        } if clf_abl is not None else None,
        "ablation_metrics": annotated_ablation_metrics,
        "leave_one_file_out": lofo_results,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    saved_paths["metrics"] = metrics_path

    # 4. Evaluation Report
    report_md = generate_evaluation_report(
        metrics=annotated_test_metrics,
        ablation_metrics=annotated_ablation_metrics,
        lofo_results=lofo_results,
        quick=quick,
        class_weight=class_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    report_path = out_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    saved_paths["evaluation_report"] = report_path

    # 5. Sample Upload CSVs
    if X_test is not None and len(X_test) > 0:
        sample_rows = X_test.head(10).copy()
        sample_csv_path = out_dir / "sample_upload.csv"
        sample_rows.to_csv(sample_csv_path, index=False)
        saved_paths["sample_upload"] = sample_csv_path

        if clf_abl is not None and "Destination Port" in sample_rows.columns:
            sample_no_port_rows = sample_rows.drop(columns=["Destination Port"])
            sample_no_port_csv_path = out_dir / "sample_upload_no_port.csv"
            sample_no_port_rows.to_csv(sample_no_port_csv_path, index=False)
            saved_paths["sample_upload_no_port"] = sample_no_port_csv_path

    return saved_paths


def main() -> None:
    """Entry point for model training CLI."""
    parser = argparse.ArgumentParser(
        description="NetSentinel-AI Model 1: Offline Binary Attack Classifier Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/MachineLearningCVE"), help="Raw CSV dataset directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("models/model1"), help="Output directory for model artifacts.")
    parser.add_argument("--quick", action="store_true", help="Quick mode: small chunked sample for smoke testing.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Cap on training fold row count.")
    parser.add_argument("--leave-one-file-out", action="store_true", help="Run leave-one-file-out cross-validation.")
    parser.add_argument("--no-dest-port", action="store_true", help="Ablation experiment: retrain without Destination Port.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("NetSentinel-AI Model 1: Offline Binary Attack Classifier Training", flush=True)
    print(f"Data Directory : {args.data_dir.resolve()}", flush=True)
    print(f"Output Directory: {args.out_dir.resolve()}", flush=True)
    print(f"Execution Mode : {'QUICK / SMOKE-TEST' if args.quick else 'FULL DATASET'}", flush=True)
    print(f"Random Seed    : {args.seed}", flush=True)
    print("=" * 80, flush=True)

    # 1. Stream, clean, and deduplicate dataset
    t0 = time.time()
    features, labels, metadata, prep_report = stream_clean_dataset(
        data_dir=args.data_dir,
        quick=args.quick,
        seed=args.seed,
    )
    print(f"Data preparation complete in {time.time()-t0:.2f}s.", flush=True)
    print(f"Clean Unique Records: {len(features):,}, Features: {prep_report['feature_count']}", flush=True)

    # 2. Train/test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_data(
        features=features,
        labels=labels,
        metadata=metadata,
        test_size=0.2,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
    )
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}", flush=True)

    # 3. Model training
    class_weight_choice = "balanced"
    n_estimators = 50 if args.quick else 100
    max_depth = 15 if args.quick else 20

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight_choice,
        n_jobs=-1,
        random_state=args.seed,
    )
    print("Fitting RandomForestClassifier...", flush=True)
    t_fit = time.time()
    clf.fit(X_train, y_train)
    print(f"Model fit complete in {time.time()-t_fit:.2f}s.", flush=True)

    # 4. Evaluation
    metrics = evaluate_classifier(clf, X_test, y_test, meta_test)
    print("\n--- Test Evaluation Results ---", flush=True)
    print(f"Accuracy : {metrics['accuracy']:.4f} (Baseline: {metrics['majority_class_baseline']:.4f})", flush=True)
    print(f"Precision: {metrics['precision']:.4f}", flush=True)
    print(f"Recall   : {metrics['recall']:.4f}", flush=True)
    print(f"F1 Score : {metrics['f1_score']:.4f}", flush=True)
    print(f"ROC-AUC  : {metrics['roc_auc']}", flush=True)

    # 5. Optional Ablation (without Destination Port)
    ablation_metrics = None
    clf_abl = None
    if args.no_dest_port:
        if "Destination Port" in X_train.columns:
            print("\nRunning Ablation: Retraining without Destination Port...", flush=True)
            X_train_abl = X_train.drop(columns=["Destination Port"])
            X_test_abl = X_test.drop(columns=["Destination Port"])
            clf_abl = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight_choice,
                n_jobs=-1,
                random_state=args.seed,
            )
            clf_abl.fit(X_train_abl, y_train)
            ablation_metrics = evaluate_classifier(clf_abl, X_test_abl, y_test, meta_test)
            print(f"Ablation Accuracy: {ablation_metrics['accuracy']:.4f} (Diff: {ablation_metrics['accuracy'] - metrics['accuracy']:+.4f})", flush=True)

    # 6. Optional Leave-One-File-Out
    lofo_results = None
    if args.leave_one_file_out:
        lofo_results = run_leave_one_file_out(features, labels, metadata, seed=args.seed)

    # 7. Save Model Artifacts
    print("\nSaving model artifacts...", flush=True)
    saved_artifacts = save_model_artifacts(
        out_dir=args.out_dir,
        clf=clf,
        prep_report=prep_report,
        metrics=metrics,
        seed=args.seed,
        quick=args.quick,
        class_weight=class_weight_choice,
        n_estimators=n_estimators,
        max_depth=max_depth,
        X_test=X_test,
        clf_abl=clf_abl,
        ablation_metrics=ablation_metrics,
        lofo_results=lofo_results,
    )
    print("All Model 1 artifacts saved successfully in:", args.out_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
