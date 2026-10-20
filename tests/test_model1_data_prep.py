"""Unit tests for Model 1 data preparation and offline classifier utilities.

All tests utilize synthetic temporary CSV fixtures. No raw dataset files are touched.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import joblib
from ml.data_prep import (
    clean_chunk,
    normalize_header_names,
    normalize_labels,
    stream_clean_dataset,
    to_binary,
    validate_upload,
)
from ml.train_model1 import (
    evaluate_classifier,
    save_model_artifacts,
    split_data,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def test_label_normalization_and_binary_mapping():
    """Verify normalization of U+FFFD replacement characters, mojibake, and binary conversion."""
    raw_labels = pd.Series([
        "BENIGN",
        "Web Attack \ufffd Brute Force",
        "Web Attack ? XSS",
        "Web Attack \ufffd Sql Injection",
        "DoS Hulk",
        "PortScan",
        "DDoS",
    ])

    norm_labels = normalize_labels(raw_labels)

    assert norm_labels[0] == "BENIGN"
    assert norm_labels[1] == "Web Attack - Brute Force"
    assert norm_labels[2] == "Web Attack - XSS"
    assert norm_labels[3] == "Web Attack - Sql Injection"
    assert norm_labels[4] == "DoS Hulk"
    assert norm_labels[5] == "PortScan"
    assert norm_labels[6] == "DDoS"

    binary = to_binary(norm_labels)
    expected_binary = pd.Series([0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    pd.testing.assert_series_equal(binary, expected_binary)


def test_header_whitespace_and_duplicate_resolution():
    """Verify whitespace stripping and detection of duplicate column names."""
    raw_headers = [
        " Destination Port",
        " Flow Duration",
        " Fwd Header Length",
        " Fwd Header Length ",
        " Label",
    ]
    norm, dups = normalize_header_names(raw_headers)
    assert norm == ["Destination Port", "Flow Duration", "Fwd Header Length", "Fwd Header Length", "Label"]
    assert "Fwd Header Length" in dups
    assert dups["Fwd Header Length"] == [2, 3]


def test_clean_chunk_nan_inf_and_negative_duration():
    """Verify filtering of NaN, +/-Inf, and negative Flow Duration values."""
    feature_cols = ["Destination Port", "Flow Duration", "Flow Bytes/s"]
    data = pd.DataFrame({
        "Destination Port": [80, 443, 8080, 22, 53],
        "Flow Duration": [100, -5, 200, 300, 400],  # row 1 has negative duration
        "Flow Bytes/s": [50.0, 10.0, np.inf, np.nan, 25.0],  # row 2 has inf, row 3 has nan
        "Label": ["BENIGN", "DDoS", "PortScan", "BENIGN", "BENIGN"],
    })

    cleaned, report = clean_chunk(data, feature_cols=feature_cols, label_col="Label")

    # Only rows 0 and 4 should survive
    assert len(cleaned) == 2
    assert list(cleaned["Destination Port"]) == [80, 53]
    assert report["input_rows"] == 5
    assert report["removed_nan_inf"] == 2  # rows 2 and 3
    assert report["removed_negative_flow_duration"] == 1  # row 1
    assert report["output_rows"] == 2


def test_metadata_and_target_strictly_excluded_from_features(tmp_path: Path):
    """Enforce that metadata and target labels can never enter the feature matrix."""
    csv_file = tmp_path / "test_exclusion.csv"
    # Create synthetic CSV with 79 columns matching schema layout
    cols = [f" Feature_{i}" for i in range(78)] + [" Label"]
    cols[34] = " Fwd Header Length"
    cols[55] = " Fwd Header Length"

    rows = []
    for r in range(10):
        row = [f"{r * 10}"] * 78
        row.append("BENIGN" if r % 2 == 0 else "DDoS")
        rows.append(row)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    features, binary_labels, metadata, prep_report = stream_clean_dataset(
        data_dir=tmp_path,
        chunk_size=5,
        quick=False,
    )

    # Features must contain 77 numeric features (78 minus duplicate position 55)
    assert len(features.columns) == 77
    assert "Label" not in features.columns
    assert "source_file" not in features.columns
    assert "original_label" not in features.columns
    assert "original_attack_category" not in features.columns
    assert features.dtypes.unique().tolist() == [np.float32]

    # Metadata must contain isolated tracking fields
    assert "source_file" in metadata.columns
    assert "original_label" in metadata.columns
    assert len(binary_labels) == len(features)
    assert len(metadata) == len(features)


def test_stream_deduplication_within_and_across_files(tmp_path: Path):
    """Verify that 64-bit row hash deduplication eliminates duplicate rows within and across files."""
    cols = [f" Feature_{i}" for i in range(78)] + [" Label"]
    cols[34] = " Fwd Header Length"
    cols[55] = " Fwd Header Length"

    # File 1: Row A, Row B, Row A (intra-file duplicate)
    row_a = ["10"] * 78 + ["BENIGN"]
    row_b = ["20"] * 78 + ["DDoS"]
    file1 = tmp_path / "file1.csv"
    with open(file1, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerow(row_a)
        writer.writerow(row_b)
        writer.writerow(row_a)

    # File 2: Row B (cross-file duplicate), Row C
    row_c = ["30"] * 78 + ["PortScan"]
    file2 = tmp_path / "file2.csv"
    with open(file2, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerow(row_b)
        writer.writerow(row_c)

    features, binary_labels, metadata, prep_report = stream_clean_dataset(
        data_dir=tmp_path,
        chunk_size=2,
        quick=False,
    )

    # Total input rows = 3 + 2 = 5
    # Unique rows = 3 (A, B, C)
    assert prep_report["total_input_rows"] == 5
    assert prep_report["removed_duplicates_hash"] == 2
    assert len(features) == 3
    assert set(metadata["original_label"].unique()) == {"BENIGN", "DDoS", "PortScan"}


def test_seeded_sampling_reproducibility(tmp_path: Path):
    """Verify that same seed produces identical samples and different seed produces different samples."""
    cols = [f" Feature_{i}" for i in range(78)] + [" Label"]
    cols[34] = " Fwd Header Length"
    cols[55] = " Fwd Header Length"

    rows = []
    for r in range(100):
        rows.append([f"{r}"] * 78 + ["BENIGN" if r < 70 else "DDoS"])

    csv_file = tmp_path / "sample_test.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    f_seed42_a, y42_a, _, _ = stream_clean_dataset(tmp_path, quick=True, quick_max_total_rows=15, seed=42)
    f_seed42_b, y42_b, _, _ = stream_clean_dataset(tmp_path, quick=True, quick_max_total_rows=15, seed=42)
    f_seed99, y99, _, _ = stream_clean_dataset(tmp_path, quick=True, quick_max_total_rows=15, seed=99)

    pd.testing.assert_frame_equal(f_seed42_a, f_seed42_b)
    pd.testing.assert_series_equal(y42_a, y42_b)

    # Different seeds should produce different selections
    assert not f_seed42_a.equals(f_seed99)


def test_quick_mode_enforces_max_10k_rows(tmp_path: Path):
    """Verify quick mode strictly enforces a maximum of 10,000 rows across classes and all files combined."""
    cols = [f" Feature_{i}" for i in range(78)] + [" Label"]
    cols[34] = " Fwd Header Length"
    cols[55] = " Fwd Header Length"

    # Create two synthetic CSV files totaling 18,000 rows (File 1: 9,000 rows, File 2: 9,000 rows)
    # File 1: rows 0..8999 (6,000 benign, 3,000 PortScan)
    rows_f1 = []
    for r in range(9000):
        rows_f1.append([f"{r}"] * 78 + ["BENIGN" if r < 6000 else "PortScan"])
    f1_path = tmp_path / "file1_quick_test.csv"
    with open(f1_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows_f1)

    # File 2: rows 9000..17999 (6,000 benign, 3,000 DDoS)
    rows_f2 = []
    for r in range(9000, 18000):
        rows_f2.append([f"{r}"] * 78 + ["BENIGN" if r < 15000 else "DDoS"])
    f2_path = tmp_path / "file2_quick_test.csv"
    with open(f2_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows_f2)

    # Test 1: Default quick_max_total_rows=10000 across multiple files
    features_10k, y_10k, meta_10k, report_10k = stream_clean_dataset(
        tmp_path,
        chunk_size=3000,
        quick=True,
        quick_max_total_rows=10000,
        seed=42,
    )

    # Total rows must strictly not exceed 10,000
    assert len(features_10k) <= 10000
    assert len(features_10k) == 10000
    assert len(y_10k) == 10000
    assert report_10k["final_output_rows"] == 10000
    assert report_10k["quick_mode"] is True

    # Both classes must be retained
    benign_count = int((y_10k == 0).sum())
    attack_count = int((y_10k == 1).sum())
    assert benign_count > 0, "Benign rows must be present in quick mode sample"
    assert attack_count > 0, "Attack rows must be present in quick mode sample"
    assert benign_count + attack_count == 10000
    # Both files must contribute rows
    assert "file1_quick_test.csv" in meta_10k["source_file"].values
    assert "file2_quick_test.csv" in meta_10k["source_file"].values

    # Test 2: Custom cap (e.g. 100 rows) strictly enforced
    features_100, y_100, _, _ = stream_clean_dataset(
        tmp_path,
        chunk_size=3000,
        quick=True,
        quick_max_total_rows=100,
        seed=42,
    )
    assert len(features_100) <= 100
    assert len(features_100) == 100
    assert len(y_100) == 100
    assert (y_100 == 0).sum() > 0
    assert (y_100 == 1).sum() > 0

    # Test 3: Reproducibility verification
    features_10k_rep, y_10k_rep, _, _ = stream_clean_dataset(
        tmp_path,
        chunk_size=3000,
        quick=True,
        quick_max_total_rows=10000,
        seed=42,
    )
    pd.testing.assert_frame_equal(features_10k, features_10k_rep)
    pd.testing.assert_series_equal(y_10k, y_10k_rep)



def test_split_data_edge_cases():
    """Verify split_data validation for empty datasets, single-class datasets, and class capping."""
    features = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
    labels_one_class = pd.Series([0, 0, 0, 0])
    metadata = pd.DataFrame({"source_file": ["f1"] * 4})

    # Single-class input rejection
    with pytest.raises(ValueError, match="Stratified split requires at least 2 distinct classes"):
        split_data(features, labels_one_class, metadata)

    # Empty input rejection
    with pytest.raises(ValueError, match="Cannot split empty dataset"):
        split_data(pd.DataFrame(), pd.Series(dtype=np.int8), pd.DataFrame())

    # Valid split with training cap
    features_valid = pd.DataFrame({"A": range(20)})
    labels_valid = pd.Series([0] * 10 + [1] * 10)
    meta_valid = pd.DataFrame({"source_file": ["f1"] * 20})

    X_train, X_test, y_train, y_test, _, _ = split_data(
        features_valid, labels_valid, meta_valid, test_size=0.2, seed=42, max_train_rows=8
    )

    # Test size is 20% of 20 = 4 rows (natural distribution preserved)
    assert len(X_test) == 4
    assert len(y_test) == 4
    # Train set was capped to max_train_rows=8
    assert len(X_train) == 8
    assert len(y_train) == 8
    assert set(y_test.unique()) == {0, 1}


def test_validate_upload_contract():
    """Verify validate_upload rejects missing columns, duplicate headers, invalid values, and ignores extra columns."""
    schema = {
        "ordered_features": ["Port", "Duration", "Bytes"],
    }

    # 1. Valid input
    df_valid = pd.DataFrame({
        " Port ": [80, 443],
        "Duration": [10.5, 20.0],
        "Bytes": [1000, 2000],
    })
    ok, cleaned, msgs = validate_upload(df_valid, schema)
    assert ok is True
    assert cleaned is not None
    assert list(cleaned.columns) == ["Port", "Duration", "Bytes"]
    assert cleaned.dtypes.tolist() == [np.float32, np.float32, np.float32]

    # 2. Missing required column -> returns None
    df_missing = pd.DataFrame({"Port": [80], "Duration": [10]})
    ok, cleaned, msgs = validate_upload(df_missing, schema)
    assert ok is False
    assert cleaned is None
    assert any("Missing" in m for m in msgs)

    # 3. Duplicate normalized headers -> returns None
    df_dup = pd.DataFrame([[80, 10, 20]], columns=["Port", " Port", "Duration"])
    ok, cleaned, msgs = validate_upload(df_dup, schema)
    assert ok is False
    assert cleaned is None
    assert any("Duplicate" in m for m in msgs)

    # 4. Extra column -> ignored and documented
    df_extra = pd.DataFrame({
        "Port": [80],
        "Duration": [10],
        "Bytes": [100],
        "ExtraCol": ["ignore_me"],
    })
    ok, cleaned, msgs = validate_upload(df_extra, schema)
    assert ok is True
    assert cleaned is not None
    assert "ExtraCol" not in cleaned.columns
    assert any("extra columns ignored" in m for m in msgs)

    # 5. Invalid values (string in numeric column) -> returns None
    df_invalid = pd.DataFrame({
        "Port": [80, "invalid_str"],
        "Duration": [10, 20],
        "Bytes": [100, 200],
    })
    ok, cleaned, msgs = validate_upload(df_invalid, schema)
    assert ok is False
    assert cleaned is None
    assert any("non-numeric, NaN, or Infinite" in m for m in msgs)


def test_undefined_roc_auc_serializes_as_null():
    """Verify that single-class test evaluation serializes roc_auc as None/null, never NaN."""
    model = DummyClassifier(strategy="constant", constant=0)
    X_train = pd.DataFrame({"A": [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 0, 0])
    model.fit(X_train, y_train)

    X_test = pd.DataFrame({"A": [5, 6]})
    y_test = pd.Series([0, 0])  # single-class test set
    meta_test = pd.DataFrame({"original_label": ["BENIGN", "BENIGN"]})

    metrics = evaluate_classifier(model, X_test, y_test, meta_test)

    assert metrics["roc_auc"] is None
    assert metrics["roc_auc_note"] is not None

    # JSON serialization must render 'null'
    json_str = json.dumps(metrics)
    assert '"roc_auc": null' in json_str
    assert "NaN" not in json_str


def test_save_model_artifacts_baseline_and_ablation_consistency(tmp_path: Path):
    """Verify baseline and ablation models, schemas, metrics, and sample uploads are saved consistently."""
    feature_names = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Flow Bytes/s"]
    prep_report = {
        "ordered_features": feature_names,
        "feature_count": len(feature_names),
        "dropped_columns": {"Fwd Header Length (position 55)": "Duplicate column"},
        "dedup_methodology": "64-bit row hash",
    }

    # Synthetic training data
    np.random.seed(42)
    X_base = pd.DataFrame(np.random.rand(20, 5).astype(np.float32), columns=feature_names)
    X_abl = X_base.drop(columns=["Destination Port"])
    y = pd.Series([0, 1] * 10, dtype=np.int8)

    # Train baseline (5 features) and ablation (4 features)
    clf_base = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X_base, y)
    clf_abl = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X_abl, y)

    metrics_base = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.95,
        "false_positive_rate": 0.04,
        "false_negative_rate": 0.04,
        "majority_class_baseline": 0.5,
        "confusion_matrix": {"true_negatives": 9, "false_positives": 1, "false_negatives": 1, "true_positives": 9},
        "roc_auc": 0.98,
        "roc_auc_note": None,
        "category_recalls": {},
        "top_feature_importances": [],
    }
    metrics_abl = {
        "accuracy": 0.90,
        "precision": 0.89,
        "recall": 0.91,
        "f1_score": 0.90,
        "false_positive_rate": 0.10,
        "false_negative_rate": 0.09,
        "majority_class_baseline": 0.5,
        "confusion_matrix": {"true_negatives": 8, "false_positives": 2, "false_negatives": 1, "true_positives": 9},
        "roc_auc": 0.95,
        "roc_auc_note": None,
        "category_recalls": {},
        "top_feature_importances": [],
    }

    saved = save_model_artifacts(
        out_dir=tmp_path,
        clf=clf_base,
        prep_report=prep_report,
        metrics=metrics_base,
        seed=42,
        quick=True,
        class_weight="balanced",
        n_estimators=5,
        max_depth=3,
        X_test=X_base,
        clf_abl=clf_abl,
        ablation_metrics=metrics_abl,
    )

    # 1. Baseline artifact saving preserves Destination Port and expected baseline feature count
    baseline_schema_path = tmp_path / "feature_schema.json"
    with open(baseline_schema_path, "r", encoding="utf-8") as f:
        schema_base = json.load(f)
    assert schema_base["feature_count"] == 5
    assert schema_base["ordered_features"] == feature_names
    assert "Destination Port" in schema_base["ordered_features"]
    assert schema_base["model_file"] == "model1_rf.joblib"

    # 2. Ablation artifact saving excludes Destination Port
    abl_schema_path = tmp_path / "feature_schema_no_port.json"
    with open(abl_schema_path, "r", encoding="utf-8") as f:
        schema_abl = json.load(f)
    assert schema_abl["feature_count"] == 4
    assert "Destination Port" not in schema_abl["ordered_features"]
    assert schema_abl["ordered_features"] == ["Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Flow Bytes/s"]
    assert schema_abl["model_file"] == "model1_rf_no_port.joblib"
    assert "Destination Port (ablation)" in schema_abl["cleaning_rules"]["dropped_columns"]

    # 3. Saved baseline model's n_features_in_ matches baseline schema
    loaded_base_model = joblib.load(tmp_path / "model1_rf.joblib")
    assert loaded_base_model.n_features_in_ == schema_base["feature_count"] == 5

    # 4. Saved ablation model's n_features_in_ matches ablation schema
    loaded_abl_model = joblib.load(tmp_path / "model1_rf_no_port.joblib")
    assert loaded_abl_model.n_features_in_ == schema_abl["feature_count"] == 4
    # Models are distinct objects and not identical
    assert loaded_base_model.n_features_in_ != loaded_abl_model.n_features_in_

    # 5. Saved ablation schema contains unique, correctly ordered feature names
    assert len(schema_abl["ordered_features"]) == len(set(schema_abl["ordered_features"]))

    # 6. Validate sample inputs against matching schemas
    sample_base = pd.read_csv(tmp_path / "sample_upload.csv")
    ok_b, cleaned_b, _ = validate_upload(sample_base, schema_base)
    assert ok_b is True
    assert cleaned_b.shape[1] == 5

    sample_abl = pd.read_csv(tmp_path / "sample_upload_no_port.csv")
    ok_a, cleaned_a, _ = validate_upload(sample_abl, schema_abl)
    assert ok_a is True
    assert cleaned_a.shape[1] == 4

    # 7. Baseline and ablation metrics are associated with correct experiment and distinct
    with open(tmp_path / "metrics.json", "r", encoding="utf-8") as f:
        metrics_file = json.load(f)
    assert metrics_file["baseline_artifact"]["model_file"] == "model1_rf.joblib"
    assert metrics_file["baseline_artifact"]["feature_count"] == 5
    assert metrics_file["baseline_artifact"]["includes_destination_port"] is True
    assert metrics_file["test_metrics"]["model_file"] == "model1_rf.joblib"
    assert metrics_file["test_metrics"]["accuracy"] == 0.95

    assert metrics_file["ablation_artifact"]["model_file"] == "model1_rf_no_port.joblib"
    assert metrics_file["ablation_artifact"]["feature_count"] == 4
    assert metrics_file["ablation_artifact"]["includes_destination_port"] is False
    assert metrics_file["ablation_metrics"]["model_file"] == "model1_rf_no_port.joblib"
    assert metrics_file["ablation_metrics"]["accuracy"] == 0.90
    assert metrics_file["ablation_metrics"]["ablation_target"] == "Destination Port"


def test_save_model_artifacts_refuses_swapped_classifier(tmp_path: Path):
    """Verify that accidentally passing baseline model as ablation model triggers an immediate refusal."""
    feature_names = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Flow Bytes/s"]
    prep_report = {
        "ordered_features": feature_names,
        "feature_count": 5,
    }
    np.random.seed(42)
    X_base = pd.DataFrame(np.random.rand(10, 5), columns=feature_names)
    X_abl = X_base.drop(columns=["Destination Port"])
    y = pd.Series([0, 1] * 5)

    clf_base = RandomForestClassifier(n_estimators=3, max_depth=2).fit(X_base, y)
    clf_abl = RandomForestClassifier(n_estimators=3, max_depth=2).fit(X_abl, y)

    dummy_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "majority_class_baseline": 0.5, "confusion_matrix": {"true_negatives": 5, "false_positives": 0, "false_negatives": 1, "true_positives": 4}, "false_positive_rate": 0.0, "false_negative_rate": 0.2, "roc_auc": 0.9, "category_recalls": {}, "top_feature_importances": []}

    # Accidentally passing clf_base (5 features) where clf_abl (4 features) is expected MUST raise ValueError
    with pytest.raises(ValueError, match="Ablation classifier n_features_in_ .* does not match"):
        save_model_artifacts(
            out_dir=tmp_path / "bad_abl",
            clf=clf_base,
            prep_report=prep_report,
            metrics=dummy_metrics,
            clf_abl=clf_base,  # BUG simulation: passing baseline model into ablation path
            ablation_metrics=dummy_metrics,
        )

    # Accidentally passing clf_abl (4 features) where clf (5 features) is expected MUST raise ValueError
    with pytest.raises(ValueError, match="Baseline classifier n_features_in_ .* does not match"):
        save_model_artifacts(
            out_dir=tmp_path / "bad_base",
            clf=clf_abl,  # BUG simulation: passing ablated model into baseline path
            prep_report=prep_report,
            metrics=dummy_metrics,
        )


def test_save_model_artifacts_baseline_only_when_no_ablation(tmp_path: Path):
    """Verify baseline-only execution cleanly omits ablation artifacts."""
    feature_names = ["Destination Port", "Flow Duration", "Total Fwd Packets"]
    prep_report = {"ordered_features": feature_names, "feature_count": 3}
    X = pd.DataFrame(np.random.rand(10, 3), columns=feature_names)
    y = pd.Series([0, 1] * 5)
    clf = RandomForestClassifier(n_estimators=3, max_depth=2).fit(X, y)

    dummy_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.9, "majority_class_baseline": 0.5, "confusion_matrix": {"true_negatives": 5, "false_positives": 0, "false_negatives": 1, "true_positives": 4}, "false_positive_rate": 0.0, "false_negative_rate": 0.2, "roc_auc": 0.9, "category_recalls": {}, "top_feature_importances": []}

    save_model_artifacts(
        out_dir=tmp_path,
        clf=clf,
        prep_report=prep_report,
        metrics=dummy_metrics,
        clf_abl=None,
        ablation_metrics=None,
    )

    assert (tmp_path / "model1_rf.joblib").exists()
    assert (tmp_path / "feature_schema.json").exists()
    assert not (tmp_path / "model1_rf_no_port.joblib").exists()
    assert not (tmp_path / "feature_schema_no_port.json").exists()

    with open(tmp_path / "metrics.json", "r", encoding="utf-8") as f:
        metrics_file = json.load(f)
    assert metrics_file["ablation_artifact"] is None
    assert metrics_file["ablation_metrics"] is None
