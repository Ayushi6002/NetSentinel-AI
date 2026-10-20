"""Unit tests for the Model 1 dataset audit utility (ml/audit_dataset.py)."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.audit_dataset import (
    analyze_column_names,
    audit_single_file,
    compare_schemas,
    find_label_column,
    run_full_audit,
)


def test_analyze_column_names_detects_duplicates_and_whitespace():
    raw_cols = [" Destination Port", " Flow Duration", " Fwd Header Length", " Fwd Header Length", " Label "]
    analysis = analyze_column_names(raw_cols)

    assert analysis["raw_duplicates"] == {" Fwd Header Length": [2, 3]}
    assert analysis["norm_duplicates"] == {"Fwd Header Length": [2, 3]}
    assert len(analysis["whitespace_affected"]) == 5
    assert analysis["normalized_columns"] == [
        "Destination Port",
        "Flow Duration",
        "Fwd Header Length",
        "Fwd Header Length",
        "Label",
    ]


def test_find_label_column_case_insensitive():
    raw_cols = [" Destination Port", " Flow Duration", " lAbEL "]
    idx, raw_name, norm_name = find_label_column(raw_cols)
    assert idx == 2
    assert raw_name == " lAbEL "
    assert norm_name == "lAbEL"


def test_compare_schemas_consistent_and_mismatch():
    meta1 = {
        "filename": "file1.csv",
        "status": "COMPLETED",
        "raw_columns": ["col_a", "col_b", " Label"],
        "normalized_columns": ["col_a", "col_b", "Label"],
    }
    meta2 = {
        "filename": "file2.csv",
        "status": "COMPLETED",
        "raw_columns": ["col_a", "col_b", " Label"],
        "normalized_columns": ["col_a", "col_b", "Label"],
    }
    res_consistent = compare_schemas([meta1, meta2])
    assert res_consistent["status"] == "CONSISTENT"
    assert res_consistent["all_identical_raw"] is True

    meta3 = {
        "filename": "file3.csv",
        "status": "COMPLETED",
        "raw_columns": ["col_a", "col_c", " Label"],
        "normalized_columns": ["col_a", "col_c", "Label"],
    }
    res_mismatch = compare_schemas([meta1, meta3])
    assert res_mismatch["status"] == "MISMATCH_DETECTED"
    assert res_mismatch["all_identical_raw"] is False
    assert len(res_mismatch["mismatches"]) == 1


def test_audit_single_file_with_synthetic_csv(tmp_path: Path):
    csv_file = tmp_path / "test_traffic.csv"
    # Create test CSV with duplicate column name, whitespace, Inf, NaN, and duplicate row
    header = [" Destination Port", "Flow Bytes/s", " Fwd Header Length", " Fwd Header Length", " Label"]
    rows = [
        ["80", "100.5", "20", "20", "BENIGN"],
        ["443", "Infinity", "32", "32", "DDoS"],
        ["80", "100.5", "20", "20", "BENIGN"],  # exact duplicate of row 0
        ["8080", "", "40", "40", "PortScan"],  # missing Flow Bytes/s
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    global_hashes = set()
    result = audit_single_file(csv_file, chunk_size=2, global_hashes=global_hashes)

    assert result["filename"] == "test_traffic.csv"
    assert result["status"] == "COMPLETED"
    assert result["is_complete"] is True
    assert result["row_count"] == 4
    assert result["column_count"] == 5
    assert result["raw_duplicates"] == {" Fwd Header Length": [2, 3]}
    assert result["duplicate_assessment"]["intra_file_duplicates"] == 1
    assert result["duplicate_assessment"]["unique_rows_count"] == 3
    assert result["label_distribution"] == {"BENIGN": 2, "DDoS": 1, "PortScan": 1}

    # Flow Bytes/s should have 1 missing and 1 infinite value
    flow_bytes_col = [c for c in result["columns"] if c["normalized_name"] == "Flow Bytes/s"][0]
    assert flow_bytes_col["missing_count"] == 1
    assert flow_bytes_col["inf_count"] == 1


def test_duplicate_accounting_intra_and_cross_file(tmp_path: Path):
    """Test precise separation of intra-file duplicates vs. cross-file duplicates."""
    data_dir = tmp_path / "multi_file_data"
    output_dir = tmp_path / "reports_dup"
    data_dir.mkdir()

    # File 1: [A, B, A, C]
    # Rows: 4, Unique: 3 (A, B, C), Intra-file dups: 1 (A), Cross-file dups: 0
    file1 = data_dir / "file1.csv"
    with open(file1, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([" Port", " Flow Duration", " Label"])
        writer.writerow(["80", "100", "BENIGN"])    # A
        writer.writerow(["443", "200", "BENIGN"])   # B
        writer.writerow(["80", "100", "BENIGN"])    # A (intra-file dup)
        writer.writerow(["8080", "300", "DDoS"])    # C

    # File 2: [B, D, D]
    # Rows: 3, Unique: 2 (B, D), Intra-file dups: 1 (D), Cross-file dups: 1 (B was in File 1)
    file2 = data_dir / "file2.csv"
    with open(file2, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([" Port", " Flow Duration", " Label"])
        writer.writerow(["443", "200", "BENIGN"])   # B (cross-file dup)
        writer.writerow(["22", "50", "SSH-Patator"]) # D
        writer.writerow(["22", "50", "SSH-Patator"]) # D (intra-file dup)

    summary = run_full_audit(data_dir=data_dir, output_dir=output_dir, chunk_size=10)

    assert summary["audit_metadata"]["overall_audit_status"] == "COMPLETED"
    assert summary["audit_metadata"]["is_complete_audit"] is True

    overview = summary["dataset_overview"]
    assert overview["total_rows_audited"] == 7
    assert overview["total_unique_records_global"] == 4  # A, B, C, D
    assert overview["total_intra_file_duplicates"] == 2  # A in file1, D in file2
    assert overview["total_cross_file_duplicates"] == 1  # B in file2
    assert overview["total_duplicate_records_global"] == 3  # 2 intra + 1 cross

    # Inspect per-file details
    f1_res = [f for f in summary["files"] if f["filename"] == "file1.csv"][0]
    assert f1_res["duplicate_assessment"]["intra_file_duplicates"] == 1
    assert f1_res["duplicate_assessment"]["cross_file_duplicates"] == 0
    assert f1_res["duplicate_assessment"]["unique_rows_count"] == 3

    f2_res = [f for f in summary["files"] if f["filename"] == "file2.csv"][0]
    assert f2_res["duplicate_assessment"]["intra_file_duplicates"] == 1
    assert f2_res["duplicate_assessment"]["cross_file_duplicates"] == 1
    assert f2_res["duplicate_assessment"]["unique_rows_count"] == 2


def test_audit_handles_failed_file_gracefully(tmp_path: Path):
    """Test that empty or unreadable CSV files receive status FAILED and set audit to INCOMPLETE."""
    data_dir = tmp_path / "failed_data"
    output_dir = tmp_path / "reports_failed"
    data_dir.mkdir()

    # Valid file
    valid_file = data_dir / "valid.csv"
    with open(valid_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([" Port", " Label"])
        writer.writerow(["80", "BENIGN"])

    # Empty / unreadable file
    empty_file = data_dir / "empty.csv"
    empty_file.touch()

    summary = run_full_audit(data_dir=data_dir, output_dir=output_dir, chunk_size=10)

    assert summary["audit_metadata"]["overall_audit_status"] == "INCOMPLETE"
    assert summary["audit_metadata"]["is_complete_audit"] is False
    assert "empty.csv" in summary["audit_metadata"]["incomplete_files"]

    empty_res = [f for f in summary["files"] if f["filename"] == "empty.csv"][0]
    assert empty_res["status"] == "FAILED"
    assert empty_res["is_complete"] is False
    assert len(empty_res["errors"]) > 0

    valid_res = [f for f in summary["files"] if f["filename"] == "valid.csv"][0]
    assert valid_res["status"] == "COMPLETED"
    assert valid_res["is_complete"] is True


def test_run_full_audit_generates_all_reports(tmp_path: Path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    data_dir.mkdir()

    csv_file = data_dir / "sample.csv"
    header = [" Port", "Flow Bytes/s", " Label"]
    rows = [["80", "50.0", "BENIGN"], ["443", "10.0", "BENIGN"]]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    summary = run_full_audit(data_dir=data_dir, output_dir=output_dir, chunk_size=10)

    assert summary["dataset_overview"]["total_rows_audited"] == 2
    assert (output_dir / "dataset_audit_summary.json").exists()
    assert (output_dir / "files_summary.csv").exists()
    assert (output_dir / "label_distribution.csv").exists()
    assert (output_dir / "column_anomalies.csv").exists()
    assert (output_dir / "schema_comparison.csv").exists()

    # Verify new fields exist in files_summary.csv
    df_files = pd.read_csv(output_dir / "files_summary.csv")
    assert "status" in df_files.columns
    assert "intra_file_duplicates" in df_files.columns
    assert "cross_file_duplicates" in df_files.columns
