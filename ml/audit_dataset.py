"""
NetSentinel-AI Model 1 - Read-Only Dataset Audit Utility
=========================================================

Audits all raw CSV flow files in the CICIDS2017 dataset directory
(default: data/raw/MachineLearningCVE/) without modifying, moving,
or deleting source files.

Features & Compliance:
1. Strictly read-only operations on source files.
2. Chunk-based streaming via pandas to maintain a minimal memory footprint.
3. Reports file size, row count, column count, and exact original column names.
4. Distinguishes raw column names, normalized (whitespace-trimmed) names,
   and identifies duplicates hidden by pandas' automatic column mangling (.1, .2).
5. Identifies label column case-insensitively and computes per-file and
   dataset-wide label distributions (handling special characters like U+FFFD).
6. Identifies missing values (NaN), infinite numeric values (Inf/-Inf), and
   non-numeric invalid values per column and affected rows.
7. Performs cross-file schema comparisons (missing, extra, mismatched columns).
8. Assesses duplicate records via streaming 64-bit row hashing:
   - Differentiates intra-file duplicates from cross-file duplicates.
   - Explicitly framed as a 64-bit streaming hash assessment (avoiding claims of
     mathematical exactness or treating collision bounds as guarantees).
   - Compares against within-chunk lower-bound checks.
9. Resilient to encoding issues (using encoding_errors='replace' and fallback
   logging without silently presenting corrupt data as fully audited).
10. Explicit per-file statuses (COMPLETED, PARTIAL, FAILED) and overall audit
    status (COMPLETED, INCOMPLETE). Does not present partial counts as complete totals.
11. Emits a clean terminal summary and saves machine-readable JSON and CSV
    reports under `reports/model1_dataset_audit/`.
12. Does not train models, sample data, modify Phase 1 code, or commit raw data.

Usage:
------
Default execution:
    python ml/audit_dataset.py

Custom options:
    python ml/audit_dataset.py --data-dir data/raw/MachineLearningCVE \\
                               --output-dir reports/model1_dataset_audit \\
                               --chunk-size 100000
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Ensure standard output can handle Unicode characters (such as \ufffd in CICIDS2017)
# on Windows consoles without throwing UnicodeEncodeError.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def detect_encoding(file_path: Path) -> str:
    """Detect file encoding with safe fallback candidates.

    Tries utf-8, utf-8-sig, latin-1, and cp1252 on the initial file bytes.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                for _ in range(50):
                    if not f.readline():
                        break
            return enc
        except (UnicodeDecodeError, OSError):
            continue
    return "latin-1"  # Latin-1 accepts all single-byte values


def extract_raw_header(file_path: Path, encoding: str) -> List[str]:
    """Extract raw column header strings directly from the first line of the CSV.

    Bypasses pandas automatic column mangling to preserve the true original header.
    """
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        header_line = f.readline()
    if not header_line:
        return []
    reader = csv.reader([header_line])
    return next(reader, [])


def analyze_column_names(raw_columns: List[str]) -> Dict[str, Any]:
    """Analyze raw and normalized column names to detect whitespace anomalies,

    exact duplicates in the raw header, and duplicates after trimming.
    """
    normalized_columns = [col.strip() for col in raw_columns]

    # Check for exact duplicate raw column names
    raw_counts = Counter(raw_columns)
    raw_duplicates = {col: [idx for idx, c in enumerate(raw_columns) if c == col]
                      for col, count in raw_counts.items() if count > 1}

    # Check for duplicate normalized column names
    norm_counts = Counter(normalized_columns)
    norm_duplicates = {col: [idx for idx, c in enumerate(normalized_columns) if c == col]
                       for col, count in norm_counts.items() if count > 1}

    # Identify columns with leading or trailing whitespace
    whitespace_affected = [
        {"index": idx, "raw": col, "normalized": col.strip()}
        for idx, col in enumerate(raw_columns) if col != col.strip()
    ]

    return {
        "raw_columns": raw_columns,
        "normalized_columns": normalized_columns,
        "raw_duplicates": raw_duplicates,
        "norm_duplicates": norm_duplicates,
        "whitespace_affected": whitespace_affected,
    }


def find_label_column(raw_columns: List[str]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Locate the label column case-insensitively after whitespace trimming.

    Returns:
        (column_index, raw_name, normalized_name)
    """
    for idx, col in enumerate(raw_columns):
        stripped = col.strip().lower()
        if stripped == "label":
            return idx, col, col.strip()

    # Fallback: check if 'label' is in stripped column name
    for idx, col in enumerate(raw_columns):
        if "label" in col.strip().lower():
            return idx, col, col.strip()

    return None, None, None


def compare_schemas(files_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare schemas across all audited CSV files.

    Checks for column count mismatches, raw name mismatches, and normalized mismatches.
    """
    valid_files = [f for f in files_meta if f.get("status") in ("COMPLETED", "PARTIAL") and f.get("raw_columns")]
    if not valid_files:
        return {"status": "NO_VALID_FILES", "mismatches": []}

    reference = valid_files[0]
    ref_file = reference["filename"]
    ref_raw = reference["raw_columns"]
    ref_norm = reference["normalized_columns"]

    all_identical_raw = True
    all_identical_norm = True
    mismatches = []

    for file_info in valid_files:
        fname = file_info["filename"]
        curr_raw = file_info["raw_columns"]
        curr_norm = file_info["normalized_columns"]

        raw_match = (curr_raw == ref_raw)
        norm_match = (curr_norm == ref_norm)

        if not raw_match:
            all_identical_raw = False
        if not norm_match:
            all_identical_norm = False

        diff_details = {}
        if not raw_match:
            missing_in_curr = [c for c in ref_raw if c not in curr_raw]
            extra_in_curr = [c for c in curr_raw if c not in ref_raw]
            diff_details["raw_missing_vs_ref"] = missing_in_curr
            diff_details["raw_extra_vs_ref"] = extra_in_curr
            if len(curr_raw) != len(ref_raw):
                diff_details["column_count_difference"] = {
                    "expected": len(ref_raw),
                    "actual": len(curr_raw)
                }

        if not norm_match:
            missing_norm = [c for c in ref_norm if c not in curr_norm]
            extra_norm = [c for c in curr_norm if c not in ref_norm]
            diff_details["norm_missing_vs_ref"] = missing_norm
            diff_details["norm_extra_vs_ref"] = extra_norm

        if diff_details:
            mismatches.append({
                "file": fname,
                "differences": diff_details
            })

    return {
        "status": "CONSISTENT" if (all_identical_raw and all_identical_norm) else "MISMATCH_DETECTED",
        "reference_file": ref_file,
        "reference_column_count": len(ref_raw),
        "all_identical_raw": all_identical_raw,
        "all_identical_normalized": all_identical_norm,
        "mismatches": mismatches,
    }


def audit_single_file(
    file_path: Path,
    chunk_size: int,
    global_hashes: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """Audit a single CSV file in chunks using pandas.

    Returns detailed metadata, column quality stats, label frequencies,
    duplicate assessments (intra-file and cross-file), and status (COMPLETED, PARTIAL, FAILED).
    """
    file_name = file_path.name
    try:
        file_size_bytes = file_path.stat().st_size
    except OSError:
        file_size_bytes = 0
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    encoding = detect_encoding(file_path)
    file_errors: List[str] = []
    file_warnings: List[str] = []

    try:
        raw_columns = extract_raw_header(file_path, encoding)
    except Exception as e:
        raw_columns = []
        file_errors.append(f"Failed to extract header: {type(e).__name__}: {str(e)}")

    if not raw_columns:
        return {
            "filename": file_name,
            "file_path": str(file_path.resolve()),
            "status": "FAILED",
            "is_complete": False,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb,
            "encoding": encoding,
            "row_count": 0,
            "column_count": 0,
            "raw_columns": [],
            "normalized_columns": [],
            "raw_duplicates": {},
            "norm_duplicates": {},
            "whitespace_affected_columns": [],
            "pandas_mangled_differences": [],
            "label_column": {"index": None, "raw_name": None, "normalized_name": None},
            "label_distribution": {},
            "duplicate_assessment": {
                "within_chunk_duplicates": 0,
                "intra_file_duplicates": 0,
                "cross_file_duplicates": 0,
                "unique_rows_count": 0,
                "intra_file_duplicate_pct": 0.0,
                "assessment_methodology": "64-bit streaming hash assessment (pandas hash_pandas_object)",
            },
            "quality_metrics": {
                "total_missing_values": 0,
                "total_inf_values": 0,
                "total_invalid_numeric_values": 0,
                "anomalous_rows_count": 0,
                "anomalous_rows_pct": 0.0,
                "replacement_chars_detected": 0,
            },
            "columns": [],
            "chunks_processed": 0,
            "errors": file_errors or ["Empty or unreadable CSV header."],
            "warnings": file_warnings,
        }

    col_analysis = analyze_column_names(raw_columns)
    label_idx, label_raw_col, label_norm_col = find_label_column(raw_columns)

    num_cols = len(raw_columns)
    col_missing_counts = [0] * num_cols
    col_inf_counts = [0] * num_cols
    col_invalid_numeric_counts = [0] * num_cols
    col_inferred_dtypes = ["unknown"] * num_cols

    total_rows = 0
    total_anomalous_rows = 0
    replacement_chars_detected = 0
    label_counter: Counter = Counter()

    # Duplicate assessment accumulators:
    # 1. within_chunk_duplicates: partial check / lower bound strictly within individual chunks
    # 2. intra_file_duplicates: 64-bit hash duplicates within this file
    # 3. cross_file_duplicates: unique rows in this file that already appeared in earlier audited files
    within_chunk_duplicates = 0
    intra_file_duplicates = 0
    cross_file_duplicates = 0
    file_hashes: Set[int] = set()

    pandas_mangled_columns: List[str] = []
    chunks_processed = 0
    processing_aborted = False

    try:
        # Create chunk iterator with encoding_errors='replace' for resilience
        chunk_iter = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            encoding=encoding,
            encoding_errors="replace",
            low_memory=False,
            on_bad_lines="warn"
        )

        for chunk_idx, chunk in enumerate(chunk_iter):
            chunk_len = len(chunk)
            if chunk_len == 0:
                continue

            chunks_processed += 1
            total_rows += chunk_len

            if chunk_idx == 0:
                pandas_mangled_columns = list(chunk.columns)

            # Within-chunk exact duplicate count (partial lower bound)
            within_chunk_duplicates += int(chunk.duplicated().sum())

            # 64-bit row hash duplicate assessment
            row_hashes = pd.util.hash_pandas_object(chunk, index=False)
            for h in row_hashes:
                h_val = int(h)
                if h_val in file_hashes:
                    intra_file_duplicates += 1
                else:
                    file_hashes.add(h_val)
                    # Check if this row was already seen in a prior file
                    if global_hashes is not None:
                        if h_val in global_hashes:
                            cross_file_duplicates += 1
                        else:
                            global_hashes.add(h_val)

            # Label frequency
            if label_idx is not None and label_idx < len(chunk.columns):
                actual_label_col = chunk.columns[label_idx]
                labels_in_chunk = chunk[actual_label_col].value_counts(dropna=False)
                for label_val, count in labels_in_chunk.items():
                    str_label = str(label_val) if pd.notna(label_val) else "<MISSING_LABEL>"
                    if "\ufffd" in str_label:
                        replacement_chars_detected += int(count)
                    label_counter[str_label] += int(count)

            # Column-wise missing, infinite, and invalid numeric values
            chunk_row_anomaly_mask = np.zeros(chunk_len, dtype=bool)

            for col_i, col_name in enumerate(chunk.columns):
                if col_i >= num_cols:
                    break

                series = chunk[col_name]
                col_inferred_dtypes[col_i] = str(series.dtype)

                na_mask = series.isna()
                na_count = int(na_mask.sum())
                col_missing_counts[col_i] += na_count

                is_label = (col_i == label_idx)
                if not is_label:
                    if np.issubdtype(series.dtype, np.number):
                        inf_mask = np.isinf(series)
                        inf_count = int(inf_mask.sum())
                        col_inf_counts[col_i] += inf_count
                        chunk_row_anomaly_mask |= na_mask | inf_mask
                    else:
                        numeric_series = pd.to_numeric(series, errors="coerce")
                        inf_mask = np.isinf(numeric_series)
                        inf_count = int(inf_mask.sum())
                        col_inf_counts[col_i] += inf_count

                        invalid_mask = series.notna() & numeric_series.isna()
                        invalid_count = int(invalid_mask.sum())
                        col_invalid_numeric_counts[col_i] += invalid_count

                        chunk_row_anomaly_mask |= na_mask | inf_mask | invalid_mask

            total_anomalous_rows += int(chunk_row_anomaly_mask.sum())

    except Exception as e:
        processing_aborted = True
        file_errors.append(f"Error during CSV chunk processing: {type(e).__name__}: {str(e)}")

    if replacement_chars_detected > 0:
        file_warnings.append(
            f"Replacement character U+FFFD detected in {replacement_chars_detected:,} label instances. "
            "Raw file contains non-standard characters."
        )

    # Determine explicit file status
    if processing_aborted:
        status = "PARTIAL" if chunks_processed > 0 else "FAILED"
    else:
        status = "COMPLETED"

    is_complete = (status == "COMPLETED")

    # Construct per-column summary
    columns_summary = []
    for idx in range(num_cols):
        raw_name = raw_columns[idx]
        norm_name = col_analysis["normalized_columns"][idx]
        mangled_name = pandas_mangled_columns[idx] if idx < len(pandas_mangled_columns) else raw_name
        missing_cnt = col_missing_counts[idx]
        missing_pct = round((missing_cnt / total_rows * 100), 4) if total_rows > 0 else 0.0
        inf_cnt = col_inf_counts[idx]
        inf_pct = round((inf_cnt / total_rows * 100), 4) if total_rows > 0 else 0.0
        invalid_cnt = col_invalid_numeric_counts[idx]

        columns_summary.append({
            "column_index": idx,
            "raw_name": raw_name,
            "normalized_name": norm_name,
            "pandas_mangled_name": mangled_name,
            "dtype": col_inferred_dtypes[idx],
            "is_raw_duplicate": raw_name in col_analysis["raw_duplicates"],
            "is_norm_duplicate": norm_name in col_analysis["norm_duplicates"],
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "inf_count": inf_cnt,
            "inf_pct": inf_pct,
            "invalid_numeric_count": invalid_cnt,
        })

    mangled_differences = []
    for idx in range(min(len(raw_columns), len(pandas_mangled_columns))):
        if raw_columns[idx] != pandas_mangled_columns[idx]:
            mangled_differences.append({
                "column_index": idx,
                "raw_name": raw_columns[idx],
                "pandas_name": pandas_mangled_columns[idx]
            })

    unique_rows_file = len(file_hashes)

    return {
        "filename": file_name,
        "file_path": str(file_path.resolve()),
        "status": status,
        "is_complete": is_complete,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "encoding": encoding,
        "row_count": total_rows,
        "column_count": num_cols,
        "chunks_processed": chunks_processed,
        "raw_columns": raw_columns,
        "normalized_columns": col_analysis["normalized_columns"],
        "raw_duplicates": col_analysis["raw_duplicates"],
        "norm_duplicates": col_analysis["norm_duplicates"],
        "whitespace_affected_columns": col_analysis["whitespace_affected"],
        "pandas_mangled_differences": mangled_differences,
        "label_column": {
            "index": label_idx,
            "raw_name": label_raw_col,
            "normalized_name": label_norm_col,
        },
        "label_distribution": dict(label_counter),
        "duplicate_assessment": {
            "within_chunk_duplicates": within_chunk_duplicates,
            "within_chunk_note": "Partial check / lower bound (exact duplicates within each individual chunk window)",
            "intra_file_duplicates": intra_file_duplicates,
            "cross_file_duplicates": cross_file_duplicates,
            "unique_rows_count": unique_rows_file,
            "intra_file_duplicate_pct": round((intra_file_duplicates / total_rows * 100), 2) if total_rows > 0 else 0.0,
            "assessment_methodology": "64-bit streaming row hash assessment (pandas hash_pandas_object)",
            "methodology_note": (
                "Identifies matching rows across streaming chunks via 64-bit integer hashes without storing full rows in RAM. "
                "This is a high-confidence hash-based assessment rather than a memory-exhaustive byte-by-byte comparison guarantee."
            ),
        },
        "quality_metrics": {
            "total_missing_values": sum(col_missing_counts),
            "total_inf_values": sum(col_inf_counts),
            "total_invalid_numeric_values": sum(col_invalid_numeric_counts),
            "anomalous_rows_count": total_anomalous_rows,
            "anomalous_rows_pct": round((total_anomalous_rows / total_rows * 100), 4) if total_rows > 0 else 0.0,
            "replacement_chars_detected": replacement_chars_detected,
        },
        "columns": columns_summary,
        "errors": file_errors,
        "warnings": file_warnings,
    }


def run_full_audit(
    data_dir: Path,
    output_dir: Path,
    chunk_size: int = 100000
) -> Dict[str, Any]:
    """Execute complete dataset audit on all CSV files in data_dir,

    outputting terminal reports and writing machine-readable JSON/CSV artifacts.
    """
    start_time = time.time()
    csv_files = sorted(list(data_dir.glob("*.csv")))

    if not csv_files:
        print(f"[!] Warning: No CSV files found in {data_dir}", flush=True)
        return {
            "audit_status": "NO_CSV_FILES",
            "is_complete_audit": False,
            "data_dir": str(data_dir),
            "files": [],
        }

    print("=" * 90, flush=True)
    print("NetSentinel-AI Model 1: Read-Only Dataset Audit", flush=True)
    print(f"Dataset Location : {data_dir.resolve()}", flush=True)
    print(f"CSV Files Found  : {len(csv_files)}", flush=True)
    print(f"Chunk Size       : {chunk_size:,} rows", flush=True)
    print(f"Audit Start Time : {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("=" * 90, flush=True)

    files_audit_results: List[Dict[str, Any]] = []
    global_hashes: Set[int] = set()
    global_label_counter: Counter = Counter()

    total_dataset_rows = 0
    total_dataset_bytes = 0
    total_intra_file_duplicates = 0
    total_cross_file_duplicates = 0

    # Stream-audit each file individually
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"\n[{idx}/{len(csv_files)}] Auditing: {csv_file.name} ...", flush=True)
        file_start = time.time()

        res = audit_single_file(csv_file, chunk_size, global_hashes=global_hashes)
        files_audit_results.append(res)

        total_dataset_rows += res["row_count"]
        total_dataset_bytes += res["file_size_bytes"]
        total_intra_file_duplicates += res["duplicate_assessment"]["intra_file_duplicates"]
        total_cross_file_duplicates += res["duplicate_assessment"]["cross_file_duplicates"]

        for lbl, cnt in res["label_distribution"].items():
            global_label_counter[lbl] += cnt

        file_elapsed = time.time() - file_start
        status_flag = f"[{res['status']}]"
        print(
            f"    -> {status_flag} in {file_elapsed:.2f}s | Rows: {res['row_count']:,} | "
            f"Size: {res['file_size_mb']} MB | "
            f"Intra-File Dups: {res['duplicate_assessment']['intra_file_duplicates']:,} | "
            f"Cross-File Dups: {res['duplicate_assessment']['cross_file_duplicates']:,} | "
            f"Missing: {res['quality_metrics']['total_missing_values']:,} | "
            f"Infs: {res['quality_metrics']['total_inf_values']:,}",
            flush=True
        )
        if res["errors"]:
            for err in res["errors"]:
                print(f"       [!] Error: {err}", flush=True)
        if res["warnings"]:
            for warn in res["warnings"]:
                print(f"       [*] Warning: {warn}", flush=True)

    # Schema consistency analysis
    schema_results = compare_schemas(files_audit_results)

    # Determine overall audit status
    failed_or_partial_files = [
        f["filename"] for f in files_audit_results if f["status"] != "COMPLETED"
    ]
    is_complete_audit = (len(failed_or_partial_files) == 0)
    overall_status = "COMPLETED" if is_complete_audit else "INCOMPLETE"

    # Global duplicate metrics
    total_unique_global_rows = len(global_hashes)
    total_global_duplicates = total_intra_file_duplicates + total_cross_file_duplicates
    global_duplicate_pct = (
        round((total_global_duplicates / total_dataset_rows * 100), 2)
        if total_dataset_rows > 0 else 0.0
    )

    total_audit_time = round(time.time() - start_time, 2)

    # Master summary payload
    audit_summary: Dict[str, Any] = {
        "audit_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": total_audit_time,
            "overall_audit_status": overall_status,
            "is_complete_audit": is_complete_audit,
            "incomplete_files": failed_or_partial_files,
            "python_version": sys.version,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "chunk_size": chunk_size,
            "data_directory": str(data_dir.resolve()),
            "output_directory": str(output_dir.resolve()),
        },
        "dataset_overview": {
            "file_count": len(csv_files),
            "files_completed": sum(1 for f in files_audit_results if f["status"] == "COMPLETED"),
            "files_partial": sum(1 for f in files_audit_results if f["status"] == "PARTIAL"),
            "files_failed": sum(1 for f in files_audit_results if f["status"] == "FAILED"),
            "total_size_bytes": total_dataset_bytes,
            "total_size_mb": round(total_dataset_bytes / (1024 * 1024), 2),
            "total_rows_audited": total_dataset_rows,
            "rows_accounting_note": (
                "Complete total of all records across all 8 files."
                if is_complete_audit else
                "PARTIAL TOTAL: One or more files failed or terminated prematurely during processing."
            ),
            "total_unique_records_global": total_unique_global_rows,
            "total_intra_file_duplicates": total_intra_file_duplicates,
            "total_cross_file_duplicates": total_cross_file_duplicates,
            "total_duplicate_records_global": total_global_duplicates,
            "global_duplicate_percentage": global_duplicate_pct,
            "duplicate_assessment_methodology": {
                "assessment_type": "64-bit streaming row hash assessment (pandas hash_pandas_object)",
                "technical_details": (
                    "Tracks uint64 row hashes in streaming hash sets across chunk boundaries. "
                    "This achieves high collision resistance for multi-million row datasets without storing full raw records in memory. "
                    "It is framed as a high-confidence hash assessment rather than a memory-exhaustive byte-by-byte equality guarantee."
                ),
                "intra_file_definition": "Duplicate records occurring within the same CSV file.",
                "cross_file_definition": "Records in a CSV that duplicate records already observed in preceding CSV files.",
                "within_chunk_definition": "Lower-bound check counting duplicates strictly within isolated chunk buffers.",
            },
        },
        "schema_audit": schema_results,
        "global_label_distribution": [
            {
                "label": lbl,
                "count": cnt,
                "percentage": round((cnt / total_dataset_rows * 100), 4) if total_dataset_rows > 0 else 0.0
            }
            for lbl, cnt in global_label_counter.most_common()
        ],
        "files": files_audit_results,
    }

    # Save reports
    save_reports(audit_summary, output_dir)

    # Print terminal summary
    print_terminal_summary(audit_summary)

    return audit_summary


def save_reports(audit_summary: Dict[str, Any], output_dir: Path) -> None:
    """Write structured JSON and CSV reports under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comprehensive JSON Report
    json_path = output_dir / "dataset_audit_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    # 2. Files Summary CSV
    files_csv_path = output_dir / "files_summary.csv"
    files_rows = []
    for f in audit_summary["files"]:
        files_rows.append({
            "file_name": f["filename"],
            "status": f["status"],
            "is_complete": f["is_complete"],
            "file_size_mb": f["file_size_mb"],
            "file_size_bytes": f["file_size_bytes"],
            "row_count": f["row_count"],
            "column_count": f["column_count"],
            "unique_rows": f["duplicate_assessment"]["unique_rows_count"],
            "intra_file_duplicates": f["duplicate_assessment"]["intra_file_duplicates"],
            "intra_file_duplicate_pct": f["duplicate_assessment"]["intra_file_duplicate_pct"],
            "cross_file_duplicates": f["duplicate_assessment"]["cross_file_duplicates"],
            "within_chunk_duplicates": f["duplicate_assessment"]["within_chunk_duplicates"],
            "total_missing_values": f["quality_metrics"]["total_missing_values"],
            "total_inf_values": f["quality_metrics"]["total_inf_values"],
            "total_invalid_numeric": f["quality_metrics"]["total_invalid_numeric_values"],
            "anomalous_rows_count": f["quality_metrics"]["anomalous_rows_count"],
            "anomalous_rows_pct": f["quality_metrics"]["anomalous_rows_pct"],
            "replacement_chars_detected": f["quality_metrics"]["replacement_chars_detected"],
            "raw_duplicate_columns_count": len(f["raw_duplicates"]),
            "pandas_mangled_columns_count": len(f["pandas_mangled_differences"]),
            "encoding": f["encoding"],
            "errors_count": len(f["errors"]),
            "warnings_count": len(f["warnings"]),
        })
    pd.DataFrame(files_rows).to_csv(files_csv_path, index=False, encoding="utf-8")

    # 3. Label Distribution CSV
    labels_csv_path = output_dir / "label_distribution.csv"
    label_rows = []
    for f in audit_summary["files"]:
        f_total = f["row_count"]
        for lbl, cnt in f["label_distribution"].items():
            label_rows.append({
                "scope": f["filename"],
                "file_status": f["status"],
                "label": lbl,
                "count": cnt,
                "percentage": round((cnt / f_total * 100), 4) if f_total > 0 else 0.0
            })
    for item in audit_summary["global_label_distribution"]:
        label_rows.append({
            "scope": "ALL_FILES_GLOBAL",
            "file_status": audit_summary["audit_metadata"]["overall_audit_status"],
            "label": item["label"],
            "count": item["count"],
            "percentage": item["percentage"]
        })
    pd.DataFrame(label_rows).to_csv(labels_csv_path, index=False, encoding="utf-8")

    # 4. Column Anomalies & Quality CSV
    col_csv_path = output_dir / "column_anomalies.csv"
    col_rows = []
    for f in audit_summary["files"]:
        for c in f["columns"]:
            if (c["missing_count"] > 0 or c["inf_count"] > 0 or
                    c["invalid_numeric_count"] > 0 or c["is_raw_duplicate"] or c["is_norm_duplicate"]):
                col_rows.append({
                    "file_name": f["filename"],
                    "file_status": f["status"],
                    "column_index": c["column_index"],
                    "raw_name": c["raw_name"],
                    "normalized_name": c["normalized_name"],
                    "pandas_mangled_name": c["pandas_mangled_name"],
                    "dtype": c["dtype"],
                    "is_raw_duplicate": c["is_raw_duplicate"],
                    "is_norm_duplicate": c["is_norm_duplicate"],
                    "missing_count": c["missing_count"],
                    "missing_pct": c["missing_pct"],
                    "inf_count": c["inf_count"],
                    "inf_pct": c["inf_pct"],
                    "invalid_numeric_count": c["invalid_numeric_count"],
                })
    pd.DataFrame(col_rows).to_csv(col_csv_path, index=False, encoding="utf-8")

    # 5. Schema Comparison CSV
    schema_csv_path = output_dir / "schema_comparison.csv"
    schema_rows = []
    valid_files = [f for f in audit_summary["files"] if f["status"] in ("COMPLETED", "PARTIAL") and f["columns"]]
    ref_columns = valid_files[0]["columns"] if valid_files else []
    for col_info in ref_columns:
        schema_rows.append({
            "column_index": col_info["column_index"],
            "reference_raw_name": col_info["raw_name"],
            "reference_normalized_name": col_info["normalized_name"],
            "pandas_mangled_name": col_info["pandas_mangled_name"],
            "is_raw_duplicate": col_info["is_raw_duplicate"],
            "is_norm_duplicate": col_info["is_norm_duplicate"],
            "all_files_consistent": audit_summary["schema_audit"].get("all_identical_raw", False),
        })
    pd.DataFrame(schema_rows).to_csv(schema_csv_path, index=False, encoding="utf-8")


def print_terminal_summary(summary: Dict[str, Any]) -> None:
    """Print a clean, structured terminal summary of the audit."""
    ov = summary["dataset_overview"]
    meta = summary["audit_metadata"]
    schema = summary["schema_audit"]

    print("\n" + "=" * 94, flush=True)
    print("                      DATASET AUDIT EXECUTIVE SUMMARY", flush=True)
    print("=" * 94, flush=True)

    if not meta["is_complete_audit"]:
        print("**********************************************************************************************", flush=True)
        print(" [!] WARNING: AUDIT STATUS IS INCOMPLETE", flush=True)
        print(f"     Incomplete Files ({len(meta['incomplete_files'])}): {', '.join(meta['incomplete_files'])}", flush=True)
        print("     Total rows and percentages below represent partial data only.", flush=True)
        print("**********************************************************************************************", flush=True)

    print(f"Overall Audit Status: {meta['overall_audit_status']}", flush=True)
    print(f"Files Completed     : {ov['files_completed']} of {ov['file_count']} (Partial: {ov['files_partial']}, Failed: {ov['files_failed']})", flush=True)
    print(f"Total Dataset Size  : {ov['total_size_mb']:,} MB ({ov['total_size_bytes']:,} bytes)", flush=True)
    print(f"Total Rows Audited  : {ov['total_rows_audited']:,}", flush=True)
    print(f"Global Unique Rows  : {ov['total_unique_records_global']:,}", flush=True)
    print(f"Intra-File Dups     : {ov['total_intra_file_duplicates']:,} (duplicates within the same file)", flush=True)
    print(f"Cross-File Dups     : {ov['total_cross_file_duplicates']:,} (rows repeated across multiple files)", flush=True)
    print(f"Total Duplicates    : {ov['total_duplicate_records_global']:,} ({ov['global_duplicate_percentage']}%)", flush=True)
    print(f"Execution Time      : {meta['execution_time_seconds']} seconds", flush=True)
    print("-" * 94, flush=True)

    # Schema Status
    print("1. SCHEMA CONSISTENCY & DUPLICATE COLUMN ANALYSIS:", flush=True)
    print(f"   Schema Status Across Files  : {schema['status']}", flush=True)
    print(f"   Reference Columns Count     : {schema.get('reference_column_count', 'N/A')}", flush=True)
    print(f"   All Raw Headers Match Exactly: {schema.get('all_identical_raw', False)}", flush=True)
    print(f"   All Stripped Headers Match  : {schema.get('all_identical_normalized', False)}", flush=True)

    valid_files = [f for f in summary["files"] if f["status"] in ("COMPLETED", "PARTIAL")]
    ref_file = valid_files[0] if valid_files else None
    if ref_file and ref_file["raw_duplicates"]:
        print("   [!] Duplicate Column Names in Raw CSV Header:", flush=True)
        for dup_col, indices in ref_file["raw_duplicates"].items():
            print(f"       * {repr(dup_col)} appears {len(indices)} times at indices {indices}", flush=True)
    if ref_file and ref_file["pandas_mangled_differences"]:
        print("   [!] Pandas Column Mangling Detected:", flush=True)
        for m in ref_file["pandas_mangled_differences"]:
            print(f"       * Index {m['column_index']}: Raw {repr(m['raw_name'])} -> Mangled as {repr(m['pandas_name'])}", flush=True)

    print("-" * 94, flush=True)

    # File Summary Table
    print("2. PER-FILE INVENTORY & DATA QUALITY SUMMARY:", flush=True)
    header = f"{'Filename':<36} {'Status':<10} {'Size MB':>8} {'Rows':>10} {'IntraDup':>9} {'CrossDup':>9} {'Miss':>6} {'Infs':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for f in summary["files"]:
        print(
            f"{f['filename'][:34]:<36} "
            f"{f['status']:<10} "
            f"{f['file_size_mb']:>8.1f} "
            f"{f['row_count']:>10,} "
            f"{f['duplicate_assessment']['intra_file_duplicates']:>9,} "
            f"{f['duplicate_assessment']['cross_file_duplicates']:>9,} "
            f"{f['quality_metrics']['total_missing_values']:>6,} "
            f"{f['quality_metrics']['total_inf_values']:>6,}",
            flush=True
        )
    print("-" * 94, flush=True)

    # Duplicate Methodology Breakdown
    print("3. DUPLICATE RECORDS METHODOLOGY & BREAKDOWN:", flush=True)
    print("   * Assessment Methodology: 64-bit Streaming Hash Assessment (pandas hash_pandas_object)", flush=True)
    print("     - High-confidence hash detection across chunks without caching complete DataFrames in memory.", flush=True)
    print("     - Intra-file duplicates: rows matching an earlier row within the SAME CSV file.", flush=True)
    print("     - Cross-file duplicates: unique rows in a file that match a row in an EARLIER CSV file.", flush=True)
    print("     - Within-chunk duplicates: lower bound (only captures duplicate rows inside the same chunk buffer).", flush=True)
    print("   * Comparison by file:", flush=True)
    for f in summary["files"]:
        intra_d = f["duplicate_assessment"]["intra_file_duplicates"]
        cross_d = f["duplicate_assessment"]["cross_file_duplicates"]
        chunk_d = f["duplicate_assessment"]["within_chunk_duplicates"]
        print(
            f"     - {f['filename'][:33]:<34} Intra: {intra_d:>7,} | Cross: {cross_d:>7,} | Within-Chunk (Lower Bound): {chunk_d:>7,}",
            flush=True
        )
    print("-" * 94, flush=True)

    # Column Quality Anomalies Summary
    print("4. COLUMN-LEVEL ANOMALIES (MISSING & INFINITE VALUES):", flush=True)
    total_missing = sum(f["quality_metrics"]["total_missing_values"] for f in summary["files"])
    total_infs = sum(f["quality_metrics"]["total_inf_values"] for f in summary["files"])
    print(f"   Total Missing (NaN) Values Across Dataset : {total_missing:,}", flush=True)
    print(f"   Total Infinite (Inf) Values Across Dataset: {total_infs:,}", flush=True)

    col_anomaly_agg: Dict[str, Dict[str, int]] = {}
    for f in summary["files"]:
        for c in f["columns"]:
            norm = c["normalized_name"]
            if c["missing_count"] > 0 or c["inf_count"] > 0:
                if norm not in col_anomaly_agg:
                    col_anomaly_agg[norm] = {"missing": 0, "inf": 0}
                col_anomaly_agg[norm]["missing"] += c["missing_count"]
                col_anomaly_agg[norm]["inf"] += c["inf_count"]

    if col_anomaly_agg:
        print("   Affected Columns Breakdown:", flush=True)
        for col_name, counts in col_anomaly_agg.items():
            print(f"     * {col_name:<30}: Missing={counts['missing']:,} | Infinite={counts['inf']:,}", flush=True)
    else:
        print("   No missing or infinite values found.", flush=True)
    print("-" * 94, flush=True)

    # Global Label Distribution Table
    print("5. GLOBAL LABEL DISTRIBUTION (ALL FILES COMBINED):", flush=True)
    lbl_header = f"{'Label Name':<35} {'Record Count':>15} {'Percentage':>12}"
    print(lbl_header, flush=True)
    print("-" * len(lbl_header), flush=True)
    for item in summary["global_label_distribution"]:
        print(f"{item['label']:<35} {item['count']:>15,} {item['percentage']:>11.4f}%", flush=True)
    print("-" * 94, flush=True)

    # Generated Reports List
    print("6. GENERATED AUDIT ARTIFACTS:", flush=True)
    out_dir = Path(meta["output_directory"])
    print(f"   Reports Directory: {out_dir.resolve()}", flush=True)
    print(f"   - JSON Summary   : {out_dir / 'dataset_audit_summary.json'}", flush=True)
    print(f"   - Files Summary  : {out_dir / 'files_summary.csv'}", flush=True)
    print(f"   - Label Summary  : {out_dir / 'label_distribution.csv'}", flush=True)
    print(f"   - Anomalies CSV  : {out_dir / 'column_anomalies.csv'}", flush=True)
    print(f"   - Schema CSV     : {out_dir / 'schema_comparison.csv'}", flush=True)
    print("=" * 94 + "\n", flush=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NetSentinel-AI Model 1: Read-Only Dataset Audit Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/MachineLearningCVE"),
        help="Path to directory containing raw CSV dataset files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/model1_dataset_audit"),
        help="Path to directory where audit reports will be stored."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows per chunk for streaming CSV processing."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for command line execution."""
    args = parse_arguments()

    if not args.data_dir.exists():
        print(f"[!] Error: Data directory does not exist: {args.data_dir.resolve()}", flush=True)
        sys.exit(1)

    run_full_audit(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
