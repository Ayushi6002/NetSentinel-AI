"""
NetSentinel-AI Model 1 - Offline Binary Attack Classifier Data Preparation
==========================================================================

Pipeline Execution Order:
-------------------------
1. read in chunks
2. normalize headers (strip whitespace, detect duplicates)
3. resolve duplicate columns (drop duplicate copies verified identical across raw data)
4. normalize labels (standardize Web Attack variants and special characters)
5. numeric coercion (convert features to numeric, replace inf/-inf with NaN)
6. drop NaN/inf rows & drop negative Flow Duration rows
7. dedup (within-file AND cross-file streaming 64-bit row hash set across full stream)
8. sample (in quick mode: guarantee representation of both classes with fixed seed)
9. cast features to float32
10. return features + binary labels + metadata + removal report

Important Architecture & Design Notes:
--------------------------------------
- Deduplication vs. Sampling Order: Streaming deduplication is applied over the
  entire data stream first using a 64-bit row hash set (`pd.util.hash_pandas_object`).
  Seeded sampling (if requested) is applied only to deduplicated rows, ensuring the
  sampled records do not depend on arbitrary chunk boundary placement.
- Hash-based deduplication is a high-confidence assessment with negligible collision
  probability (~10^-7 for ~2.8M rows), not a mathematically collision-free guarantee.
- Feature and Metadata Isolation: Feature matrices strictly contain flow telemetry
  features cast to `float32`. Metadata (`original_label`, `source_file`) and the binary
  target are kept in isolated structures and are never permitted inside the feature matrix.
- Quick Mode Strategy: In quick mode, the total sample across both classes and all
  files combined strictly never exceeds quick_max_total_rows (default: 10,000).
  When sufficient valid rows exist, both classes are preserved (up to 5,000 attack
  rows and up to 5,000 benign rows for the 10,000 total cap, or taking all available
  minority class rows and filling the remainder with the majority class) using a fixed
  reproducible seed. Quick-mode metrics are smoke-test numbers only, not final evaluation numbers.
- Dropped Columns Transparency: Every dropped column is logged with an explicit justification.
- Position 55 ('Fwd Header Length') is dropped because it was verified 100% identical to
  position 34 across all 2,830,743 rows of the raw dataset.
- Validation: `validate_upload` strictly enforces the feature schema for offline inference.
  If validation fails, it returns no data (`None`) and detailed error diagnostics.
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# Canonical raw column index 55 verified identical to index 34 across all 2,830,743 rows
DROPPED_COLUMN_REASONS = {
    "Fwd Header Length.1": (
        "Duplicate copy of 'Fwd Header Length' at column position 55; empirically verified "
        "to be 100% identical to position 34 across all 2,830,743 rows of raw CICIDS2017 data."
    )
}


def normalize_header_names(raw_columns: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """Normalize column header names by stripping whitespace and identifying duplicates.

    Returns:
        (normalized_columns, duplicate_info_dict)
    """
    normalized = [c.strip() for c in raw_columns]
    counts = {}
    duplicates = {}
    for idx, name in enumerate(normalized):
        if name in counts:
            counts[name].append(idx)
            duplicates[name] = counts[name]
        else:
            counts[name] = [idx]

    return normalized, duplicates


def normalize_labels(label_series: pd.Series) -> pd.Series:
    """Normalize raw attack labels into standardized naming.

    Replaces malformed U+FFFD ('\ufffd') and mojibake variants in Web Attacks:
      'Web Attack \ufffd Brute Force' -> 'Web Attack - Brute Force'
      'Web Attack \ufffd XSS'         -> 'Web Attack - XSS'
      'Web Attack \ufffd Sql Injection'-> 'Web Attack - Sql Injection'
    """
    def _clean_single_label(val: Any) -> str:
        if pd.isna(val):
            return "UNKNOWN"
        s = str(val).strip()
        # Handle replacement character \ufffd and common mojibake/question-mark variants
        s = re.sub(r"Web Attack\s*[\ufffd\?\-\–\—]+\s*", "Web Attack - ", s)
        # Collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    return label_series.apply(_clean_single_label)


def to_binary(labels: pd.Series) -> pd.Series:
    """Map normalized labels to binary integer targets:

    BENIGN -> 0, any attack class -> 1.
    """
    return (labels != "BENIGN").astype(np.int8)


def find_label_column_name(columns: List[str]) -> Optional[str]:
    """Find the label column case-insensitively after whitespace trimming."""
    for col in columns:
        if col.strip().lower() == "label":
            return col
    for col in columns:
        if "label" in col.strip().lower():
            return col
    return None


def clean_chunk(
    chunk: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Perform numeric coercion, inf replacement, and row filtering on a chunk.

    Drops:
      1. Rows with NaN or inf in any feature column.
      2. Rows where 'Flow Duration' is negative (< 0).

    Returns:
        (cleaned_chunk, removal_report)
    """
    input_rows = len(chunk)
    if input_rows == 0:
        return chunk, {"input_rows": 0, "removed_nan_inf": 0, "removed_negative_flow_duration": 0, "output_rows": 0}

    # Extract features and label
    features = chunk[feature_cols].copy()

    # Coerce features to numeric float64, converting unparseable strings to NaN
    for col in feature_cols:
        if not np.issubdtype(features[col].dtype, np.number):
            features[col] = pd.to_numeric(features[col], errors="coerce")

    # Replace inf and -inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # Mask for NaN/Inf across all feature columns
    nan_inf_mask = features.isna().any(axis=1)
    removed_nan_inf = int(nan_inf_mask.sum())

    # Retain rows free of NaN/Inf for the next check
    valid_mask = ~nan_inf_mask

    # Find Flow Duration column to check for negative values
    flow_dur_cols = [c for c in feature_cols if c.strip().lower() == "flow duration"]
    removed_neg_duration = 0
    if flow_dur_cols:
        dur_col = flow_dur_cols[0]
        # Only check rows that are currently valid
        neg_dur_mask = valid_mask & (features[dur_col] < 0)
        removed_neg_duration = int(neg_dur_mask.sum())
        valid_mask = valid_mask & ~neg_dur_mask

    cleaned_chunk = chunk[valid_mask].copy()
    output_rows = len(cleaned_chunk)

    removal_report = {
        "input_rows": input_rows,
        "removed_nan_inf": removed_nan_inf,
        "removed_negative_flow_duration": removed_neg_duration,
        "output_rows": output_rows,
    }
    return cleaned_chunk, removal_report


def stream_clean_dataset(
    data_dir: Path,
    chunk_size: int = 100000,
    quick: bool = False,
    quick_max_total_rows: int = 10000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, Any]]:
    """Stream, clean, deduplicate, and prepare features, labels, and metadata from raw CSVs.

    Order:
      1. Ingest in chunks from data_dir.
      2. Normalize headers, resolve duplicate 'Fwd Header Length' at position 55.
      3. Normalize labels (Web Attack fixes).
      4. Coerce numeric, drop NaN/Inf and negative Flow Duration.
      5. Full stream 64-bit row hash deduplication (BEFORE any sampling or split).
      6. If quick=True: sample from deduplicated stream ensuring both classes have representation
         and total rows strictly do not exceed quick_max_total_rows (default: 10,000).
      7. Cast feature matrix to float32.

    Returns:
        (features_df, binary_labels, metadata_df, data_prep_report)
    """
    csv_files = sorted(list(data_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in data directory: {data_dir.resolve()}")

    # Determine canonical columns and verify header from the first file
    with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    raw_header = next(csv.reader([first_line]))

    # Identify duplicate columns
    # In CICIDS2017: Index 34 is ' Fwd Header Length', Index 55 is ' Fwd Header Length'
    # Drop index 55 as empirically verified identical
    feature_indices: List[int] = []
    feature_names: List[str] = []
    dropped_columns: Dict[str, str] = {}
    label_raw_name: Optional[str] = None
    label_col_index: Optional[int] = None

    for idx, raw_col in enumerate(raw_header):
        stripped = raw_col.strip()
        if stripped.lower() == "label":
            label_raw_name = raw_col
            label_col_index = idx
            continue

        if idx == 55 and stripped == "Fwd Header Length":
            dropped_columns[f"{stripped} (position {idx})"] = DROPPED_COLUMN_REASONS["Fwd Header Length.1"]
            continue

        feature_indices.append(idx)
        feature_names.append(stripped)

    # Data accumulators
    clean_features_list: List[pd.DataFrame] = []
    clean_labels_list: List[pd.Series] = []
    clean_metadata_list: List[pd.DataFrame] = []

    seen_hashes: Set[int] = set()
    total_input_rows = 0
    total_nan_inf_dropped = 0
    total_neg_duration_dropped = 0
    total_duplicate_rows_dropped = 0

    rng = np.random.RandomState(seed)
    target_attacks_per_file = max(
        quick_max_total_rows // 2,
        int(np.ceil(quick_max_total_rows / len(csv_files))),
    )
    target_benign_per_file = max(
        quick_max_total_rows // 2,
        int(np.ceil((quick_max_total_rows * 1.5) / len(csv_files))),
    )

    for csv_file in csv_files:
        file_name = csv_file.name
        file_attacks_kept = 0
        file_benign_kept = 0

        chunk_iter = pd.read_csv(
            csv_file,
            chunksize=chunk_size,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
            on_bad_lines="warn",
        )

        for chunk in chunk_iter:
            total_input_rows += len(chunk)

            # Map pandas chunk columns to required features and label
            # Pandas renames duplicate position 55 to '.1'
            col_map = {chunk.columns[idx]: feature_names[i] for i, idx in enumerate(feature_indices)}
            pandas_label_col = chunk.columns[label_col_index] if label_col_index is not None else chunk.columns[-1]

            cols_to_use = list(col_map.keys()) + [pandas_label_col]
            chunk_subset = chunk[cols_to_use].copy()
            chunk_subset.rename(columns=col_map, inplace=True)

            # Normalize label strings
            raw_labels = chunk_subset[pandas_label_col]
            norm_labels = normalize_labels(raw_labels)
            chunk_subset["__NORMALIZED_LABEL__"] = norm_labels
            chunk_subset.drop(columns=[pandas_label_col], inplace=True)

            # Clean numeric values, drop NaN/Inf and negative Flow Duration
            cleaned_subchunk, report = clean_chunk(
                chunk_subset,
                feature_cols=feature_names,
                label_col="__NORMALIZED_LABEL__",
            )
            total_nan_inf_dropped += report["removed_nan_inf"]
            total_neg_duration_dropped += report["removed_negative_flow_duration"]

            if len(cleaned_subchunk) == 0:
                continue

            # Streaming 64-bit row hash deduplication on cleaned features + label
            features_part = cleaned_subchunk[feature_names]
            row_hashes = pd.util.hash_pandas_object(features_part, index=False)

            is_dup_mask = np.zeros(len(cleaned_subchunk), dtype=bool)
            for i, h in enumerate(row_hashes):
                h_val = int(h)
                if h_val in seen_hashes:
                    is_dup_mask[i] = True
                else:
                    seen_hashes.add(h_val)

            dups_in_chunk = int(is_dup_mask.sum())
            total_duplicate_rows_dropped += dups_in_chunk

            deduped_chunk = cleaned_subchunk[~is_dup_mask]
            if len(deduped_chunk) == 0:
                continue

            # In quick mode: collect separate attack and benign quotas per file to prevent
            # early benign chunks from crowding out attack rows located later in the file/dataset.
            if quick:
                is_attack = deduped_chunk["__NORMALIZED_LABEL__"] != "BENIGN"
                attacks = deduped_chunk[is_attack]
                benign = deduped_chunk[~is_attack]

                parts_to_keep = []
                if file_attacks_kept < target_attacks_per_file and len(attacks) > 0:
                    take_att = min(len(attacks), target_attacks_per_file - file_attacks_kept)
                    parts_to_keep.append(attacks.iloc[:take_att])
                    file_attacks_kept += take_att

                if file_benign_kept < target_benign_per_file and len(benign) > 0:
                    take_ben = min(len(benign), target_benign_per_file - file_benign_kept)
                    parts_to_keep.append(benign.iloc[:take_ben])
                    file_benign_kept += take_ben

                if parts_to_keep:
                    deduped_chunk = pd.concat(parts_to_keep, axis=0)
                else:
                    # If both quotas for this file are satisfied, we can stop reading this file early
                    if file_attacks_kept >= target_attacks_per_file and file_benign_kept >= target_benign_per_file:
                        break
                    continue

            # Separate features, labels, and metadata
            feat_df = deduped_chunk[feature_names].astype(np.float32)
            lbl_series = deduped_chunk["__NORMALIZED_LABEL__"]
            meta_df = pd.DataFrame({
                "original_label": lbl_series.values,
                "source_file": file_name,
            }, index=deduped_chunk.index)

            clean_features_list.append(feat_df)
            clean_labels_list.append(lbl_series)
            clean_metadata_list.append(meta_df)

    if not clean_features_list:
        raise ValueError("Data preparation produced an empty dataset. All rows were filtered or missing.")

    all_features = pd.concat(clean_features_list, axis=0, ignore_index=True)
    all_raw_labels = pd.concat(clean_labels_list, axis=0, ignore_index=True)
    all_metadata = pd.concat(clean_metadata_list, axis=0, ignore_index=True)

    # In quick mode: strictly enforce maximum quick_max_total_rows across all files combined
    if quick and len(all_raw_labels) > quick_max_total_rows:
        attack_idx = np.where(all_raw_labels != "BENIGN")[0]
        benign_idx = np.where(all_raw_labels == "BENIGN")[0]

        if len(attack_idx) > 0 and len(benign_idx) > 0:
            target_attacks = min(len(attack_idx), quick_max_total_rows // 2)
            target_benign = min(len(benign_idx), quick_max_total_rows - target_attacks)
            if target_benign < (quick_max_total_rows - target_attacks):
                target_attacks = min(len(attack_idx), quick_max_total_rows - target_benign)

            rng.shuffle(attack_idx)
            rng.shuffle(benign_idx)
            selected_attack_idx = attack_idx[:target_attacks]
            selected_benign_idx = benign_idx[:target_benign]
            selected_idx = np.sort(np.concatenate([selected_attack_idx, selected_benign_idx]))
        else:
            all_idx = np.arange(len(all_raw_labels))
            rng.shuffle(all_idx)
            selected_idx = np.sort(all_idx[:quick_max_total_rows])

        all_features = all_features.iloc[selected_idx].reset_index(drop=True)
        all_raw_labels = all_raw_labels.iloc[selected_idx].reset_index(drop=True)
        all_metadata = all_metadata.iloc[selected_idx].reset_index(drop=True)

    binary_labels = to_binary(all_raw_labels)

    removal_report = {
        "total_input_rows": total_input_rows,
        "removed_nan_inf": total_nan_inf_dropped,
        "removed_negative_flow_duration": total_neg_duration_dropped,
        "removed_duplicates_hash": total_duplicate_rows_dropped,
        "final_output_rows": len(all_features),
        "feature_count": len(feature_names),
        "ordered_features": feature_names,
        "dropped_columns": dropped_columns,
        "dedup_methodology": (
            "64-bit streaming row hash assessment (pandas hash_pandas_object) applied over the full stream "
            "prior to train/test splitting or subsampling. Identifies duplicate rows across streaming chunks "
            "without caching complete DataFrames in memory."
        ),
        "quick_mode": quick,
    }

    return all_features, binary_labels, all_metadata, removal_report


def validate_upload(
    df: pd.DataFrame,
    feature_schema: Dict[str, Any],
) -> Tuple[bool, Optional[pd.DataFrame], List[str]]:
    """Validate incoming CSV flow DataFrame for offline Model 1 inference.

    Rules:
      1. Strips whitespace from column headers.
      2. Rejects if duplicate headers exist after whitespace stripping.
      3. Rejects if any required feature in feature_schema['ordered_features'] is missing.
      4. Extra columns: ignored and reported as informational warnings.
      5. Column order: reordered to match exact feature_schema['ordered_features'].
      6. Numeric validation: rejects if any cell contains NaN, Inf, -Inf, or non-numeric values.
         Reports invalid row counts and example indices.
      7. Returns no data (None) whenever validation fails.

    Returns:
        (is_valid, cleaned_float32_df_or_None, list_of_errors_or_warnings)
    """
    messages: List[str] = []

    if df is None or len(df) == 0:
        return False, None, ["Input DataFrame is empty or null."]

    # 1. Normalize headers
    raw_cols = list(df.columns)
    norm_cols = [str(c).strip() for c in raw_cols]

    # 2. Check for duplicate headers
    seen_cols: Set[str] = set()
    dup_cols: List[str] = []
    for c in norm_cols:
        if c in seen_cols:
            dup_cols.append(c)
        seen_cols.add(c)

    if dup_cols:
        return False, None, [f"Validation failed: Duplicate normalized column names detected: {sorted(list(set(dup_cols)))}"]

    # Map normalized names to original column names
    col_mapping = {norm: orig for norm, orig in zip(norm_cols, raw_cols)}

    # 3. Check for required features
    required_features: List[str] = feature_schema.get("ordered_features", [])
    if not required_features:
        return False, None, ["Feature schema contains no 'ordered_features'."]

    missing_features = [req for req in required_features if req not in seen_cols]
    if missing_features:
        return False, None, [f"Validation failed: Missing {len(missing_features)} required feature columns: {missing_features[:10]}"]

    # 4. Check for extra columns (policy: ignore and document)
    extra_cols = [c for c in seen_cols if c not in set(required_features)]
    if extra_cols:
        messages.append(f"Notice: {len(extra_cols)} extra columns ignored: {extra_cols[:5]}")

    # 5. Extract features in exact schema order
    ordered_original_cols = [col_mapping[req] for req in required_features]
    features_df = df[ordered_original_cols].copy()
    features_df.columns = required_features

    # 6. Coerce to numeric float64 to detect non-numeric strings, NaN, Inf
    invalid_rows_mask = np.zeros(len(features_df), dtype=bool)

    for col in required_features:
        series = features_df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        # Check non-numeric strings (original not na, but numeric is na)
        non_numeric_mask = series.notna() & numeric_series.isna()
        # Check NaN or Inf
        na_or_inf_mask = numeric_series.isna() | np.isinf(numeric_series)

        col_invalid_mask = non_numeric_mask | na_or_inf_mask
        invalid_rows_mask |= col_invalid_mask

    invalid_row_count = int(invalid_rows_mask.sum())
    if invalid_row_count > 0:
        example_indices = list(np.where(invalid_rows_mask)[0][:5])
        return False, None, [
            f"Validation failed: {invalid_row_count} rows contain non-numeric, NaN, or Infinite values. "
            f"Example invalid row indices: {example_indices}"
        ]

    # Convert valid features to float32
    cleaned_df = features_df.astype(np.float32)
    return True, cleaned_df, messages
