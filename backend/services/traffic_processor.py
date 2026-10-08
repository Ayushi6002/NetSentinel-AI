"""
Traffic Processor — NetSentinel AI Phase 1
===========================================
Receives raw traffic records (from CSV or future live collectors)
and produces a clean, normalised pandas DataFrame using the
canonical column schema defined in backend/schemas/traffic_schema.py.

Design principles:
  - No Flask dependency (can be used standalone / in tests / CLI)
  - No Supabase dependency
  - Validates and sanitises each record defensively
  - Logs warnings for bad records; does NOT crash on bad input
  - Returns an empty DataFrame (with correct columns) on total failure
"""

import ipaddress
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from backend.schemas.traffic_schema import (
    ALL_COLUMNS,
    REQUIRED_COLUMNS,
    KNOWN_PROTOCOLS,
    PROTOCOLS_WITHOUT_PORTS,
    SOURCE_CSV,
    COLUMN_TIMESTAMP,
    COLUMN_SOURCE_IP,
    COLUMN_DESTINATION_IP,
    COLUMN_SOURCE_PORT,
    COLUMN_DESTINATION_PORT,
    COLUMN_PROTOCOL,
    COLUMN_PACKET_COUNT,
    COLUMN_BYTE_COUNT,
    COLUMN_DURATION,
    COLUMN_TRAFFIC_RATE,
    COLUMN_SOURCE_TYPE,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Column aliases: maps alternative CSV column names → canonical names
# Allows the processor to accept slightly different CSV headers.
# ----------------------------------------------------------------
COLUMN_ALIASES: dict[str, str] = {
    "src_ip":           COLUMN_SOURCE_IP,
    "dst_ip":           COLUMN_DESTINATION_IP,
    "src_port":         COLUMN_SOURCE_PORT,
    "dst_port":         COLUMN_DESTINATION_PORT,
    "proto":            COLUMN_PROTOCOL,
    "packets":          COLUMN_PACKET_COUNT,
    "bytes":            COLUMN_BYTE_COUNT,
    "dur":              COLUMN_DURATION,
    "rate":             COLUMN_TRAFFIC_RATE,
    "time":             COLUMN_TIMESTAMP,
}


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column set."""
    return pd.DataFrame(columns=ALL_COLUMNS)


def _normalise_protocol(value) -> str:
    """
    Upper-case and clean a protocol name.
    Maps any unrecognised protocol to 'OTHER'.
    """
    if pd.isna(value) or str(value).strip() == "":
        return "OTHER"
    proto = str(value).strip().upper()
    return proto if proto in KNOWN_PROTOCOLS else "OTHER"


def _safe_port(value, protocol: Optional[str] = None) -> Optional[int]:
    """
    Convert a port value to int in [0, 65535], or None if invalid.
    Protocols without transport ports (especially ICMP, ARP) must always have None.
    """
    if protocol and str(protocol).strip().upper() in PROTOCOLS_WITHOUT_PORTS:
        return None
    if value is None or isinstance(value, bool) or pd.isna(value):
        return None
    try:
        port = int(float(value))
        if 0 <= port <= 65535:
            return port
        return None
    except (ValueError, TypeError):
        return None


def _safe_positive_int(value, column_name: str) -> Optional[int]:
    """Convert to a positive integer, or None on failure."""
    if pd.isna(value):
        return None
    try:
        result = int(float(value))
        if result < 0:
            logger.debug("Negative value %s ignored for column '%s'", value, column_name)
            return None
        return result
    except (ValueError, TypeError):
        logger.debug("Cannot parse '%s' as int for column '%s'", value, column_name)
        return None


def _safe_float(value, column_name: str) -> Optional[float]:
    """Convert to a non-negative float, or None on failure."""
    if pd.isna(value):
        return None
    try:
        result = float(value)
        if result < 0:
            return None
        return result
    except (ValueError, TypeError):
        logger.debug("Cannot parse '%s' as float for column '%s'", value, column_name)
        return None


def _safe_timestamp(value) -> Optional[pd.Timestamp]:
    """Parse a timestamp value. Returns None if unparseable."""
    if pd.isna(value):
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        logger.debug("Cannot parse timestamp: %s", value)
        return None


def _safe_ip(value) -> Optional[str]:
    """
    Validate an IPv4/IPv6 address using Python's standard library ipaddress module.
    Returns normalized canonical IP string if valid, or None for invalid/malformed.
    Ensures invalid IP addresses are rejected before reaching storage/database.
    """
    if value is None or isinstance(value, bool) or pd.isna(value):
        return None
    ip_str = str(value).strip()
    if not ip_str:
        return None
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return str(ip_obj)
    except (ValueError, TypeError):
        return None


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any aliased columns to their canonical names."""
    rename_map = {
        alias: canonical
        for alias, canonical in COLUMN_ALIASES.items()
        if alias in df.columns and canonical not in df.columns
    }
    if rename_map:
        logger.debug("Renaming CSV columns via alias map: %s", rename_map)
        df = df.rename(columns=rename_map)
    return df


def _add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any canonical columns that are absent, filled with None."""
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def _compute_traffic_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate traffic_rate = byte_count / duration (bytes/sec).
    If duration is 0 or NaN, traffic_rate remains NaN.
    Only overwrites traffic_rate if it is NaN (i.e. not already supplied).
    """
    mask_missing = df[COLUMN_TRAFFIC_RATE].isna()
    mask_valid_dur = df[COLUMN_DURATION].notna() & (df[COLUMN_DURATION] > 0)
    mask = mask_missing & mask_valid_dur

    df.loc[mask, COLUMN_TRAFFIC_RATE] = (
        df.loc[mask, COLUMN_BYTE_COUNT] / df.loc[mask, COLUMN_DURATION]
    )
    return df


def normalise_dataframe(df: pd.DataFrame, source_type: str = SOURCE_CSV) -> pd.DataFrame:
    """
    Core normalisation function.
    Accepts a raw DataFrame (any column names), applies aliases,
    validates and cleans every field, and returns a canonical DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.
    source_type : str
        'csv' or 'live' — recorded on every row for traceability.

    Returns
    -------
    pd.DataFrame with ALL_COLUMNS columns. May be empty if all rows fail.
    """
    if df is None or df.empty:
        logger.warning("normalise_dataframe: received empty input.")
        return _empty_df()

    # Strip whitespace from all column headers
    df.columns = [c.strip().lower() for c in df.columns]

    # Apply column aliases (e.g. src_ip → source_ip)
    df = _apply_aliases(df)

    # Add any missing canonical columns as None
    df = _add_missing_columns(df)

    # ---- Field-by-field normalisation ----

    df[COLUMN_TIMESTAMP] = df[COLUMN_TIMESTAMP].apply(_safe_timestamp)
    df[COLUMN_SOURCE_IP] = df[COLUMN_SOURCE_IP].apply(_safe_ip)
    df[COLUMN_DESTINATION_IP] = df[COLUMN_DESTINATION_IP].apply(_safe_ip)
    df[COLUMN_PROTOCOL] = df[COLUMN_PROTOCOL].apply(_normalise_protocol)

    # Protocols that do not use transport ports (especially ICMP, ARP) must have NULL (None) ports.
    # For TCP/UDP, validate ports in [0, 65535].
    df[COLUMN_SOURCE_PORT] = pd.Series(
        [
            None if proto in PROTOCOLS_WITHOUT_PORTS else _safe_port(p)
            for p, proto in zip(df[COLUMN_SOURCE_PORT], df[COLUMN_PROTOCOL])
        ],
        index=df.index,
        dtype=object,
    )
    df[COLUMN_DESTINATION_PORT] = pd.Series(
        [
            None if proto in PROTOCOLS_WITHOUT_PORTS else _safe_port(p)
            for p, proto in zip(df[COLUMN_DESTINATION_PORT], df[COLUMN_PROTOCOL])
        ],
        index=df.index,
        dtype=object,
    )

    df[COLUMN_PACKET_COUNT] = df[COLUMN_PACKET_COUNT].apply(
        lambda v: _safe_positive_int(v, COLUMN_PACKET_COUNT)
    )
    df[COLUMN_BYTE_COUNT] = df[COLUMN_BYTE_COUNT].apply(
        lambda v: _safe_positive_int(v, COLUMN_BYTE_COUNT)
    )
    df[COLUMN_DURATION] = df[COLUMN_DURATION].apply(
        lambda v: _safe_float(v, COLUMN_DURATION)
    )
    df[COLUMN_TRAFFIC_RATE] = df[COLUMN_TRAFFIC_RATE].apply(
        lambda v: _safe_float(v, COLUMN_TRAFFIC_RATE)
    )

    # Stamp source type
    df[COLUMN_SOURCE_TYPE] = source_type

    # Compute traffic_rate where it is missing
    df = _compute_traffic_rate(df)

    # ---- Drop records missing required fields ----
    before = len(df)
    for col in REQUIRED_COLUMNS:
        df = df[df[col].notna()]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            "normalise_dataframe: dropped %d/%d records missing required fields.",
            dropped, before,
        )

    # Return only canonical columns, in canonical order
    return df[ALL_COLUMNS].reset_index(drop=True)


class TrafficProcessor:
    """
    Stateless helper that wraps normalise_dataframe for easy use
    by collectors and API routes.
    """

    def process(self, df: pd.DataFrame, source_type: str = SOURCE_CSV) -> pd.DataFrame:
        """Normalise a raw DataFrame and return a canonical one."""
        return normalise_dataframe(df, source_type=source_type)

    def process_records(self, records: list[dict], source_type: str = SOURCE_CSV) -> pd.DataFrame:
        """Normalise a list of raw record dicts."""
        if not records:
            return _empty_df()
        raw_df = pd.DataFrame(records)
        return self.process(raw_df, source_type=source_type)
