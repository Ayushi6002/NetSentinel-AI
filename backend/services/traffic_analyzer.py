"""
Traffic Analyzer — NetSentinel AI Phase 1
==========================================
Operates on a normalised traffic DataFrame (output of TrafficProcessor)
and computes the statistics required for the Phase 1 dashboard.

Design principles:
  - Accepts a pandas DataFrame — completely independent of Flask/Supabase
  - All functions are pure: given the same DataFrame, return same result
  - Returns plain Python structures (dicts / lists) that JSON-serialise cleanly
  - Handles empty DataFrames gracefully (returns zeroed/empty results)
  - Can be used in tests, CLI scripts, or from Flask routes
"""

import logging
from typing import Any

import pandas as pd

from backend.schemas.traffic_schema import (
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
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------

def _safe_int(value) -> int:
    """Convert a numpy/pandas scalar to a plain Python int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value, decimals: int = 2) -> float:
    """Convert a numpy/pandas scalar to a rounded Python float safely."""
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return 0.0


def _is_empty(df: pd.DataFrame) -> bool:
    return df is None or df.empty


# ----------------------------------------------------------------
# 1. Overview / summary statistics
# ----------------------------------------------------------------

def compute_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate high-level traffic overview metrics.

    Returns a dict containing:
      total_records, total_packets, total_bytes_mb, total_bytes,
      unique_source_ips, unique_destination_ips, unique_connections,
      avg_traffic_rate_kbps, avg_packet_count, avg_duration_sec
    """
    if _is_empty(df):
        return {
            "total_records": 0,
            "total_packets": 0,
            "total_bytes": 0,
            "total_bytes_mb": 0.0,
            "unique_source_ips": 0,
            "unique_destination_ips": 0,
            "unique_connections": 0,
            "avg_traffic_rate_kbps": 0.0,
            "avg_packet_count": 0.0,
            "avg_duration_sec": 0.0,
        }

    total_packets = _safe_int(df[COLUMN_PACKET_COUNT].sum(skipna=True))
    total_bytes   = _safe_int(df[COLUMN_BYTE_COUNT].sum(skipna=True))

    # A "connection" / flow is a unique (src_ip, dst_ip, src_port, dst_port, protocol) tuple
    unique_connections = _safe_int(
        df[[COLUMN_SOURCE_IP, COLUMN_DESTINATION_IP,
            COLUMN_SOURCE_PORT, COLUMN_DESTINATION_PORT,
            COLUMN_PROTOCOL]].drop_duplicates().shape[0]
    )

    avg_rate = _safe_float(df[COLUMN_TRAFFIC_RATE].mean(skipna=True))
    avg_rate_kbps = round(avg_rate / 1024, 2) if avg_rate else 0.0

    return {
        "total_records": len(df),
        "total_packets": total_packets,
        "total_bytes": total_bytes,
        "total_bytes_mb": round(total_bytes / (1024 * 1024), 4),
        "unique_source_ips": _safe_int(df[COLUMN_SOURCE_IP].nunique()),
        "unique_destination_ips": _safe_int(df[COLUMN_DESTINATION_IP].nunique()),
        "unique_connections": unique_connections,
        "avg_traffic_rate_kbps": avg_rate_kbps,
        "avg_packet_count": _safe_float(df[COLUMN_PACKET_COUNT].mean(skipna=True)),
        "avg_duration_sec": _safe_float(df[COLUMN_DURATION].mean(skipna=True)),
    }


# ----------------------------------------------------------------
# 2. Protocol distribution
# ----------------------------------------------------------------

def compute_protocol_distribution(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Compute per-protocol statistics.

    Returns a list of dicts, sorted by byte_count descending:
      [{"protocol": "TCP", "count": 30, "packets": 500,
        "bytes": 800000, "percentage": 60.0}, ...]
    """
    if _is_empty(df):
        return []

    groups = (
        df.groupby(COLUMN_PROTOCOL, observed=True)
        .agg(
            count=(COLUMN_PROTOCOL, "count"),
            packets=(COLUMN_PACKET_COUNT, "sum"),
            bytes=(COLUMN_BYTE_COUNT, "sum"),
        )
        .reset_index()
    )

    total_records = len(df)
    result = []
    for _, row in groups.iterrows():
        result.append({
            "protocol": str(row[COLUMN_PROTOCOL]),
            "count": _safe_int(row["count"]),
            "packets": _safe_int(row["packets"]),
            "bytes": _safe_int(row["bytes"]),
            "percentage": round(_safe_float(row["count"]) / total_records * 100, 2),
        })

    result.sort(key=lambda x: x["bytes"], reverse=True)
    return result


# ----------------------------------------------------------------
# 3. Top source IPs
# ----------------------------------------------------------------

def compute_top_sources(df: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    """
    Identify the most active source IP addresses by byte count.

    Returns a list of up to n dicts:
      [{"source_ip": "192.168.1.10", "connections": 15,
        "packets": 200, "bytes": 450000}, ...]
    """
    if _is_empty(df):
        return []

    groups = (
        df.groupby(COLUMN_SOURCE_IP, observed=True)
        .agg(
            connections=(COLUMN_SOURCE_IP, "count"),
            packets=(COLUMN_PACKET_COUNT, "sum"),
            bytes=(COLUMN_BYTE_COUNT, "sum"),
        )
        .reset_index()
        .sort_values("bytes", ascending=False)
        .head(n)
    )

    result = []
    for _, row in groups.iterrows():
        result.append({
            "source_ip": str(row[COLUMN_SOURCE_IP]),
            "connections": _safe_int(row["connections"]),
            "packets": _safe_int(row["packets"]),
            "bytes": _safe_int(row["bytes"]),
        })
    return result


# ----------------------------------------------------------------
# 4. Top destination IPs
# ----------------------------------------------------------------

def compute_top_destinations(df: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    """
    Identify the most targeted destination IP addresses by byte count.

    Returns a list of up to n dicts:
      [{"destination_ip": "142.250.80.46", "connections": 5,
        "packets": 180, "bytes": 320000}, ...]
    """
    if _is_empty(df):
        return []

    groups = (
        df.groupby(COLUMN_DESTINATION_IP, observed=True)
        .agg(
            connections=(COLUMN_DESTINATION_IP, "count"),
            packets=(COLUMN_PACKET_COUNT, "sum"),
            bytes=(COLUMN_BYTE_COUNT, "sum"),
        )
        .reset_index()
        .sort_values("bytes", ascending=False)
        .head(n)
    )

    result = []
    for _, row in groups.iterrows():
        result.append({
            "destination_ip": str(row[COLUMN_DESTINATION_IP]),
            "connections": _safe_int(row["connections"]),
            "packets": _safe_int(row["packets"]),
            "bytes": _safe_int(row["bytes"]),
        })
    return result


# ----------------------------------------------------------------
# 5. Traffic timeline
# ----------------------------------------------------------------

def compute_timeline(df: pd.DataFrame, freq: str = "1min") -> list[dict[str, Any]]:
    """
    Aggregate traffic volume over time at the given frequency.

    Parameters
    ----------
    df   : normalised traffic DataFrame
    freq : pandas offset alias, e.g. '1min', '5min', '1h'

    Returns a list of time-bucketed dicts, sorted chronologically:
      [{"bucket": "2024-11-01T08:00:00", "records": 10,
        "packets": 100, "bytes": 200000}, ...]
    """
    if _is_empty(df):
        return []

    ts_col = df[COLUMN_TIMESTAMP]
    if ts_col.isna().all():
        logger.warning("compute_timeline: all timestamps are NaN — returning empty timeline.")
        return []

    df_ts = df[[COLUMN_TIMESTAMP, COLUMN_PACKET_COUNT, COLUMN_BYTE_COUNT]].copy()
    df_ts = df_ts.dropna(subset=[COLUMN_TIMESTAMP])
    df_ts = df_ts.set_index(COLUMN_TIMESTAMP)
    df_ts.index = pd.DatetimeIndex(df_ts.index)

    try:
        grouped = df_ts.resample(freq).agg(
            records=(COLUMN_PACKET_COUNT, "count"),
            packets=(COLUMN_PACKET_COUNT, "sum"),
            bytes=(COLUMN_BYTE_COUNT, "sum"),
        )
    except Exception as exc:
        logger.error("compute_timeline: resample failed: %s", exc)
        return []

    result = []
    for bucket_ts, row in grouped.iterrows():
        result.append({
            "bucket": bucket_ts.isoformat(),
            "records": _safe_int(row["records"]),
            "packets": _safe_int(row["packets"]),
            "bytes": _safe_int(row["bytes"]),
        })
    return result


# ----------------------------------------------------------------
# 6. Connection statistics
# ----------------------------------------------------------------

def compute_connection_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Basic per-connection statistics.

    Returns a dict:
      avg_bytes_per_connection, max_bytes_connection,
      avg_packets_per_connection, max_packets_connection,
      avg_duration_sec, max_duration_sec,
      top_protocol_by_connections
    """
    if _is_empty(df):
        return {
            "avg_bytes_per_connection": 0.0,
            "max_bytes_connection": 0,
            "avg_packets_per_connection": 0.0,
            "max_packets_connection": 0,
            "avg_duration_sec": 0.0,
            "max_duration_sec": 0.0,
            "top_protocol_by_connections": None,
        }

    top_proto = None
    proto_counts = df[COLUMN_PROTOCOL].value_counts()
    if not proto_counts.empty:
        top_proto = str(proto_counts.index[0])

    return {
        "avg_bytes_per_connection": _safe_float(df[COLUMN_BYTE_COUNT].mean(skipna=True)),
        "max_bytes_connection": _safe_int(df[COLUMN_BYTE_COUNT].max(skipna=True)),
        "avg_packets_per_connection": _safe_float(df[COLUMN_PACKET_COUNT].mean(skipna=True)),
        "max_packets_connection": _safe_int(df[COLUMN_PACKET_COUNT].max(skipna=True)),
        "avg_duration_sec": _safe_float(df[COLUMN_DURATION].mean(skipna=True)),
        "max_duration_sec": _safe_float(df[COLUMN_DURATION].max(skipna=True)),
        "top_protocol_by_connections": top_proto,
    }


# ----------------------------------------------------------------
# Convenience: run all analyses in one call
# ----------------------------------------------------------------

def analyse_all(df: pd.DataFrame, timeline_freq: str = "1min") -> dict[str, Any]:
    """
    Run the complete Phase 1 analysis suite and return all results
    in a single dict, suitable for caching or bulk API responses.
    """
    return {
        "overview": compute_overview(df),
        "protocols": compute_protocol_distribution(df),
        "top_sources": compute_top_sources(df),
        "top_destinations": compute_top_destinations(df),
        "timeline": compute_timeline(df, freq=timeline_freq),
        "connection_stats": compute_connection_stats(df),
    }
