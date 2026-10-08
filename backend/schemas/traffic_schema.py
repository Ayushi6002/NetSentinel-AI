"""
Traffic Schema — NetSentinel AI Phase 1
========================================
Defines the canonical field names and types for a normalized
network traffic record. This module is the single source of truth
for what a "traffic record" looks like throughout the application.

This module has NO external dependencies (Flask, Supabase, pandas, etc.).
It is safe to import from anywhere in the project.
"""

from typing import Optional

# ----------------------------------------------------------------
# Canonical column names used throughout the Phase 1 pipeline.
# All processors and analyzers must output records using these names.
# ----------------------------------------------------------------

COLUMN_TIMESTAMP         = "timestamp"
COLUMN_SOURCE_IP         = "source_ip"
COLUMN_DESTINATION_IP    = "destination_ip"
COLUMN_SOURCE_PORT       = "source_port"
COLUMN_DESTINATION_PORT  = "destination_port"
COLUMN_PROTOCOL          = "protocol"
COLUMN_PACKET_COUNT      = "packet_count"
COLUMN_BYTE_COUNT        = "byte_count"
COLUMN_DURATION          = "duration"
COLUMN_TRAFFIC_RATE      = "traffic_rate"
COLUMN_SOURCE_TYPE       = "source_type"   # 'csv' or 'live'

# Ordered list of all expected columns (matches Supabase schema)
ALL_COLUMNS = [
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
]

# Columns that must not be missing/null in a valid record
REQUIRED_COLUMNS = [
    COLUMN_TIMESTAMP,
    COLUMN_SOURCE_IP,
    COLUMN_DESTINATION_IP,
    COLUMN_PROTOCOL,
]

# Known protocol names (used for normalisation)
KNOWN_PROTOCOLS = {"TCP", "UDP", "ICMP", "DNS", "HTTP", "HTTPS", "ARP", "OTHER"}

# Protocols that do not use transport-layer (TCP/UDP) ports.
# For these protocols, source_port and destination_port must be normalized to NULL (None).
PROTOCOLS_WITHOUT_PORTS = {"ICMP", "ARP"}

# Source type constants
SOURCE_CSV  = "csv"
SOURCE_LIVE = "live"


def is_port_protocol(protocol: Optional[str]) -> bool:
    """Return True if protocol uses transport ports (e.g. TCP, UDP)."""
    if not protocol:
        return True
    return protocol.strip().upper() not in PROTOCOLS_WITHOUT_PORTS


def make_empty_record() -> dict:
    """Return an empty traffic record dict with all fields set to None."""
    return {col: None for col in ALL_COLUMNS}
