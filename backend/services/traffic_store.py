"""
Traffic Store — NetSentinel AI Phase 1
=======================================
In-memory storage for normalised traffic data in Phase 1.

Architecture:
  - Holds one active DataFrame in memory (the "loaded dataset")
  - Provides a clean interface for loading CSV data
  - Gracefully handles the Supabase-unavailable case
  - Does NOT depend on Flask or Supabase being available
  - Supabase integration is fully optional and fails gracefully

The store is a singleton module-level object to keep state
simple without introducing complex dependency injection.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.services.traffic_processor import TrafficProcessor
from backend.schemas.traffic_schema import ALL_COLUMNS, SOURCE_CSV

logger = logging.getLogger(__name__)

# Singleton processor instance
_processor = TrafficProcessor()

# In-memory DataFrame holding the currently loaded traffic dataset
_traffic_df: pd.DataFrame = pd.DataFrame(columns=ALL_COLUMNS)


# ----------------------------------------------------------------
# Load from CSV
# ----------------------------------------------------------------

def load_from_csv(file_path: str | Path) -> tuple[bool, str, int]:
    """
    Load and normalise a CSV file into the in-memory store.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    (success: bool, message: str, records_loaded: int)
    """
    global _traffic_df
    path = Path(file_path)

    if not path.exists():
        msg = f"CSV file not found: {path}"
        logger.error(msg)
        return False, msg, 0

    if path.stat().st_size == 0:
        msg = f"CSV file is empty: {path}"
        logger.warning(msg)
        return False, msg, 0

    try:
        raw_df = pd.read_csv(path)
    except Exception as exc:
        msg = f"Failed to read CSV '{path}': {exc}"
        logger.error(msg)
        return False, msg, 0

    if raw_df.empty:
        msg = "CSV loaded but contains no data rows."
        logger.warning(msg)
        _traffic_df = pd.DataFrame(columns=ALL_COLUMNS)
        return False, msg, 0

    normalised = _processor.process(raw_df, source_type=SOURCE_CSV)
    _traffic_df = normalised

    count = len(normalised)
    msg = f"Loaded {count} records from '{path.name}' (SIMULATED/SAMPLE DATA)."
    logger.info(msg)
    return True, msg, count


# ----------------------------------------------------------------
# Access the in-memory dataset
# ----------------------------------------------------------------

def get_dataframe() -> pd.DataFrame:
    """Return the currently loaded normalised traffic DataFrame."""
    return _traffic_df


def is_loaded() -> bool:
    """Return True if the store contains at least one record."""
    return not _traffic_df.empty


def get_record_count() -> int:
    """Return the number of records currently in the store."""
    return len(_traffic_df)


def clear() -> None:
    """Clear all in-memory traffic data."""
    global _traffic_df
    _traffic_df = pd.DataFrame(columns=ALL_COLUMNS)
    logger.info("Traffic store cleared.")


# ----------------------------------------------------------------
# Supabase adapter (optional / graceful fallback)
# ----------------------------------------------------------------

def push_to_supabase(records: list[dict]) -> tuple[bool, str]:
    """
    Attempt to push records to Supabase.
    Returns (False, reason) if credentials are missing or Supabase is unavailable.
    The main application continues working regardless of the return value.

    This function is intentionally a no-op in Phase 1 until real
    Supabase credentials are configured.
    """
    from utils.supabase_client import get_supabase_config, get_supabase_client

    config = get_supabase_config()
    if not config["ready"]:
        msg = "Supabase credentials not configured — skipping push. Local data is unaffected."
        logger.info(msg)
        return False, msg

    client = get_supabase_client()
    if client is None:
        msg = "Supabase client could not be initialised — skipping push."
        logger.warning(msg)
        return False, msg

    try:
        response = client.table("network_traffic").insert(records).execute()
        count = len(response.data) if response.data else 0
        msg = f"Pushed {count} records to Supabase."
        logger.info(msg)
        return True, msg
    except Exception as exc:
        msg = f"Supabase push failed: {exc}"
        logger.warning(msg)
        return False, msg
