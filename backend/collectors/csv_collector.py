"""
CSV Collector — NetSentinel AI Phase 1
========================================
Implements the BaseCollector interface for CSV-based traffic data.

This is the PRIMARY data source for Phase 1 and always functions
without live packet capture, Supabase credentials, or internet access.

Usage:
    collector = CsvCollector("data/sample_network_traffic.csv")
    if collector.is_available:
        raw_df = collector.collect()
        # pass raw_df to TrafficProcessor
"""

import logging
from pathlib import Path

import pandas as pd

from backend.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)


class CsvCollector(BaseCollector):
    """Reads raw network traffic records from a CSV file."""

    def __init__(self, file_path: str | Path):
        self._path = Path(file_path)

    @property
    def source_name(self) -> str:
        return f"CSV:{self._path.name}"

    @property
    def is_available(self) -> bool:
        return self._path.exists() and self._path.stat().st_size > 0

    def collect(self, **kwargs) -> pd.DataFrame:
        """
        Read the CSV file and return a raw DataFrame.
        Column normalisation is done by TrafficProcessor, not here.
        Returns an empty DataFrame on any failure.
        """
        if not self.is_available:
            logger.error("CsvCollector: file not available at '%s'", self._path)
            return pd.DataFrame()

        try:
            df = pd.read_csv(self._path)
            logger.info("CsvCollector: loaded %d rows from '%s'", len(df), self._path.name)
            return df
        except Exception as exc:
            logger.error("CsvCollector: failed to read '%s': %s", self._path, exc)
            return pd.DataFrame()
