"""
Base Collector — NetSentinel AI Phase 1
========================================
Defines the abstract interface that all traffic collectors must implement.

This keeps the TrafficProcessor completely independent of the data source.
New collectors (live capture, PCAP files, etc.) just need to implement
this interface and the rest of the pipeline works unchanged.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseCollector(ABC):
    """
    Abstract base class for all traffic data collectors.

    Subclasses:
      - CsvCollector   : loads from a CSV file (Phase 1, always available)
      - LiveCollector  : captures from a live interface (optional, Phase 1 extension)
    """

    @abstractmethod
    def collect(self, **kwargs) -> pd.DataFrame:
        """
        Collect raw traffic records and return them as a raw DataFrame.
        Column names may be non-canonical — the TrafficProcessor will normalise them.

        Returns
        -------
        pd.DataFrame
            Raw traffic data. May be empty if no data was available.
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable name of this collector (used in logs and API responses)."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Returns True if this collector can currently operate.
        (e.g. file exists, network interface is present, library is installed)
        """
        ...
