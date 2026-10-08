"""
Live Collector (Stub) — NetSentinel AI Phase 1
================================================
Extension point for live network packet capture.

STATUS: NOT IMPLEMENTED in Phase 1.

Reason: Live packet capture on Windows requires either:
  - Npcap (WinPcap successor) driver installed at the OS level
  - Scapy (with Npcap) or PyShark (with Wireshark/tshark) installed
  - Administrator / elevated privileges
  - The correct network interface name

These are environment-specific prerequisites that cannot be guaranteed
on all development machines. Installing them blindly risks:
  1. Breaking environments that lack admin rights
  2. Introducing large driver dependencies for a feature that is optional

Decision: The CSV/sample-data path in CsvCollector is fully sufficient
to demonstrate Phase 1. Live capture will be added as a separate
controlled step once Phases 1 and 2 are stable.

This module exists only to document the extension point and to ensure
the application does not crash when live capture is requested but unavailable.
"""

import logging

import pandas as pd

from backend.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)

# Flag — set to True only if Npcap + Scapy are confirmed to be installed
LIVE_CAPTURE_AVAILABLE = False


class LiveCollector(BaseCollector):
    """
    Stub live traffic collector.
    Always returns 'unavailable' in Phase 1.
    Replace the body of collect() in a future phase.
    """

    def __init__(self, interface: str = ""):
        self._interface = interface

    @property
    def source_name(self) -> str:
        return f"Live:{self._interface or 'default'}"

    @property
    def is_available(self) -> bool:
        """Live capture is disabled in Phase 1."""
        return LIVE_CAPTURE_AVAILABLE

    def collect(self, duration_seconds: int = 10, **kwargs) -> pd.DataFrame:
        """
        Placeholder. Returns an empty DataFrame with a log warning.
        Future implementation will use Scapy or similar library here.
        """
        logger.warning(
            "LiveCollector.collect() called, but live capture is not enabled in Phase 1. "
            "Returning empty DataFrame. Use CSV/sample-data mode instead."
        )
        return pd.DataFrame()
