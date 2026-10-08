"""
Test: Traffic Processor — NetSentinel AI Phase 1
=================================================
Tests for the traffic normalisation and validation layer.

Coverage:
  - Canonical column output
  - CSV loading with alias column names
  - Timestamp parsing
  - Protocol normalisation
  - Port validation
  - Numeric field validation
  - Missing required fields → row dropped
  - Empty input handling
  - traffic_rate auto-calculation
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import pandas as pd

from backend.services.traffic_processor import normalise_dataframe, TrafficProcessor
from backend.schemas.traffic_schema import (
    ALL_COLUMNS, REQUIRED_COLUMNS,
    COLUMN_TIMESTAMP, COLUMN_SOURCE_IP, COLUMN_DESTINATION_IP,
    COLUMN_PROTOCOL, COLUMN_PACKET_COUNT, COLUMN_BYTE_COUNT,
    COLUMN_DURATION, COLUMN_TRAFFIC_RATE, COLUMN_SOURCE_PORT,
    COLUMN_DESTINATION_PORT, COLUMN_SOURCE_TYPE,
    SOURCE_CSV,
)


def _make_valid_row(**overrides) -> dict:
    """Return a minimal valid raw traffic row dict."""
    row = {
        "timestamp": "2024-11-01 08:00:00",
        "source_ip": "192.168.1.10",
        "destination_ip": "8.8.8.8",
        "source_port": 54321,
        "destination_port": 53,
        "protocol": "UDP",
        "packet_count": 5,
        "byte_count": 400,
        "duration": 0.1,
        "traffic_rate": 4000.0,
    }
    row.update(overrides)
    return row


class TestNormaliseDataframe(unittest.TestCase):

    def test_output_has_all_canonical_columns(self):
        """Output DataFrame must have exactly the canonical column set."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row()]))
        for col in ALL_COLUMNS:
            self.assertIn(col, df.columns, f"Missing canonical column: {col}")

    def test_valid_row_preserved(self):
        """A fully valid row should survive normalisation unchanged."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row()]))
        self.assertEqual(len(df), 1)

    def test_protocol_uppercased(self):
        """Protocol names must be upper-cased."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(protocol="tcp")]))
        self.assertEqual(df.iloc[0][COLUMN_PROTOCOL], "TCP")

    def test_unknown_protocol_becomes_other(self):
        """Unrecognised protocols must be mapped to 'OTHER'."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(protocol="OSPF")]))
        self.assertEqual(df.iloc[0][COLUMN_PROTOCOL], "OTHER")

    def test_invalid_timestamp_row_dropped(self):
        """Row with completely unparseable timestamp should be dropped (required field)."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(timestamp="NOT-A-DATE")]))
        # timestamp becomes NaT, which is NaN, so required-field drop removes it
        self.assertEqual(len(df), 0)

    def test_missing_source_ip_row_dropped(self):
        """Row with missing source_ip must be dropped."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(source_ip=None)]))
        self.assertEqual(len(df), 0)

    def test_missing_destination_ip_row_dropped(self):
        """Row with missing destination_ip must be dropped."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(destination_ip="")]))
        self.assertEqual(len(df), 0)

    def test_missing_protocol_defaults_to_other(self):
        """Missing protocol becomes 'OTHER' and the row is kept."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(protocol=None)]))
        # protocol='OTHER' is not null so the row should survive
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0][COLUMN_PROTOCOL], "OTHER")

    def test_invalid_port_becomes_none(self):
        """Out-of-range port number should become None."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(source_port=99999)]))
        self.assertIsNone(df.iloc[0][COLUMN_SOURCE_PORT])

    def test_negative_packet_count_becomes_none(self):
        """Negative packet_count should become None."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row(packet_count=-5)]))
        self.assertIsNone(df.iloc[0][COLUMN_PACKET_COUNT])

    def test_traffic_rate_computed_when_missing(self):
        """traffic_rate should be calculated from byte_count/duration if not supplied."""
        row = _make_valid_row(byte_count=1000, duration=2.0)
        row.pop("traffic_rate", None)
        df = normalise_dataframe(pd.DataFrame([row]))
        self.assertAlmostEqual(df.iloc[0][COLUMN_TRAFFIC_RATE], 500.0, places=1)

    def test_source_type_set(self):
        """source_type must be stamped on every row."""
        df = normalise_dataframe(pd.DataFrame([_make_valid_row()]), source_type=SOURCE_CSV)
        self.assertEqual(df.iloc[0][COLUMN_SOURCE_TYPE], SOURCE_CSV)

    def test_empty_dataframe_returns_empty(self):
        """Empty input must produce an empty output with canonical columns."""
        df = normalise_dataframe(pd.DataFrame())
        self.assertTrue(df.empty)
        for col in ALL_COLUMNS:
            self.assertIn(col, df.columns)

    def test_none_input_returns_empty(self):
        """None input must not crash."""
        df = normalise_dataframe(None)
        self.assertTrue(df.empty)

    def test_multiple_rows_mixed_validity(self):
        """Valid rows are kept; invalid rows are dropped."""
        rows = [
            _make_valid_row(),                          # valid
            _make_valid_row(source_ip=None),            # invalid – dropped
            _make_valid_row(protocol="tcp"),            # valid – protocol normalised
        ]
        df = normalise_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(df), 2)

    def test_column_alias_src_ip(self):
        """src_ip should be recognised as an alias for source_ip."""
        raw = {
            "timestamp": "2024-11-01 08:00:00",
            "src_ip": "10.0.0.1",
            "destination_ip": "8.8.8.8",
            "protocol": "TCP",
            "packet_count": 3,
            "byte_count": 300,
        }
        df = normalise_dataframe(pd.DataFrame([raw]))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0][COLUMN_SOURCE_IP], "10.0.0.1")

    def test_icmp_ports_become_none_from_zero(self):
        """ICMP records with port 0 must have source_port and destination_port normalized to None."""
        row = _make_valid_row(protocol="ICMP", source_port=0, destination_port=0)
        df = normalise_dataframe(pd.DataFrame([row]))
        self.assertEqual(len(df), 1)
        self.assertIsNone(df.iloc[0][COLUMN_SOURCE_PORT])
        self.assertIsNone(df.iloc[0][COLUMN_DESTINATION_PORT])

    def test_icmp_ports_become_none_from_integers(self):
        """ICMP records with any port numbers must have source_port and destination_port normalized to None."""
        row = _make_valid_row(protocol="ICMP", source_port=1234, destination_port=5678)
        df = normalise_dataframe(pd.DataFrame([row]))
        self.assertEqual(len(df), 1)
        self.assertIsNone(df.iloc[0][COLUMN_SOURCE_PORT])
        self.assertIsNone(df.iloc[0][COLUMN_DESTINATION_PORT])

    def test_tcp_ports_remain_unchanged(self):
        """TCP ports must be preserved accurately."""
        row = _make_valid_row(protocol="TCP", source_port=49200, destination_port=443)
        df = normalise_dataframe(pd.DataFrame([row]))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0][COLUMN_SOURCE_PORT], 49200)
        self.assertEqual(df.iloc[0][COLUMN_DESTINATION_PORT], 443)

    def test_udp_ports_remain_unchanged(self):
        """UDP ports must be preserved accurately."""
        row = _make_valid_row(protocol="UDP", source_port=54321, destination_port=53)
        df = normalise_dataframe(pd.DataFrame([row]))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0][COLUMN_SOURCE_PORT], 54321)
        self.assertEqual(df.iloc[0][COLUMN_DESTINATION_PORT], 53)

    def test_valid_ipv4_addresses_accepted(self):
        """Valid IPv4 addresses must be accepted."""
        row = _make_valid_row(source_ip="192.168.1.50", destination_ip="10.0.0.1")
        df = normalise_dataframe(pd.DataFrame([row]))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0][COLUMN_SOURCE_IP], "192.168.1.50")
        self.assertEqual(df.iloc[0][COLUMN_DESTINATION_IP], "10.0.0.1")

    def test_valid_ipv6_addresses_accepted(self):
        """Valid IPv6 addresses must be accepted and normalized."""
        row = _make_valid_row(source_ip="2001:db8::1", destination_ip="::1")
        df = normalise_dataframe(pd.DataFrame([row]))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0][COLUMN_SOURCE_IP], "2001:db8::1")
        self.assertEqual(df.iloc[0][COLUMN_DESTINATION_IP], "::1")

    def test_invalid_ip_addresses_rejected(self):
        """Malformed IP addresses must be rejected and row dropped."""
        rows = [
            _make_valid_row(source_ip="999.999.999.999"),
            _make_valid_row(destination_ip="not.an.ip.address"),
            _make_valid_row(source_ip="192.168.1"),
            _make_valid_row(destination_ip="256.0.0.1"),
        ]
        df = normalise_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(df), 0)


class TestTrafficProcessor(unittest.TestCase):

    def test_process_records(self):
        """process_records should normalise a list of raw dicts."""
        processor = TrafficProcessor()
        records = [_make_valid_row(), _make_valid_row(protocol="udp")]
        df = processor.process_records(records)
        self.assertEqual(len(df), 2)
        self.assertTrue(all(df[COLUMN_PROTOCOL].isin(["UDP"])))

    def test_process_empty_records(self):
        processor = TrafficProcessor()
        df = processor.process_records([])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
