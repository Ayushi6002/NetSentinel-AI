"""
Test: Traffic Analyzer — NetSentinel AI Phase 1
=================================================
Tests for the statistical analysis engine.

Coverage:
  - Overview statistics
  - Protocol distribution
  - Top source IPs
  - Top destination IPs
  - Timeline aggregation
  - Connection statistics
  - Empty DataFrame handling for each function
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
import pandas as pd

from backend.services.traffic_processor import normalise_dataframe
from backend.services import traffic_analyzer as analyzer
from backend.schemas.traffic_schema import ALL_COLUMNS


def _make_test_df() -> pd.DataFrame:
    """Create a small normalised DataFrame for testing."""
    rows = [
        {
            "timestamp": "2024-11-01 08:00:00", "source_ip": "192.168.1.10",
            "destination_ip": "8.8.8.8", "source_port": 54321, "destination_port": 53,
            "protocol": "UDP", "packet_count": 2, "byte_count": 200, "duration": 0.1,
            "traffic_rate": 2000.0,
        },
        {
            "timestamp": "2024-11-01 08:00:10", "source_ip": "192.168.1.10",
            "destination_ip": "142.250.80.46", "source_port": 49200, "destination_port": 443,
            "protocol": "TCP", "packet_count": 18, "byte_count": 24500, "duration": 1.2,
            "traffic_rate": 20416.7,
        },
        {
            "timestamp": "2024-11-01 08:00:20", "source_ip": "192.168.1.25",
            "destination_ip": "8.8.8.8", "source_port": 62300, "destination_port": 53,
            "protocol": "UDP", "packet_count": 2, "byte_count": 172, "duration": 0.05,
            "traffic_rate": 3440.0,
        },
        {
            "timestamp": "2024-11-01 08:01:00", "source_ip": "10.0.0.5",
            "destination_ip": "192.168.1.10", "source_port": None, "destination_port": None,
            "protocol": "ICMP", "packet_count": 4, "byte_count": 336, "duration": 0.2,
            "traffic_rate": 1680.0,
        },
        {
            "timestamp": "2024-11-01 08:01:30", "source_ip": "192.168.1.10",
            "destination_ip": "142.250.80.46", "source_port": 49201, "destination_port": 443,
            "protocol": "TCP", "packet_count": 22, "byte_count": 33500, "duration": 1.6,
            "traffic_rate": 20937.5,
        },
    ]
    return normalise_dataframe(pd.DataFrame(rows))


class TestComputeOverview(unittest.TestCase):

    def setUp(self):
        self.df = _make_test_df()

    def test_total_records(self):
        result = analyzer.compute_overview(self.df)
        self.assertEqual(result["total_records"], 5)

    def test_total_packets(self):
        result = analyzer.compute_overview(self.df)
        # 2+18+2+4+22 = 48
        self.assertEqual(result["total_packets"], 48)

    def test_total_bytes(self):
        result = analyzer.compute_overview(self.df)
        # 200+24500+172+336+33500 = 58708
        self.assertEqual(result["total_bytes"], 58708)

    def test_unique_source_ips(self):
        result = analyzer.compute_overview(self.df)
        # 192.168.1.10, 192.168.1.25, 10.0.0.5 => 3
        self.assertEqual(result["unique_source_ips"], 3)

    def test_unique_destination_ips(self):
        result = analyzer.compute_overview(self.df)
        # 8.8.8.8, 142.250.80.46, 192.168.1.10 => 3
        self.assertEqual(result["unique_destination_ips"], 3)

    def test_total_bytes_mb(self):
        result = analyzer.compute_overview(self.df)
        expected_mb = round(58708 / (1024 * 1024), 4)
        self.assertAlmostEqual(result["total_bytes_mb"], expected_mb, places=4)

    def test_empty_df_returns_zeros(self):
        result = analyzer.compute_overview(pd.DataFrame(columns=ALL_COLUMNS))
        self.assertEqual(result["total_records"], 0)
        self.assertEqual(result["total_packets"], 0)
        self.assertEqual(result["total_bytes"], 0)


class TestProtocolDistribution(unittest.TestCase):

    def setUp(self):
        self.df = _make_test_df()

    def test_returns_list(self):
        result = analyzer.compute_protocol_distribution(self.df)
        self.assertIsInstance(result, list)

    def test_all_protocols_present(self):
        result = analyzer.compute_protocol_distribution(self.df)
        protocols = {r["protocol"] for r in result}
        self.assertIn("TCP", protocols)
        self.assertIn("UDP", protocols)
        self.assertIn("ICMP", protocols)

    def test_percentage_sums_to_100(self):
        result = analyzer.compute_protocol_distribution(self.df)
        total = sum(r["percentage"] for r in result)
        self.assertAlmostEqual(total, 100.0, places=1)

    def test_tcp_count(self):
        result = analyzer.compute_protocol_distribution(self.df)
        tcp = next(r for r in result if r["protocol"] == "TCP")
        self.assertEqual(tcp["count"], 2)

    def test_empty_df_returns_empty_list(self):
        result = analyzer.compute_protocol_distribution(pd.DataFrame(columns=ALL_COLUMNS))
        self.assertEqual(result, [])


class TestTopSources(unittest.TestCase):

    def setUp(self):
        self.df = _make_test_df()

    def test_returns_list(self):
        result = analyzer.compute_top_sources(self.df)
        self.assertIsInstance(result, list)

    def test_top_source_is_highest_bytes(self):
        result = analyzer.compute_top_sources(self.df)
        # 192.168.1.10 has most bytes: 200+24500+33500 = 58200
        self.assertEqual(result[0]["source_ip"], "192.168.1.10")

    def test_n_parameter_respected(self):
        result = analyzer.compute_top_sources(self.df, n=2)
        self.assertLessEqual(len(result), 2)

    def test_empty_df_returns_empty_list(self):
        result = analyzer.compute_top_sources(pd.DataFrame(columns=ALL_COLUMNS))
        self.assertEqual(result, [])


class TestTopDestinations(unittest.TestCase):

    def setUp(self):
        self.df = _make_test_df()

    def test_returns_list(self):
        result = analyzer.compute_top_destinations(self.df)
        self.assertIsInstance(result, list)

    def test_top_destination_is_highest_bytes(self):
        result = analyzer.compute_top_destinations(self.df)
        # 142.250.80.46 has 24500+33500 = 58000 bytes
        self.assertEqual(result[0]["destination_ip"], "142.250.80.46")

    def test_empty_df_returns_empty_list(self):
        result = analyzer.compute_top_destinations(pd.DataFrame(columns=ALL_COLUMNS))
        self.assertEqual(result, [])


class TestTimeline(unittest.TestCase):

    def setUp(self):
        self.df = _make_test_df()

    def test_returns_list(self):
        result = analyzer.compute_timeline(self.df, freq="1min")
        self.assertIsInstance(result, list)

    def test_buckets_are_non_empty(self):
        result = analyzer.compute_timeline(self.df, freq="1min")
        self.assertGreater(len(result), 0)

    def test_bucket_keys_present(self):
        result = analyzer.compute_timeline(self.df, freq="1min")
        for entry in result:
            self.assertIn("bucket", entry)
            self.assertIn("records", entry)
            self.assertIn("packets", entry)
            self.assertIn("bytes", entry)

    def test_total_records_matches(self):
        result = analyzer.compute_timeline(self.df, freq="1min")
        total = sum(r["records"] for r in result)
        self.assertEqual(total, len(self.df))

    def test_empty_df_returns_empty_list(self):
        result = analyzer.compute_timeline(pd.DataFrame(columns=ALL_COLUMNS))
        self.assertEqual(result, [])


class TestConnectionStats(unittest.TestCase):

    def setUp(self):
        self.df = _make_test_df()

    def test_returns_dict(self):
        result = analyzer.compute_connection_stats(self.df)
        self.assertIsInstance(result, dict)

    def test_max_bytes_is_positive(self):
        result = analyzer.compute_connection_stats(self.df)
        self.assertGreater(result["max_bytes_connection"], 0)

    def test_top_protocol_present(self):
        result = analyzer.compute_connection_stats(self.df)
        self.assertIn(result["top_protocol_by_connections"], {"TCP", "UDP", "ICMP"})

    def test_empty_df_returns_zeros(self):
        result = analyzer.compute_connection_stats(pd.DataFrame(columns=ALL_COLUMNS))
        self.assertEqual(result["max_bytes_connection"], 0)
        self.assertIsNone(result["top_protocol_by_connections"])


class TestAnalyseAll(unittest.TestCase):

    def test_all_keys_present(self):
        df = _make_test_df()
        result = analyzer.analyse_all(df)
        for key in ("overview", "protocols", "top_sources", "top_destinations",
                    "timeline", "connection_stats"):
            self.assertIn(key, result, f"Missing key in analyse_all: {key}")


if __name__ == "__main__":
    unittest.main()
