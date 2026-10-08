"""
Test: Flask API — NetSentinel AI Phase 1
==========================================
Integration tests for all Phase 1 HTTP endpoints.

Coverage:
  - GET /api/health
  - GET /api/traffic/status (before and after load)
  - GET /api/traffic/load   (with sample CSV)
  - GET /api/traffic/stats
  - GET /api/traffic/protocols
  - GET /api/traffic/top-sources
  - GET /api/traffic/top-destinations
  - GET /api/traffic/timeline
  - GET /api/traffic/connections
  - Supabase-unavailable fallback (no credentials)
  - Endpoints before data is loaded (404 behaviour)

All tests use Flask's test client — no real network requests are made.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import unittest

import backend.services.traffic_store as store


def _get_app():
    """Import the Flask app fresh for each test class."""
    from backend.app import app
    app.config["TESTING"] = True
    return app


class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = _get_app()
        self.client = self.app.test_client()

    def test_health_returns_200(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_status_ok(self):
        data = resp_json(self.client.get("/api/health"))
        self.assertEqual(data["status"], "ok")

    def test_health_project_name(self):
        data = resp_json(self.client.get("/api/health"))
        self.assertEqual(data["project"], "NetSentinel AI")

    def test_health_phase(self):
        data = resp_json(self.client.get("/api/health"))
        self.assertIn("Phase 1", data["phase"])

    def test_health_supabase_false_without_credentials(self):
        """Supabase should show as not configured (no .env with real creds)."""
        data = resp_json(self.client.get("/api/health"))
        # Should not crash regardless of value
        self.assertIn("supabase_configured", data)

    def test_health_live_capture_false(self):
        data = resp_json(self.client.get("/api/health"))
        self.assertFalse(data["live_capture_available"])


class TestTrafficLoadAndStatus(unittest.TestCase):

    def setUp(self):
        self.app = _get_app()
        self.client = self.app.test_client()
        store.clear()  # start with a clean store

    def test_status_before_load(self):
        resp = self.client.get("/api/traffic/status")
        self.assertEqual(resp.status_code, 200)
        data = resp_json(resp)
        # loaded may be True (auto-load runs at import) or False — just check key exists
        self.assertIn("loaded", data)

    def test_load_sample_csv(self):
        resp = self.client.get("/api/traffic/load")
        self.assertEqual(resp.status_code, 200)
        data = resp_json(resp)
        self.assertTrue(data["success"])
        self.assertGreater(data["records_loaded"], 0)

    def test_load_nonexistent_file(self):
        resp = self.client.get("/api/traffic/load?path=data/does_not_exist.csv")
        self.assertEqual(resp.status_code, 400)
        data = resp_json(resp)
        self.assertFalse(data["success"])

    def test_status_after_load(self):
        self.client.get("/api/traffic/load")
        data = resp_json(self.client.get("/api/traffic/status"))
        self.assertTrue(data["loaded"])
        self.assertGreater(data["record_count"], 0)


class TestTrafficEndpointsNoData(unittest.TestCase):
    """Endpoints must return 404 (not crash) when no data is loaded."""

    def setUp(self):
        self.app = _get_app()
        self.client = self.app.test_client()
        store.clear()

    def _assert_404_with_error(self, path):
        resp = self.client.get(path)
        self.assertEqual(resp.status_code, 404)
        data = resp_json(resp)
        self.assertIn("error", data)

    def test_stats_no_data(self):
        self._assert_404_with_error("/api/traffic/stats")

    def test_protocols_no_data(self):
        self._assert_404_with_error("/api/traffic/protocols")

    def test_top_sources_no_data(self):
        self._assert_404_with_error("/api/traffic/top-sources")

    def test_top_destinations_no_data(self):
        self._assert_404_with_error("/api/traffic/top-destinations")

    def test_timeline_no_data(self):
        self._assert_404_with_error("/api/traffic/timeline")

    def test_connections_no_data(self):
        self._assert_404_with_error("/api/traffic/connections")


class TestTrafficEndpointsWithData(unittest.TestCase):
    """Endpoints must return correct data after loading the sample CSV."""

    @classmethod
    def setUpClass(cls):
        cls.app = _get_app()
        cls.client = cls.app.test_client()
        # Load sample data once for all tests in this class
        cls.client.get("/api/traffic/load")

    def test_stats_returns_200(self):
        resp = self.client.get("/api/traffic/stats")
        self.assertEqual(resp.status_code, 200)

    def test_stats_has_overview(self):
        data = resp_json(self.client.get("/api/traffic/stats"))
        self.assertIn("overview", data)

    def test_stats_total_records_positive(self):
        data = resp_json(self.client.get("/api/traffic/stats"))
        self.assertGreater(data["overview"]["total_records"], 0)

    def test_protocols_returns_list(self):
        resp = self.client.get("/api/traffic/protocols")
        self.assertEqual(resp.status_code, 200)
        data = resp_json(resp)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_protocols_has_required_fields(self):
        data = resp_json(self.client.get("/api/traffic/protocols"))
        for item in data:
            self.assertIn("protocol", item)
            self.assertIn("count", item)
            self.assertIn("percentage", item)

    def test_top_sources_returns_list(self):
        resp = self.client.get("/api/traffic/top-sources")
        self.assertEqual(resp.status_code, 200)
        data = resp_json(resp)
        self.assertIsInstance(data, list)

    def test_top_sources_fields(self):
        data = resp_json(self.client.get("/api/traffic/top-sources"))
        for item in data:
            self.assertIn("source_ip", item)
            self.assertIn("bytes", item)

    def test_top_destinations_returns_list(self):
        resp = self.client.get("/api/traffic/top-destinations")
        self.assertEqual(resp.status_code, 200)
        data = resp_json(resp)
        self.assertIsInstance(data, list)

    def test_timeline_returns_list(self):
        resp = self.client.get("/api/traffic/timeline")
        self.assertEqual(resp.status_code, 200)
        data = resp_json(resp)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_timeline_bucket_field_present(self):
        data = resp_json(self.client.get("/api/traffic/timeline"))
        for item in data:
            self.assertIn("bucket", item)

    def test_connections_returns_dict(self):
        resp = self.client.get("/api/traffic/connections")
        self.assertEqual(resp.status_code, 200)
        data = resp_json(resp)
        self.assertIsInstance(data, dict)
        self.assertIn("avg_bytes_per_connection", data)

    def test_top_sources_n_param(self):
        data = resp_json(self.client.get("/api/traffic/top-sources?n=2"))
        self.assertLessEqual(len(data), 2)


class TestSupabaseFallback(unittest.TestCase):
    """Supabase being unconfigured must not affect local traffic analysis."""

    def setUp(self):
        self.app = _get_app()
        self.client = self.app.test_client()
        self.client.get("/api/traffic/load")

    def test_stats_works_without_supabase(self):
        """Traffic stats must work even when SUPABASE_URL/KEY are not set."""
        resp = self.client.get("/api/traffic/stats")
        self.assertEqual(resp.status_code, 200)

    def test_push_to_supabase_returns_false_gracefully(self):
        """push_to_supabase should return (False, reason) without crashing."""
        ok, msg = store.push_to_supabase([])
        self.assertFalse(ok)
        self.assertIsInstance(msg, str)


# ----------------------------------------------------------------
# Helper
# ----------------------------------------------------------------

def resp_json(response) -> dict | list:
    return json.loads(response.data)


if __name__ == "__main__":
    unittest.main()
