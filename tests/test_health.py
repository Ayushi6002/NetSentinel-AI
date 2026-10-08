"""
Test: Health Endpoint — NetSentinel AI Phase 1
================================================
Verifies the /api/health endpoint returns the required response.
Kept minimal — comprehensive health tests are in test_api.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unittest
from backend.app import app


class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_returns_200(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

    def test_health_status_ok(self):
        import json
        data = json.loads(self.client.get("/api/health").data)
        self.assertEqual(data.get("status"), "ok")

    def test_health_project_name(self):
        import json
        data = json.loads(self.client.get("/api/health").data)
        self.assertEqual(data.get("project"), "NetSentinel AI")

    def test_health_phase(self):
        import json
        data = json.loads(self.client.get("/api/health").data)
        self.assertIn("Phase 1", data.get("phase", ""))


if __name__ == "__main__":
    unittest.main()
