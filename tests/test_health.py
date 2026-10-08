import unittest
from backend.app import app


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("project"), "NetSentinel AI")
        self.assertEqual(data.get("phase"), "Phase 1 - Network Traffic Intelligence")


if __name__ == '__main__':
    unittest.main()
