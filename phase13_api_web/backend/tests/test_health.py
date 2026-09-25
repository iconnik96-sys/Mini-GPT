"""
Unit tests for Health check endpoint.
"""

import unittest
from fastapi.testclient import TestClient
from phase13_api_web.backend.app import app


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["service"], "MiniGPT Studio API")

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("MiniGPT", data["service"])
        self.assertIn("version", data)


if __name__ == "__main__":
    unittest.main()
