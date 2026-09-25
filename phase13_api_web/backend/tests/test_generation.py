"""
Unit and integration tests for generation and streaming endpoints.
"""

import unittest
import json
from fastapi.testclient import TestClient
from phase13_api_web.backend.app import app


class TestGenerationEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_validation_empty_prompt_rejected(self):
        payload = {
            "model": "minigpt-programming",
            "prompt": "   ",
            "max_new_tokens": 20,
        }
        res = self.client.post("/api/generate", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_validation_invalid_temperature_rejected(self):
        payload = {
            "model": "minigpt-programming",
            "prompt": "class Animal",
            "temperature": -0.5,
        }
        res = self.client.post("/api/generate", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_validation_invalid_max_tokens_rejected(self):
        payload = {
            "model": "minigpt-programming",
            "prompt": "class Animal",
            "max_new_tokens": 0,
        }
        res = self.client.post("/api/generate", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_validation_unknown_model_rejected(self):
        payload = {
            "model": "gpt-99-turbo",
            "prompt": "Hello world",
            "max_new_tokens": 10,
        }
        res = self.client.post("/api/generate", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unknown model", res.json()["detail"])

    def test_character_oov_rejection(self):
        # minigpt-baseline was trained on Shakespeare (no digits 0-9)
        payload = {
            "model": "minigpt-baseline",
            "prompt": "PORT 8080",
            "max_new_tokens": 10,
        }
        res = self.client.post("/api/generate", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("characters not supported", res.json()["detail"].lower())

    def test_real_generation_minigpt_programming(self):
        # Test real generation with from-scratch model
        payload = {
            "model": "minigpt-programming",
            "prompt": "class Service",
            "max_new_tokens": 25,
            "temperature": 0.8,
            "top_k": 20,
            "seed": 42,
        }
        res = self.client.post("/api/generate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model"], "minigpt-programming")
        self.assertEqual(data["prompt"], "class Service")
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 0)
        self.assertEqual(data["generated_tokens"], 25)
        self.assertGreater(data["generation_time_seconds"], 0.0)
        self.assertGreater(data["tokens_per_second"], 0.0)
        self.assertEqual(data["token_unit"], "char")

    def test_streaming_generation_minigpt_programming(self):
        # Test Server-Sent Events (SSE) streaming endpoint
        payload = {
            "model": "minigpt-programming",
            "prompt": "public void",
            "max_new_tokens": 15,
            "temperature": 0.8,
            "top_k": 20,
            "seed": 42,
        }
        res = self.client.post("/api/generate/stream", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers.get("content-type", ""))

        lines = [line.strip() for line in res.text.split("\n") if line.strip().startswith("data:")]
        self.assertGreaterEqual(len(lines), 2)

        # Check token payloads
        token_lines = lines[:-1]
        for line in token_lines:
            content = json.loads(line.replace("data:", "").strip())
            self.assertIn("token", content)

        # Check completion payload
        done_payload = json.loads(lines[-1].replace("data:", "").strip())
        self.assertTrue(done_payload.get("done"))
        self.assertEqual(done_payload.get("tokens_generated"), 15)
        self.assertIn("generation_time_seconds", done_payload)
        self.assertIn("tokens_per_second", done_payload)


if __name__ == "__main__":
    unittest.main()
