"""
Unit tests for Models metadata endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from phase13_api_web.backend.app import app


class TestModelsEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_list_models_returns_all_registered(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        models = data["models"]
        self.assertGreaterEqual(len(models), 4)

        model_ids = {m["id"] for m in models}
        expected_ids = {"minigpt-baseline", "minigpt-programming", "distilgpt2", "distilgpt2-lora"}
        self.assertTrue(expected_ids.issubset(model_ids))

    def test_get_valid_model_info_minigpt_programming(self):
        res = self.client.get("/api/models/minigpt-programming")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["id"], "minigpt-programming")
        self.assertEqual(data["tokenizer"], "character")
        self.assertEqual(data["parameters"], 114944)
        self.assertEqual(data["trainable_parameters"], 114944)
        self.assertEqual(data["trainable_percentage"], 100.0)

    def test_get_valid_model_info_distilgpt2_lora(self):
        res = self.client.get("/api/models/distilgpt2-lora")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["id"], "distilgpt2-lora")
        self.assertEqual(data["tokenizer"], "bpe")
        self.assertEqual(data["parameters"], 82060032)
        self.assertEqual(data["trainable_parameters"], 147456)
        self.assertAlmostEqual(data["trainable_percentage"], 0.1797, places=3)
        self.assertEqual(data["type"], "peft_lora")

    def test_get_unknown_model_returns_404(self):
        res = self.client.get("/api/models/non-existent-model-xyz")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())


if __name__ == "__main__":
    unittest.main()
