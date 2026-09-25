"""Unit Tests for Phase 8 Experiments and Metric Tracking.

Phase 8: Experiments and Improvements
-------------------------------------
Tests:
1. Perplexity computation from cross-entropy loss.
2. Perplexity numerical safety (overflow handling for large loss).
3. Perplexity lower bound (loss <= 0 returns 1.0).
4. Parameter count calculation accuracy.
5. Tokens-per-second throughput computation.
6. Safe division handling for zero/near-zero duration.
7. Sampling configuration list completeness (7 configurations).
8. Single experiment execution and JSON persistence.
9. Experiment JSON payload schema validation.
10. Sampling experiment execution across all 7 decoding setups.
"""

import json
import math
import os
import shutil
import unittest
import uuid

import torch
import torch.nn as nn

from config import MiniGPTConfig
from src.checkpoint import save_checkpoint
from src.dataset import TextDataset
from src.experiment import (
    calculate_tokens_per_sec,
    compute_perplexity,
    count_parameters,
    get_default_sampling_configs,
    run_experiment,
    run_sampling_experiment,
)
from src.model import MiniGPT
from src.tokenizer import CharTokenizer


class TestExperiments(unittest.TestCase):
    """Test suite covering metrics, experiment tracking, and JSON persistence."""

    def setUp(self) -> None:
        """Sets up temporary workspace directory for test artifacts."""
        torch.manual_seed(1337)
        self.temp_dir = os.path.join(os.getcwd(), f"tmp_test_exp_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self) -> None:
        """Cleans up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_compute_perplexity_standard(self) -> None:
        """Test 1: Verify perplexity equals exp(loss) for standard values."""
        loss_val = math.log(52.0)  # ~3.9512
        ppl = compute_perplexity(loss_val)
        self.assertAlmostEqual(ppl, 52.0, places=4)

        loss_zero = 0.0
        self.assertEqual(compute_perplexity(loss_zero), 1.0)

    def test_2_compute_perplexity_overflow_safety(self) -> None:
        """Test 2: Verify high loss does not raise OverflowError and clamps safely."""
        huge_loss = 250.0  # exp(250) would overflow standard float
        ppl = compute_perplexity(huge_loss)
        self.assertTrue(math.isfinite(ppl))
        self.assertAlmostEqual(ppl, math.exp(50.0), delta=1e-3)

    def test_3_compute_perplexity_non_positive_loss(self) -> None:
        """Test 3: Verify non-positive loss returns 1.0 perplexity."""
        self.assertEqual(compute_perplexity(-5.0), 1.0)

    def test_4_count_parameters(self) -> None:
        """Test 4: Verify count_parameters accurately counts trainable weights."""
        cfg = MiniGPTConfig(
            vocab_size=10,
            block_size=8,
            n_embd=16,
            n_head=2,
            n_layer=1,
            dropout=0.0,
        )
        model = MiniGPT(cfg)
        expected_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        counted = count_parameters(model)
        self.assertEqual(counted, expected_params)
        self.assertGreater(counted, 0)

    def test_5_calculate_tokens_per_sec(self) -> None:
        """Test 5: Verify tokens/sec calculation."""
        tokens = 1000
        duration = 2.0
        tps = calculate_tokens_per_sec(tokens, duration)
        self.assertEqual(tps, 500.0)

    def test_6_tokens_per_sec_zero_duration_safe(self) -> None:
        """Test 6: Verify zero duration does not raise ZeroDivisionError."""
        tps = calculate_tokens_per_sec(100, 0.0)
        self.assertTrue(math.isfinite(tps))
        self.assertGreater(tps, 0)

    def test_7_default_sampling_configs(self) -> None:
        """Test 7: Verify 7 standard decoding configurations are defined."""
        configs = get_default_sampling_configs()
        self.assertEqual(len(configs), 7)
        names = [c["name"] for c in configs]
        self.assertTrue(any("Greedy" in n for n in names))
        self.assertTrue(any("0.5" in n for n in names))
        self.assertTrue(any("0.8" in n for n in names))
        self.assertTrue(any("1.0" in n for n in names))
        self.assertTrue(any("Top-k 5" in n for n in names))
        self.assertTrue(any("Top-k 10" in n for n in names))
        self.assertTrue(any("Top-k 20" in n for n in names))

    def test_8_run_experiment_saves_json(self) -> None:
        """Test 8: Verify run_experiment executes training and saves JSON result."""
        corpus = "ABCDEFGHIJ\n" * 10
        ds = TextDataset(raw_text=corpus, block_size=8, batch_size=4, device="cpu")
        cfg = MiniGPTConfig(
            vocab_size=ds.vocab_size,
            block_size=8,
            batch_size=4,
            n_embd=16,
            n_head=2,
            n_layer=1,
            dropout=0.0,
            learning_rate=1e-3,
            device="cpu",
        )

        res = run_experiment(
            name="test_mini_exp",
            cfg=cfg,
            max_iters=5,
            eval_interval=5,
            eval_iters=2,
            dataset=ds,
            checkpoint_dir=self.temp_dir,
            results_dir=self.temp_dir,
            verbose=False,
        )

        json_path = os.path.join(self.temp_dir, "test_mini_exp.json")
        self.assertTrue(os.path.isfile(json_path), "Experiment JSON was not written to disk.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["experiment_name"], "test_mini_exp")
        self.assertIn("metrics", data)
        self.assertIn("model_architecture", data)
        self.assertIn("generations", data)
        self.assertIn("val_perplexity", data["metrics"])
        self.assertIn("tokens_per_second", data["metrics"])

    def test_9_sampling_experiment_all_configurations(self) -> None:
        """Test 9: Verify run_sampling_experiment generates outputs for all 7 sampling setups."""
        tiny_text = "ABCDEFGHIJ"
        tokenizer = CharTokenizer(tiny_text)
        cfg = MiniGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=8,
            n_embd=16,
            n_head=2,
            n_layer=1,
            device="cpu",
        )
        model = MiniGPT(cfg)

        ckpt_path = os.path.join(self.temp_dir, "sampling_test.pt")
        save_checkpoint(ckpt_path, model=model, tokenizer=tokenizer, config=cfg)

        res = run_sampling_experiment(
            checkpoint_path=ckpt_path,
            prompt="A",
            max_new_tokens=5,
            results_dir=self.temp_dir,
            verbose=False,
        )

        json_path = os.path.join(self.temp_dir, "exp4_sampling.json")
        self.assertTrue(os.path.isfile(json_path))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["samples"]), 7)
        for s in data["samples"]:
            self.assertTrue(s["output"].startswith("A"))
            self.assertEqual(len(s["output"]), 1 + 5)


if __name__ == "__main__":
    unittest.main()
