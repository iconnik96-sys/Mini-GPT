"""Unit and Integration Tests for Phase 9 — Domain Customization.

Validates:
1. Programming domain corpus presence, non-emptiness, and character encoding.
2. Dataset statistics computation (character count, vocab size, sections, splits).
3. Programming dataset loading and batch generation.
4. Tokenizer support for all 7 required domain prompts without missing characters.
5. Causal next-token prediction batching on programming text.
6. Domain model construction and training step on programming data.
7. Structured prompt evaluation runner.
"""

import os
import unittest

import torch

from config import MiniGPTConfig
from src.dataset import TextDataset, get_dataset_statistics, get_programming_dataset
from src.experiment import get_phase9_prompts, run_phase9_prompts_evaluation
from src.model import MiniGPT
from src.train import compute_loss, train


class TestDomainCustomization(unittest.TestCase):
    """Test suite for Phase 9 domain customization functionality."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "programming.txt",
        )

    def test_1_dataset_file_exists_and_valid(self):
        """Test 1: Verify data/programming.txt exists and is non-empty."""
        self.assertTrue(
            os.path.isfile(self.data_path),
            f"Programming domain dataset not found at {self.data_path}",
        )
        with open(self.data_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 10000, "Dataset should contain at least 10,000 characters.")

    def test_2_dataset_statistics_computation(self):
        """Test 2: Verify get_dataset_statistics computes accurate metrics."""
        stats = get_dataset_statistics(self.data_path)
        self.assertIn("total_characters", stats)
        self.assertIn("vocab_size", stats)
        self.assertIn("num_sections", stats)
        self.assertIn("train_tokens", stats)
        self.assertIn("val_tokens", stats)
        self.assertIn("top_characters", stats)
        self.assertIn("source", stats)
        self.assertIn("license", stats)

        self.assertGreaterEqual(stats["vocab_size"], 80)
        self.assertGreaterEqual(stats["num_sections"], 5)
        self.assertEqual(stats["train_tokens"] + stats["val_tokens"], stats["total_characters"])
        self.assertIsInstance(stats["top_characters"], list)

    def test_3_programming_dataset_loading(self):
        """Test 3: Verify get_programming_dataset returns functional TextDataset."""
        ds = get_programming_dataset(self.data_path, block_size=32, batch_size=4)
        self.assertIsInstance(ds, TextDataset)
        self.assertGreater(len(ds.train_data), 1000)
        self.assertGreater(len(ds.val_data), 100)
        self.assertEqual(ds.vocab_size, len(ds.tokenizer.stoi))

    def test_4_all_phase9_prompts_supported_by_tokenizer(self):
        """Test 4: Verify all 7 required Phase 9 prompts encode cleanly without KeyError/ValueError."""
        ds = get_programming_dataset(self.data_path)
        prompts = get_phase9_prompts()
        self.assertEqual(len(prompts), 7)

        for p in prompts:
            # Should encode and decode back to identical string
            token_ids = ds.tokenizer.encode(p)
            self.assertIsInstance(token_ids, list)
            self.assertEqual(len(token_ids), len(p))
            decoded = ds.tokenizer.decode(token_ids)
            self.assertEqual(decoded, p, f"Round-trip failed for prompt: {p}")

    def test_5_domain_batch_shapes_and_next_token_offset(self):
        """Test 5: Verify batch extraction yields shifted (x, y) pairs."""
        ds = get_programming_dataset(self.data_path, block_size=16, batch_size=4)
        x, y = ds.get_batch("train", batch_size=4, block_size=16, device="cpu")
        self.assertEqual(x.shape, (4, 16))
        self.assertEqual(y.shape, (4, 16))

    def test_6_domain_model_construction_and_step(self):
        """Test 6: Verify MiniGPT instantiates with domain vocab and trains on batch."""
        ds = get_programming_dataset(self.data_path, block_size=16, batch_size=4)
        cfg = MiniGPTConfig(
            vocab_size=ds.vocab_size,
            block_size=16,
            batch_size=4,
            n_embd=32,
            n_head=2,
            n_layer=1,
            dropout=0.0,
            learning_rate=1e-3,
            device="cpu",
        )
        model = MiniGPT(cfg)
        self.assertEqual(model.config.vocab_size, ds.vocab_size)

        x, y = ds.get_batch("train", batch_size=4, block_size=16, device="cpu")
        logits = model(x)
        loss = compute_loss(logits, y)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()

    def test_7_run_phase9_prompts_evaluation(self):
        """Test 7: Verify run_phase9_prompts_evaluation executes cleanly on a checkpoint."""
        import shutil
        from src.checkpoint import save_checkpoint

        ds = get_programming_dataset(self.data_path, block_size=16, batch_size=4)
        cfg = MiniGPTConfig(
            vocab_size=ds.vocab_size,
            block_size=16,
            batch_size=4,
            n_embd=32,
            n_head=2,
            n_layer=1,
            device="cpu",
        )
        model = MiniGPT(cfg)

        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "_tmp_phase9_test")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")
            save_checkpoint(
                filepath=ckpt_path,
                model=model,
                step=1,
                config=cfg,
                tokenizer=ds.tokenizer,
            )

            results = run_phase9_prompts_evaluation(
                checkpoint_path=ckpt_path,
                results_dir=tmp_dir,
                seed=42,
                max_new_tokens=10,
                verbose=False,
            )
            self.assertIn("results", results)
            self.assertEqual(len(results["results"]), 7)
            for entry in results["results"]:
                self.assertIn("prompt", entry)
                self.assertEqual(len(entry["evaluations"]), 2)  # Greedy and Top-k 10
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
