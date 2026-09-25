"""Comprehensive unit tests for MiniGPT Dataset and Batching (Phase 3)."""

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.dataset import TextDataset, get_batch
from src.tokenizer import CharTokenizer


class TestTextDataset(unittest.TestCase):
    """Test suite covering data loading, tokenization, splits, batching, and context shifting."""

    @classmethod
    def setUpClass(cls):
        # Use existing data/input.txt
        cls.data_path = os.path.join(PROJECT_ROOT, "data", "input.txt")
        # Ensure data file exists for tests
        assert os.path.isfile(cls.data_path), f"Sample data missing at {cls.data_path}"
        with open(cls.data_path, "r", encoding="utf-8") as f:
            cls.raw_text = f.read()

    def setUp(self):
        # Initialize dataset with moderate block_size and batch_size for rapid testing
        self.block_size = 32
        self.batch_size = 4
        self.dataset = TextDataset(
            file_path=self.data_path,
            block_size=self.block_size,
            batch_size=self.batch_size,
            train_split=0.9,
            device="cpu",
            seed=42,
        )

    def test_1_dataset_loading(self):
        """Test 1: Verify that corpus loads successfully from file and validates correctly."""
        self.assertGreater(len(self.dataset.raw_text), 0)
        self.assertEqual(self.dataset.raw_text, self.raw_text)

        # Non-existent file should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            TextDataset(file_path="non_existent_file.txt", block_size=16, batch_size=2)

    def test_2_tokenization(self):
        """Test 2: Verify that dataset uses CharTokenizer and produces integer token IDs."""
        self.assertIsInstance(self.dataset.tokenizer, CharTokenizer)
        self.assertGreater(self.dataset.vocab_size, 0)

        # Check train and val data are non-empty sequences of integers
        train_list = self.dataset.train_data.tolist()
        val_list = self.dataset.val_data.tolist()
        self.assertIsInstance(train_list, list)
        self.assertIsInstance(val_list, list)
        self.assertIsInstance(train_list[0], int)
        self.assertIsInstance(val_list[0], int)

        # Roundtrip check: decoding train + val tokens should restore original text
        full_tokens = train_list + val_list
        reconstructed_text = self.dataset.tokenizer.decode(full_tokens)
        self.assertEqual(reconstructed_text, self.raw_text)

    def test_3_train_val_split(self):
        """Test 3: Verify train/val splits exist, train > val (90/10), deterministic, no overlap."""
        total_tokens = len(self.dataset.train_data) + len(self.dataset.val_data)
        self.assertEqual(total_tokens, len(self.raw_text))

        # Check 90/10 proportionality
        self.assertGreater(len(self.dataset.train_data), len(self.dataset.val_data))
        expected_train_len = int(len(self.raw_text) * 0.9)
        self.assertEqual(len(self.dataset.train_data), expected_train_len)
        self.assertEqual(len(self.dataset.val_data), len(self.raw_text) - expected_train_len)

        # Determinism: creating another dataset with same parameters yields exact same splits
        ds2 = TextDataset(file_path=self.data_path, block_size=self.block_size, batch_size=self.batch_size)
        self.assertEqual(self.dataset.train_data.tolist(), ds2.train_data.tolist())
        self.assertEqual(self.dataset.val_data.tolist(), ds2.val_data.tolist())

        # No overlap check: slices in raw text correspond cleanly to the partition
        self.assertEqual(
            self.dataset.tokenizer.decode(self.dataset.train_data.tolist()),
            self.raw_text[:expected_train_len],
        )
        self.assertEqual(
            self.dataset.tokenizer.decode(self.dataset.val_data.tolist()),
            self.raw_text[expected_train_len:],
        )

    def test_4_context_window_shape(self):
        """Test 4: Verify x and y shapes match (batch_size, block_size)."""
        for split in ("train", "val"):
            x, y = self.dataset.get_batch(split)
            self.assertEqual(x.shape, (self.batch_size, self.block_size))
            self.assertEqual(y.shape, (self.batch_size, self.block_size))

    def test_5_next_token_relationship(self):
        """Test 5: Verify contiguous next-token relationship: y is x shifted by exactly 1 token."""
        for split in ("train", "val"):
            x, y = self.dataset.get_batch(split)
            x_list = x.tolist()
            y_list = y.tolist()

            for b in range(self.batch_size):
                # For every step j, y[b][j] must match x[b][j + 1]
                for j in range(self.block_size - 1):
                    self.assertEqual(
                        y_list[b][j],
                        x_list[b][j + 1],
                        f"Mismatch at batch {b}, position {j}: y[b][{j}] != x[b][{j+1}]",
                    )

                # Verify that x and y form a contiguous slice [x[0] ... x[-1], y[-1]] in the source data
                combined = x_list[b] + [y_list[b][-1]]
                source_data = (
                    self.dataset.train_data.tolist()
                    if split == "train"
                    else self.dataset.val_data.tolist()
                )
                # Find occurrences in source data
                start_token = combined[0]
                found = False
                for idx in range(len(source_data) - len(combined) + 1):
                    if source_data[idx : idx + len(combined)] == combined:
                        found = True
                        break
                self.assertTrue(found, "Batch slice (x, y) was not found as contiguous sequence in source data")

    def test_6_valid_token_ids(self):
        """Test 6: Verify all token IDs in batches are within 0 <= token_id < vocab_size."""
        vocab_size = self.dataset.vocab_size
        x, y = self.dataset.get_batch("train")

        for row in x.tolist() + y.tolist():
            for token_id in row:
                self.assertGreaterEqual(token_id, 0)
                self.assertLess(token_id, vocab_size)

    def test_7_batch_size_configuration(self):
        """Test 7: Verify that custom batch_size parameter is respected."""
        for custom_bsz in [1, 2, 7, 16]:
            x, y = self.dataset.get_batch("train", batch_size=custom_bsz)
            self.assertEqual(x.shape, (custom_bsz, self.block_size))
            self.assertEqual(y.shape, (custom_bsz, self.block_size))

    def test_8_device_placement(self):
        """Test 8: Verify that tensors are placed on the configured device."""
        x_cpu, y_cpu = self.dataset.get_batch("train", device="cpu")
        self.assertEqual(str(x_cpu.device), "cpu")
        self.assertEqual(str(y_cpu.device), "cpu")

        # Config device check
        x_cfg, y_cfg = self.dataset.get_batch("train")
        self.assertEqual(str(x_cfg.device), str(self.dataset.device))

    def test_9_repeated_batches(self):
        """Test 9: Call get_batch multiple times and verify shapes and validity."""
        for _ in range(5):
            x, y = self.dataset.get_batch("train")
            self.assertEqual(x.shape, (self.batch_size, self.block_size))
            self.assertEqual(y.shape, (self.batch_size, self.block_size))
            # Verify no NaN or null entries
            self.assertFalse(any(t is None for t in x.tolist()))
            self.assertFalse(any(t is None for t in y.tolist()))

    def test_10_small_dataset_validation(self):
        """Test 10: Verify clear ValueError when corpus is too small for block_size."""
        tiny_text = "Hello!"  # Only 6 characters
        with self.assertRaises(ValueError) as ctx:
            TextDataset(raw_text=tiny_text, block_size=10, batch_size=2)
        self.assertIn("too small for block_size", str(ctx.exception))

    def test_module_level_get_batch(self):
        """Verify module-level get_batch helper function works seamlessly."""
        x, y = get_batch("train", batch_size=2)
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(x.shape[1], config.block_size)


if __name__ == "__main__":
    unittest.main()
