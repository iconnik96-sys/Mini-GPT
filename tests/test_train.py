"""Comprehensive unit tests for MiniGPT Training Loop (Phase 5)."""

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from config import MiniGPTConfig
from src.dataset import TextDataset
from src.model import MiniGPT
from src.train import compute_loss, estimate_loss, train


class TestTrainingLoop(unittest.TestCase):
    """Test suite covering loss calculation, backprop, optimizer updates, and training flow."""

    def setUp(self):
        # Lightweight setup for rapid, isolated unit testing
        self.tiny_text = (
            "First Citizen:\n"
            "Before we proceed any further, hear me speak.\n"
            "All:\n"
            "Speak, speak.\n"
            "First Citizen:\n"
            "You are all resolved rather to die than to famish?\n"
            "All:\n"
            "Resolved. resolved.\n"
            "First Citizen:\n"
            "First, you know Caius Marcius is chief enemy to the people.\n"
        )
        self.block_size = 16
        self.batch_size = 2
        self.dataset = TextDataset(
            raw_text=self.tiny_text,
            block_size=self.block_size,
            batch_size=self.batch_size,
            train_split=0.8,
            device="cpu",
            seed=42,
        )
        self.config = MiniGPTConfig(
            batch_size=self.batch_size,
            block_size=self.block_size,
            vocab_size=self.dataset.vocab_size,
            n_embd=32,
            n_head=2,
            n_layer=2,
            dropout=0.0,
            learning_rate=1e-3,
        )
        self.model = MiniGPT(self.config).to("cpu")

    def test_1_loss_computation_finite_scalar(self):
        """Test 1: Verify loss is a finite scalar tensor."""
        x, y = self.dataset.get_batch("train", batch_size=2)
        logits = self.model(x)
        loss = compute_loss(logits, y)

        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_2_loss_shape_scalar(self):
        """Test 2: Verify loss.ndim == 0 (zero-dimensional scalar tensor)."""
        x, y = self.dataset.get_batch("train", batch_size=2)
        logits = self.model(x)
        loss = compute_loss(logits, y)

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_3_backpropagation_gradients(self):
        """Test 3: Verify loss.backward() populates gradients across trainable parameters."""
        self.model.train()
        x, y = self.dataset.get_batch("train", batch_size=2)
        logits = self.model(x)
        loss = compute_loss(logits, y)

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        # Check gradients exist on key submodules
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad, f"Parameter {name} has None gradient after backward()."
                )
                self.assertTrue(
                    torch.any(param.grad != 0),
                    f"Parameter {name} gradient is all zeros.",
                )

    def test_4_optimizer_updates_parameters(self):
        """Test 4: Verify optimizer.step() updates model weights."""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)

        x, y = self.dataset.get_batch("train", batch_size=2)
        logits = self.model(x)
        loss = compute_loss(logits, y)

        # Record parameter before update
        initial_param = self.model.lm_head.weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Verify weights have changed
        updated_param = self.model.lm_head.weight.detach()
        param_diff = torch.norm(updated_param - initial_param).item()
        self.assertGreater(
            param_diff, 0.0, "Weights did not change after optimizer step."
        )

    def test_5_training_step_no_nan(self):
        """Test 5: Run multiple consecutive training steps and confirm no NaN or Inf."""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        for step in range(5):
            x, y = self.dataset.get_batch("train", batch_size=2)
            logits = self.model(x)
            loss = compute_loss(logits, y)

            self.assertFalse(torch.isnan(loss).item(), f"NaN loss at step {step}")
            self.assertFalse(torch.isinf(loss).item(), f"Inf loss at step {step}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def test_6_loss_estimation_no_grad(self):
        """Test 6: Verify estimate_loss returns finite values and does not track gradients."""
        losses = estimate_loss(self.model, self.dataset, eval_iters=3, device="cpu")

        self.assertIn("train", losses)
        self.assertIn("val", losses)
        self.assertIsInstance(losses["train"], float)
        self.assertIsInstance(losses["val"], float)
        self.assertTrue(torch.isfinite(torch.tensor(losses["train"])))
        self.assertTrue(torch.isfinite(torch.tensor(losses["val"])))

        # Ensure no gradients were created during estimation
        for param in self.model.parameters():
            self.assertIsNone(param.grad)

    def test_7_train_eval_mode_switching(self):
        """Test 7: Verify model restores training mode after estimate_loss."""
        self.model.train()
        self.assertTrue(self.model.training)

        estimate_loss(self.model, self.dataset, eval_iters=2, device="cpu")
        self.assertTrue(
            self.model.training,
            "Model training mode was not restored after estimate_loss.",
        )

        # If model was in eval mode, it should remain in eval mode
        self.model.eval()
        estimate_loss(self.model, self.dataset, eval_iters=2, device="cpu")
        self.assertFalse(self.model.training)

    def test_8_device_handling(self):
        """Test 8: Verify CPU training; skip CUDA if unavailable."""
        # CPU verified
        result = train(
            model=self.model,
            dataset=self.dataset,
            max_iters=2,
            eval_interval=2,
            eval_iters=1,
            device="cpu",
            verbose=False,
        )
        self.assertIn("final_train_loss", result)

        # CUDA conditional
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable on this system")
        else:
            cuda_model = MiniGPT(self.config).to("cuda")
            res_cuda = train(
                model=cuda_model,
                dataset=self.dataset,
                max_iters=2,
                eval_interval=2,
                eval_iters=1,
                device="cuda",
                verbose=False,
            )
            self.assertIn("final_train_loss", res_cuda)

    def test_9_vocabulary_consistency(self):
        """Test 9: Verify model vocabulary size matches tokenizer vocabulary size."""
        self.assertEqual(self.model.config.vocab_size, self.dataset.vocab_size)
        self.assertEqual(self.model.lm_head.out_features, self.dataset.vocab_size)

        # Verify mismatch raises ValueError
        mismatched_cfg = MiniGPTConfig(
            block_size=self.block_size,
            vocab_size=self.dataset.vocab_size + 10,
            n_embd=32,
            n_head=2,
            n_layer=2,
        )
        mismatched_model = MiniGPT(mismatched_cfg)
        with self.assertRaises(ValueError):
            train(
                model=mismatched_model,
                dataset=self.dataset,
                max_iters=1,
                verbose=False,
            )

    def test_10_tiny_overfit_sanity(self):
        """Test 10: Verify that training on tiny data decreases loss."""
        torch.manual_seed(1337)
        # Train on tiny data for 30 steps with a small model
        result = train(
            dataset=self.dataset,
            cfg=self.config,
            max_iters=30,
            eval_interval=15,
            eval_iters=5,
            learning_rate=3e-3,
            device="cpu",
            seed=1337,
            verbose=False,
        )

        initial_loss = result["history"][0]["train"]
        final_loss = result["history"][30]["train"]

        # Final loss must decrease relative to random initialization
        self.assertLess(
            final_loss,
            initial_loss,
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
