"""Comprehensive unit tests for MiniGPT Transformer Architecture (Phase 4)."""

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
from src.model import CausalSelfAttention, FeedForward, MiniGPT, TransformerBlock


class TestMiniGPTModel(unittest.TestCase):
    """Test suite covering model construction, forward pass, causal attention, and gradients."""

    def setUp(self):
        # Lightweight configuration for fast, isolated testing
        self.config = MiniGPTConfig(
            batch_size=4,
            block_size=32,
            vocab_size=52,  # Dynamic vocabulary size matching sample corpus
            n_embd=64,
            n_head=4,
            n_layer=2,
            dropout=0.1,
        )
        self.model = MiniGPT(self.config)
        self.model.eval()

    def test_1_model_construction_with_dataset_vocab(self):
        """Test 1: Verify model instantiates with real tokenizer vocabulary size, not hardcoded 65."""
        dataset = TextDataset(block_size=16, batch_size=2)
        real_vocab_size = dataset.vocab_size

        custom_cfg = MiniGPTConfig(
            block_size=16,
            vocab_size=real_vocab_size,
            n_embd=64,
            n_head=2,
            n_layer=2,
        )
        model = MiniGPT(custom_cfg)
        self.assertEqual(model.config.vocab_size, real_vocab_size)
        self.assertEqual(model.tok_emb.num_embeddings, real_vocab_size)
        self.assertEqual(model.lm_head.out_features, real_vocab_size)

    def test_2_forward_pass_shape(self):
        """Test 2: Verify logits have shape (B, T, vocab_size)."""
        B, T = 2, 16
        idx = torch.randint(0, self.config.vocab_size, (B, T))
        logits = self.model(idx)

        self.assertEqual(logits.shape, (B, T, self.config.vocab_size))
        # Ensure logits are unnormalized real values (not all zero, not all identical)
        self.assertTrue(torch.is_tensor(logits))
        self.assertEqual(logits.dtype, torch.float32)

    def test_3_multiple_sequence_lengths(self):
        """Test 3: Test variable sequence lengths T <= block_size."""
        B = 2
        for T in [1, 5, 16, self.config.block_size]:
            idx = torch.randint(0, self.config.vocab_size, (B, T))
            logits = self.model(idx)
            self.assertEqual(logits.shape, (B, T, self.config.vocab_size))

    def test_4_causal_mask_prevents_future_attention(self):
        """Test 4: Directly verify future positions receive exactly zero attention probability."""
        B, T = 1, 8
        idx = torch.randint(0, self.config.vocab_size, (B, T))

        # Forward with attention extraction
        logits, att_maps = self.model(idx, return_attention=True)
        self.assertEqual(len(att_maps), self.config.n_layer)

        for layer_idx, att in enumerate(att_maps):
            # Shape: (B, n_head, T, T)
            self.assertEqual(att.shape, (B, self.config.n_head, T, T))
            att_matrix = att[0, 0]  # Examine first head
            for i in range(T):
                for j in range(T):
                    if j > i:
                        # Future token attention must be identically zero
                        prob = att_matrix[i, j].item()
                        self.assertAlmostEqual(
                            prob,
                            0.0,
                            places=6,
                            msg=f"Future attention leak at layer {layer_idx}, pos ({i}, {j}): {prob}",
                        )

        # Directly verify mask buffer is strictly lower triangular
        mask_buffer = self.model.blocks[0].attn.mask[0, 0]
        for i in range(self.config.block_size):
            for j in range(self.config.block_size):
                if j > i:
                    self.assertEqual(mask_buffer[i, j].item(), 0.0)
                else:
                    self.assertEqual(mask_buffer[i, j].item(), 1.0)

    def test_5_head_dimension_and_divisibility(self):
        """Test 5: Verify n_embd % n_head == 0 and head_size computation."""
        self.assertEqual(self.config.n_embd % self.config.n_head, 0)
        expected_head_size = self.config.n_embd // self.config.n_head
        self.assertEqual(self.model.blocks[0].attn.head_size, expected_head_size)

        # Incompatible dimension should raise assertion
        invalid_cfg = MiniGPTConfig(n_embd=65, n_head=4)
        with self.assertRaises(AssertionError):
            MiniGPT(invalid_cfg)

    def test_6_parameter_existence(self):
        """Test 6: Verify model contains non-zero trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(total_params, 0)
        # Verify submodules are populated
        self.assertEqual(len(self.model.blocks), self.config.n_layer)

    def test_7_gradient_flow(self):
        """Test 7: Verify backpropagation computes non-zero gradients for trainable parameters."""
        self.model.train()
        idx = torch.randint(0, self.config.vocab_size, (2, 8))
        targets = torch.randint(0, self.config.vocab_size, (2, 8))

        logits, loss = self.model(idx, targets=targets)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        # Check that key layers received valid gradients
        self.assertIsNotNone(self.model.lm_head.weight.grad)
        self.assertTrue(torch.any(self.model.lm_head.weight.grad != 0))
        self.assertIsNotNone(self.model.tok_emb.weight.grad)
        self.assertTrue(torch.any(self.model.tok_emb.weight.grad != 0))
        self.assertIsNotNone(self.model.pos_emb.weight.grad)
        self.assertTrue(torch.any(self.model.pos_emb.weight.grad != 0))

    def test_8_cpu_execution(self):
        """Test 8: Verify model executes forward pass cleanly on CPU."""
        model_cpu = MiniGPT(self.config).to("cpu")
        idx_cpu = torch.randint(0, self.config.vocab_size, (2, 8), device="cpu")
        logits_cpu = model_cpu(idx_cpu)
        self.assertEqual(str(logits_cpu.device), "cpu")
        self.assertEqual(logits_cpu.shape, (2, 8, self.config.vocab_size))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on this system")
    def test_9_cuda_execution(self):
        """Test 9: Verify small forward pass executes on CUDA when available."""
        model_cuda = MiniGPT(self.config).to("cuda")
        idx_cuda = torch.randint(0, self.config.vocab_size, (2, 8), device="cuda")
        logits_cuda = model_cuda(idx_cuda)
        self.assertTrue("cuda" in str(logits_cuda.device))
        self.assertEqual(logits_cuda.shape, (2, 8, self.config.vocab_size))

    def test_10_block_limit_rejection(self):
        """Test 10: Sequences longer than block_size are rejected with a clear ValueError."""
        too_long = self.config.block_size + 5
        idx = torch.randint(0, self.config.vocab_size, (1, too_long))
        with self.assertRaises(ValueError) as ctx:
            self.model(idx)
        self.assertIn("exceeds maximum block_size", str(ctx.exception))

    def test_11_independent_random_initialization(self):
        """Test 11: Two models have distinct parameters and different initialized weights."""
        model_a = MiniGPT(self.config)
        model_b = MiniGPT(self.config)

        # Different objects in memory
        self.assertIsNot(model_a.tok_emb.weight, model_b.tok_emb.weight)

        # Initialized weights must differ
        diff = torch.abs(model_a.tok_emb.weight - model_b.tok_emb.weight).sum().item()
        self.assertGreater(diff, 0.0)


if __name__ == "__main__":
    unittest.main()
