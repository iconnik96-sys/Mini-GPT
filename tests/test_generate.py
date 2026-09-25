"""Unit Tests for Autoregressive Text Generation.

Phase 7: Text Generation
-------------------------
Tests:
1. Generation returns a string.
2. Prompt is preserved in the generated output.
3. Output can contain newly generated characters.
4. max_new_tokens=0 returns the original prompt.
5. Generation respects block_size (crops context window).
6. Temperature validation works (rejects <= 0).
7. Top-k validation works (rejects <= 0 or invalid types).
8. Top-k limits the candidate distribution (top_k=1 equals greedy).
9. Greedy generation is deterministic across multiple calls.
10. Seeded sampling is reproducible.
11. Generation does not create gradients.
12. Model parameters do not change during generation.
13. Model training/evaluation state is handled correctly.
14. CPU generation works cleanly.
15. CUDA generation is tested only when CUDA is available.
16. Checkpoint-based generation works.
17. Unknown prompt characters produce a clear error.
"""

import os
from pathlib import Path
import shutil
import unittest
import uuid

import torch

from config import MiniGPTConfig
from src.checkpoint import save_checkpoint
from src.generate import generate, generate_from_checkpoint, load_from_checkpoint
from src.model import MiniGPT
from src.tokenizer import CharTokenizer


class TestGenerate(unittest.TestCase):
    """Test suite covering autoregressive generation, sampling, and CLI helpers."""

    def setUp(self) -> None:
        """Sets up deterministic seed, tiny model, tokenizer, and temp directory."""
        torch.manual_seed(1337)

        # Corpus with common letters, spaces, punctuation, and newline
        self.tiny_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,\n"
        self.tokenizer = CharTokenizer(self.tiny_text)
        self.vocab_size = self.tokenizer.vocab_size

        self.cfg = MiniGPTConfig(
            vocab_size=self.vocab_size,
            block_size=8,
            n_embd=32,
            n_head=2,
            n_layer=2,
            dropout=0.0,
            device="cpu",
        )
        self.model = MiniGPT(self.cfg)
        self.model.eval()

        self.temp_dir = os.path.join(os.getcwd(), f"tmp_test_gen_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self) -> None:
        """Cleans up temporary directory after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_generation_returns_string(self) -> None:
        """Test 1: Verify generate() returns a string."""
        out = generate(self.model, self.tokenizer, prompt="A", max_new_tokens=5)
        self.assertIsInstance(out, str, f"Expected str, got {type(out).__name__}.")

    def test_2_prompt_preserved(self) -> None:
        """Test 2: Verify prompt is preserved at the beginning of the generated output."""
        prompt = "Hello"
        out = generate(self.model, self.tokenizer, prompt=prompt, max_new_tokens=8)
        self.assertTrue(out.startswith(prompt), f"Generated text '{out}' does not start with prompt '{prompt}'.")

    def test_3_output_contains_newly_generated_chars(self) -> None:
        """Test 3: Verify output length equals prompt length plus max_new_tokens."""
        prompt = "The"
        num_new = 12
        out = generate(self.model, self.tokenizer, prompt=prompt, max_new_tokens=num_new)
        expected_len = len(prompt) + num_new
        self.assertEqual(len(out), expected_len, f"Expected length {expected_len}, got {len(out)}.")

    def test_4_max_new_tokens_zero_returns_prompt(self) -> None:
        """Test 4: Verify max_new_tokens=0 returns original prompt unchanged."""
        prompt = "ExactPrompt"
        out = generate(self.model, self.tokenizer, prompt=prompt, max_new_tokens=0)
        self.assertEqual(out, prompt, "Expected prompt to be returned unchanged when max_new_tokens=0.")

    def test_5_generation_respects_block_size(self) -> None:
        """Test 5: Verify sequence cropping works when prompt + generated tokens exceeds block_size."""
        # Configured block_size is 8. Prompt length 10 exceeds block_size.
        prompt = "ABCDEFGHIJ"  # length 10 > 8
        out = generate(self.model, self.tokenizer, prompt=prompt, max_new_tokens=12)
        # Should successfully generate 12 new tokens without shape error
        self.assertEqual(len(out), 10 + 12)
        self.assertTrue(out.startswith(prompt))

    def test_6_temperature_validation(self) -> None:
        """Test 6: Verify non-positive temperature values are rejected with ValueError."""
        with self.assertRaises(ValueError) as ctx:
            generate(self.model, self.tokenizer, prompt="A", temperature=0.0, greedy=False)
        self.assertIn("Temperature must be strictly positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            generate(self.model, self.tokenizer, prompt="A", temperature=-0.5, greedy=False)
        self.assertIn("Temperature must be strictly positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            generate(self.model, self.tokenizer, prompt="A", max_new_tokens=-1)
        self.assertIn("max_new_tokens must be non-negative", str(ctx.exception))

    def test_7_top_k_validation(self) -> None:
        """Test 7: Verify non-positive or non-integer top_k values are rejected."""
        with self.assertRaises(ValueError) as ctx:
            generate(self.model, self.tokenizer, prompt="A", top_k=0)
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            generate(self.model, self.tokenizer, prompt="A", top_k=-5)
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            generate(self.model, self.tokenizer, prompt="A", top_k="invalid")  # type: ignore
        self.assertIn("top_k must be a positive integer", str(ctx.exception))

    def test_8_top_k_limits_candidate_distribution(self) -> None:
        """Test 8: Verify top_k=1 is equivalent to deterministic greedy decoding."""
        prompt = "Test"
        out_top1 = generate(
            self.model, self.tokenizer, prompt=prompt, max_new_tokens=10, top_k=1, seed=42
        )
        out_greedy = generate(
            self.model, self.tokenizer, prompt=prompt, max_new_tokens=10, greedy=True
        )
        self.assertEqual(out_top1, out_greedy, "top_k=1 should match greedy argmax decoding.")

    def test_9_greedy_generation_is_deterministic(self) -> None:
        """Test 9: Verify greedy decoding produces identical output across separate runs."""
        prompt = "Deterministic"
        out1 = generate(self.model, self.tokenizer, prompt=prompt, max_new_tokens=15, greedy=True)
        out2 = generate(self.model, self.tokenizer, prompt=prompt, max_new_tokens=15, greedy=True)
        self.assertEqual(out1, out2, "Greedy generation must be completely deterministic.")

    def test_10_seeded_sampling_is_reproducible(self) -> None:
        """Test 10: Verify stochastic sampling is reproducible when seed is specified."""
        prompt = "Seed"
        out1 = generate(
            self.model, self.tokenizer, prompt=prompt, max_new_tokens=15, temperature=1.2, seed=1234
        )
        out2 = generate(
            self.model, self.tokenizer, prompt=prompt, max_new_tokens=15, temperature=1.2, seed=1234
        )
        self.assertEqual(out1, out2, "Seeded generation must produce identical text.")

    def test_11_generation_creates_no_gradients(self) -> None:
        """Test 11: Verify generation runs under no_grad and populates zero gradients."""
        # Ensure parameters require grad
        for p in self.model.parameters():
            p.requires_grad = True
            p.grad = None

        generate(self.model, self.tokenizer, prompt="Grad", max_new_tokens=10)

        for name, p in self.model.named_parameters():
            self.assertIsNone(p.grad, f"Parameter '{name}' should have no gradient after inference.")

    def test_12_model_parameters_do_not_change(self) -> None:
        """Test 12: Verify model weights are bit-for-bit identical before and after generation."""
        before_state = {name: p.clone() for name, p in self.model.named_parameters()}

        generate(self.model, self.tokenizer, prompt="Weights", max_new_tokens=20, greedy=True)

        for name, p in self.model.named_parameters():
            self.assertTrue(
                torch.equal(before_state[name], p),
                f"Parameter '{name}' mutated during generation.",
            )

    def test_13_model_mode_handled_correctly(self) -> None:
        """Test 13: Verify model training state is preserved and restored after generation."""
        # 13a: Model initially in train mode
        self.model.train()
        self.assertTrue(self.model.training)
        generate(self.model, self.tokenizer, prompt="Train", max_new_tokens=5)
        self.assertTrue(self.model.training, "Model should be restored to train mode.")

        # 13b: Model initially in eval mode
        self.model.eval()
        self.assertFalse(self.model.training)
        generate(self.model, self.tokenizer, prompt="Eval", max_new_tokens=5)
        self.assertFalse(self.model.training, "Model should remain in eval mode.")

    def test_14_cpu_generation_works(self) -> None:
        """Test 14: Verify explicit CPU generation operates cleanly."""
        out = generate(self.model, self.tokenizer, prompt="CPU", max_new_tokens=6, device="cpu")
        self.assertTrue(out.startswith("CPU"))
        self.assertEqual(len(out), 3 + 6)

    def test_15_cuda_generation(self) -> None:
        """Test 15: Verify CUDA generation when CUDA is available, else skip cleanly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable on this system")

        cuda_model = MiniGPT(self.cfg).to("cuda")
        out = generate(cuda_model, self.tokenizer, prompt="GPU", max_new_tokens=6, device="cuda")
        self.assertTrue(out.startswith("GPU"))
        self.assertEqual(len(out), 3 + 6)

    def test_16_checkpoint_based_generation(self) -> None:
        """Test 16: Verify model and tokenizer restoration from checkpoint for inference."""
        ckpt_path = os.path.join(self.temp_dir, "gen_test.pt")
        save_checkpoint(
            filepath=ckpt_path,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.cfg,
            step=50,
        )

        # Generate directly using helper
        out = generate_from_checkpoint(
            checkpoint_path=ckpt_path,
            prompt="FromCkpt",
            max_new_tokens=8,
            device="cpu",
            greedy=True,
        )
        self.assertTrue(out.startswith("FromCkpt"))
        self.assertEqual(len(out), len("FromCkpt") + 8)

        # Test load_from_checkpoint
        restored_model, restored_tok = load_from_checkpoint(ckpt_path, device="cpu")
        self.assertIsInstance(restored_model, MiniGPT)
        self.assertEqual(restored_tok.vocab_size, self.tokenizer.vocab_size)

    def test_17_unknown_prompt_characters_error(self) -> None:
        """Test 17: Verify prompt with character not in vocabulary raises clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            # '@' is not in tiny_text
            generate(self.model, self.tokenizer, prompt="Unknown@Char")
        self.assertIn("not present in vocabulary", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
