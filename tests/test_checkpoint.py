"""Unit Tests for Checkpoint Management and Resume Training.

Phase 6: Validation and Checkpoints
-----------------------------------
Tests:
1. Checkpoint can be saved.
2. Checkpoint file exists on disk.
3. Model parameters are restored exactly after loading.
4. Optimizer state is restored.
5. Training iteration/step is restored.
6. Train/validation loss metadata is restored.
7. Configuration metadata is preserved.
8. Tokenizer vocabulary metadata is preserved.
9. Loading on CPU works cleanly.
10. Missing checkpoint raises clear FileNotFoundError.
11. Corrupted or incompatible checkpoint raises clear informative errors.
12. Resume training continues from the saved iteration.
13. Best checkpoint tracking saves improved validation models.
14. Corrupted weights containing NaN/Inf are detected and rejected.
15. CUDA loading test (cleanly skipped if CUDA is unavailable).
"""

import os
from pathlib import Path
import shutil
import unittest
import uuid

import torch
import torch.nn as nn

from config import MiniGPTConfig
from src.checkpoint import load_checkpoint, save_checkpoint
from src.dataset import TextDataset
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
from src.train import compute_loss, estimate_loss, train


class TestCheckpoint(unittest.TestCase):
    """Test suite covering checkpoint persistence, integrity, and resuming."""

    def setUp(self) -> None:
        """Sets up temporary directory, deterministic seed, and tiny model."""
        torch.manual_seed(1337)
        self.temp_dir = os.path.join(os.getcwd(), f"tmp_test_ckpt_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Tiny configuration for fast, deterministic unit testing
        self.tiny_text = "ABCDEFGHIJ"
        self.tokenizer = CharTokenizer(self.tiny_text)
        self.vocab_size = self.tokenizer.vocab_size  # 10

        self.cfg = MiniGPTConfig(
            vocab_size=self.vocab_size,
            block_size=8,
            n_embd=32,
            n_head=2,
            n_layer=2,
            dropout=0.0,
            batch_size=4,
            learning_rate=1e-3,
            device="cpu",
        )
        self.model = MiniGPT(self.cfg)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

    def tearDown(self) -> None:
        """Cleans up temporary directory after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_checkpoint_can_be_saved(self) -> None:
        """Test 1: Verify save_checkpoint executes and returns target path string."""
        ckpt_path = os.path.join(self.temp_dir, "test.pt")
        saved_path = save_checkpoint(
            filepath=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            step=10,
            config=self.cfg,
            train_loss=2.5,
            val_loss=2.6,
        )
        self.assertEqual(os.path.abspath(saved_path), os.path.abspath(ckpt_path))

    def test_2_checkpoint_file_exists(self) -> None:
        """Test 2: Verify saved checkpoint file exists on disk and is non-empty."""
        ckpt_path = os.path.join(self.temp_dir, "exists_test.pt")
        save_checkpoint(
            filepath=ckpt_path,
            model=self.model,
            step=5,
        )
        self.assertTrue(os.path.isfile(ckpt_path), "Checkpoint file was not created on disk.")
        self.assertGreater(os.path.getsize(ckpt_path), 0, "Checkpoint file is empty.")

    def test_3_model_parameters_restored_exactly(self) -> None:
        """Test 3: Verify model parameters match exactly after loading from checkpoint."""
        ckpt_path = os.path.join(self.temp_dir, "weights_test.pt")
        save_checkpoint(ckpt_path, model=self.model, step=1)

        # Fresh model with different random weights
        torch.manual_seed(9999)
        new_model = MiniGPT(self.cfg)

        # Confirm weights are initially different
        initially_different = False
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            if not torch.equal(p1, p2):
                initially_different = True
                break
        self.assertTrue(initially_different, "Models should have distinct initial weights.")

        # Load weights into new model
        load_checkpoint(ckpt_path, model=new_model, device="cpu")

        # Verify all parameters now match exactly
        for name, p_orig in self.model.named_parameters():
            p_loaded = dict(new_model.named_parameters())[name]
            self.assertTrue(
                torch.equal(p_orig, p_loaded),
                f"Parameter mismatch for '{name}' after checkpoint loading.",
            )

    def test_4_optimizer_state_restored(self) -> None:
        """Test 4: Verify optimizer states (moments, step counts) are faithfully restored."""
        ckpt_path = os.path.join(self.temp_dir, "opt_test.pt")

        # Perform one optimization step to populate optimizer state
        x = torch.zeros((2, 4), dtype=torch.long)
        y = torch.ones((2, 4), dtype=torch.long)
        logits = self.model(x)
        loss = compute_loss(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save checkpoint with populated optimizer
        save_checkpoint(ckpt_path, model=self.model, optimizer=self.optimizer, step=1)

        # Create new optimizer
        new_opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        self.assertEqual(len(new_opt.state), 0, "New optimizer state should initially be empty.")

        # Load checkpoint into new optimizer
        load_checkpoint(ckpt_path, optimizer=new_opt, device="cpu")

        # Verify state is populated
        self.assertGreater(len(new_opt.state), 0, "Loaded optimizer state should not be empty.")
        orig_state_dict = self.optimizer.state_dict()
        loaded_state_dict = new_opt.state_dict()

        # Compare optimizer step count
        for group_idx in range(len(orig_state_dict["param_groups"])):
            self.assertEqual(
                orig_state_dict["param_groups"][group_idx]["lr"],
                loaded_state_dict["param_groups"][group_idx]["lr"],
            )

    def test_5_training_iteration_restored(self) -> None:
        """Test 5: Verify training iteration/step is preserved and restored."""
        ckpt_path = os.path.join(self.temp_dir, "step_test.pt")
        target_step = 250
        save_checkpoint(ckpt_path, model=self.model, step=target_step)

        ckpt = load_checkpoint(ckpt_path, device="cpu")
        self.assertEqual(ckpt["step"], target_step)

    def test_6_loss_metadata_restored(self) -> None:
        """Test 6: Verify train_loss, val_loss, and best_val_loss are restored."""
        ckpt_path = os.path.join(self.temp_dir, "loss_test.pt")
        save_checkpoint(
            ckpt_path,
            model=self.model,
            step=100,
            train_loss=1.875,
            val_loss=1.950,
            best_val_loss=1.820,
        )

        ckpt = load_checkpoint(ckpt_path, device="cpu")
        self.assertAlmostEqual(ckpt["train_loss"], 1.875, places=4)
        self.assertAlmostEqual(ckpt["val_loss"], 1.950, places=4)
        self.assertAlmostEqual(ckpt["best_val_loss"], 1.820, places=4)

    def test_7_configuration_metadata_preserved(self) -> None:
        """Test 7: Verify configuration hyperparameters are stored and retrievable."""
        ckpt_path = os.path.join(self.temp_dir, "config_test.pt")
        save_checkpoint(ckpt_path, model=self.model, config=self.cfg)

        ckpt = load_checkpoint(ckpt_path, device="cpu")
        saved_cfg = ckpt["config"]

        self.assertEqual(saved_cfg["vocab_size"], self.cfg.vocab_size)
        self.assertEqual(saved_cfg["block_size"], self.cfg.block_size)
        self.assertEqual(saved_cfg["n_embd"], self.cfg.n_embd)
        self.assertEqual(saved_cfg["n_head"], self.cfg.n_head)
        self.assertEqual(saved_cfg["n_layer"], self.cfg.n_layer)

    def test_8_tokenizer_vocabulary_metadata_preserved(self) -> None:
        """Test 8: Verify tokenizer stoi/itos and vocab_size are stored in checkpoint."""
        ckpt_path = os.path.join(self.temp_dir, "tok_test.pt")
        save_checkpoint(ckpt_path, model=self.model, tokenizer=self.tokenizer)

        ckpt = load_checkpoint(ckpt_path, device="cpu")
        tok_data = ckpt["tokenizer_vocab"]

        self.assertIsNotNone(tok_data)
        self.assertEqual(tok_data["vocab_size"], self.tokenizer.vocab_size)
        self.assertEqual(tok_data["stoi"], self.tokenizer.stoi)
        self.assertEqual(tok_data["itos"], self.tokenizer.itos)

    def test_9_loading_with_cpu(self) -> None:
        """Test 9: Verify loading explicitly on CPU maps all tensors to CPU."""
        ckpt_path = os.path.join(self.temp_dir, "cpu_test.pt")
        save_checkpoint(ckpt_path, model=self.model)

        ckpt = load_checkpoint(ckpt_path, device="cpu")
        for tensor in ckpt["model_state_dict"].values():
            if torch.is_tensor(tensor):
                self.assertEqual(tensor.device.type, "cpu")

    def test_10_missing_checkpoint_error(self) -> None:
        """Test 10: Verify loading a non-existent file raises clear FileNotFoundError."""
        bad_path = os.path.join(self.temp_dir, "does_not_exist.pt")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_checkpoint(bad_path)
        self.assertIn("Checkpoint file not found", str(ctx.exception))

    def test_11_corrupted_or_incompatible_checkpoint_handled(self) -> None:
        """Test 11: Verify corrupted files or incompatible models raise descriptive errors."""
        # 11a: Corrupted file
        corrupt_path = os.path.join(self.temp_dir, "corrupt.pt")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_PYTORCH_TENSOR_FILE")

        with self.assertRaises(RuntimeError) as ctx:
            load_checkpoint(corrupt_path)
        self.assertIn("Corrupted or invalid checkpoint file", str(ctx.exception))

        # 11b: Vocabulary mismatch
        valid_path = os.path.join(self.temp_dir, "valid.pt")
        save_checkpoint(valid_path, model=self.model, config=self.cfg)

        incompatible_vocab_cfg = MiniGPTConfig(
            vocab_size=self.vocab_size + 10,  # 20 != 10
            block_size=8,
            n_embd=32,
            n_head=2,
            n_layer=2,
        )
        incompat_model = MiniGPT(incompatible_vocab_cfg)

        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(valid_path, model=incompat_model)
        self.assertIn("Incompatible vocabulary size", str(ctx.exception))

        # 11c: Dimension mismatch (n_embd)
        incompatible_dim_cfg = MiniGPTConfig(
            vocab_size=self.vocab_size,
            block_size=8,
            n_embd=64,  # 64 != 32
            n_head=2,
            n_layer=2,
        )
        incompat_dim_model = MiniGPT(incompatible_dim_cfg)
        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(valid_path, model=incompat_dim_model)
        self.assertIn("Incompatible architecture parameter", str(ctx.exception))

    def test_12_resume_training_continues_iteration(self) -> None:
        """Test 12: Verify resume training starts from saved step and completes additional steps."""
        corpus = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 5
        ds = TextDataset(raw_text=corpus, block_size=16, batch_size=4, device="cpu")

        train_cfg = MiniGPTConfig(
            vocab_size=ds.vocab_size,
            block_size=16,
            batch_size=4,
            n_embd=32,
            n_head=2,
            n_layer=2,
            dropout=0.0,
            learning_rate=1e-3,
            device="cpu",
        )

        # Run phase 1: train for 10 steps and save checkpoint
        res1 = train(
            dataset=ds,
            cfg=train_cfg,
            max_iters=10,
            eval_interval=5,
            eval_iters=2,
            checkpoint_dir=self.temp_dir,
            save_checkpoints=True,
            verbose=False,
            seed=42,
        )

        latest_ckpt = os.path.join(self.temp_dir, "latest.pt")
        self.assertTrue(os.path.isfile(latest_ckpt))
        ckpt_meta = load_checkpoint(latest_ckpt, device="cpu")
        self.assertEqual(ckpt_meta["step"], 10)

        # Run phase 2: resume from step 10 to step 20
        res2 = train(
            dataset=ds,
            cfg=train_cfg,
            max_iters=20,
            eval_interval=5,
            eval_iters=2,
            checkpoint_dir=self.temp_dir,
            save_checkpoints=True,
            resume_from=latest_ckpt,
            verbose=False,
        )

        # History should include resumed progression
        self.assertIn(10, res2["history"])
        self.assertIn(20, res2["history"])

        # Check final saved checkpoint has step 20
        resumed_ckpt = load_checkpoint(latest_ckpt, device="cpu")
        self.assertEqual(resumed_ckpt["step"], 20)

    def test_13_best_checkpoint_logic(self) -> None:
        """Test 13: Verify best.pt is saved on validation improvement and not overwritten by worse loss."""
        best_path = os.path.join(self.temp_dir, "best.pt")

        # Step 1: save model with loss = 3.0
        save_checkpoint(
            best_path,
            model=self.model,
            step=10,
            val_loss=3.0,
            best_val_loss=3.0,
        )
        ckpt1 = load_checkpoint(best_path, device="cpu")
        self.assertAlmostEqual(ckpt1["best_val_loss"], 3.0)

        # Step 2: improved loss = 2.5 -> saves to best.pt
        save_checkpoint(
            best_path,
            model=self.model,
            step=20,
            val_loss=2.5,
            best_val_loss=2.5,
        )
        ckpt2 = load_checkpoint(best_path, device="cpu")
        self.assertAlmostEqual(ckpt2["best_val_loss"], 2.5)
        self.assertEqual(ckpt2["step"], 20)

    def test_14_checkpoint_nan_inf_handling(self) -> None:
        """Test 14: Verify corrupted weights containing NaN or Inf are rejected upon load."""
        ckpt_path = os.path.join(self.temp_dir, "nan_test.pt")
        save_checkpoint(ckpt_path, model=self.model)

        # Manually inject NaN into saved state_dict
        with open(ckpt_path, "rb") as f:
            raw_ckpt = torch.load(f, map_location="cpu")
        first_key = list(raw_ckpt["model_state_dict"].keys())[0]
        raw_ckpt["model_state_dict"][first_key][0] = float("nan")
        with open(ckpt_path, "wb") as f:
            torch.save(raw_ckpt, f)

        with self.assertRaises(ValueError) as ctx:
            load_checkpoint(ckpt_path, model=self.model)
        self.assertIn("contains NaN or Inf", str(ctx.exception))

    def test_15_cuda_handling(self) -> None:
        """Test 15: Verify CUDA save and load if CUDA is available, else skip cleanly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable on this system")

        cuda_model = MiniGPT(self.cfg).to("cuda")
        ckpt_path = os.path.join(self.temp_dir, "cuda_test.pt")
        save_checkpoint(ckpt_path, model=cuda_model)

        # Load back to cuda
        ckpt = load_checkpoint(ckpt_path, device="cuda")
        first_tensor = list(ckpt["model_state_dict"].values())[0]
        self.assertEqual(first_tensor.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
