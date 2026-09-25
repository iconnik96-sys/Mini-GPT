"""Basic project structure and configuration validation for Phase 1."""

import os
import unittest
import importlib


class TestProjectStructure(unittest.TestCase):
    """Verifies that all required folders, files, and configurations exist."""

    def setUp(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def test_required_directories_exist(self):
        """Ensure all required directories exist."""
        required_dirs = ["data", "checkpoints", "src", "tests"]
        for d in required_dirs:
            dir_path = os.path.join(self.root_dir, d)
            self.assertTrue(os.path.isdir(dir_path), f"Directory missing: {d}")

    def test_required_files_exist(self):
        """Ensure all root and src files exist."""
        required_files = [
            "config.py",
            "requirements.txt",
            "README.md",
            ".gitignore",
            os.path.join("src", "__init__.py"),
            os.path.join("src", "tokenizer.py"),
            os.path.join("src", "dataset.py"),
            os.path.join("src", "model.py"),
            os.path.join("src", "train.py"),
            os.path.join("src", "generate.py"),
        ]
        for f in required_files:
            file_path = os.path.join(self.root_dir, f)
            self.assertTrue(os.path.isfile(file_path), f"File missing: {f}")

    def test_config_parameters_defined(self):
        """Ensure all expected hyperparameters are present in config.py."""
        import sys
        if self.root_dir not in sys.path:
            sys.path.insert(0, self.root_dir)

        import config
        expected_params = [
            "batch_size",
            "block_size",
            "vocab_size",
            "n_embd",
            "n_head",
            "n_layer",
            "dropout",
            "learning_rate",
            "max_iters",
            "eval_interval",
            "eval_iters",
        ]
        for param in expected_params:
            self.assertTrue(
                hasattr(config, param),
                f"Parameter '{param}' missing in config.py"
            )

    def test_src_modules_importable(self):
        """Ensure src modules can be imported cleanly."""
        modules = [
            "src.tokenizer",
            "src.dataset",
            "src.model",
            "src.train",
            "src.generate",
        ]
        for mod_name in modules:
            mod = importlib.import_module(mod_name)
            self.assertIsNotNone(mod, f"Failed to import {mod_name}")


if __name__ == "__main__":
    unittest.main()
