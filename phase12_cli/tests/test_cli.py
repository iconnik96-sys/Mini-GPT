"""
Unit and Integration Tests for Phase 12 CLI Application.

Tests cover:
- CLI argument parser configuration and subcommand resolution
- Model registry catalog and metadata integrity
- Valid and invalid model ID validation
- GenerationSettings validation (temperature, top_k, max_tokens, seed)
- Missing checkpoint error handling
- Empty prompt rejection
- Character tokenizer OOV check
- Formatting functions
"""

import unittest
from unittest.mock import patch, MagicMock

from phase12_cli.cli import build_parser
from phase12_cli.config import GenerationSettings, CLIConfig
from phase12_cli.src.model_registry import (
    list_models,
    get_model_info,
    is_valid_model,
    REGISTRY
)
from phase12_cli.src.generator import check_character_oov, generate_text
from phase12_cli.src.model_loader import ModelSession, ModelMetadata
from phase12_cli.src.formatting import (
    format_banner,
    format_models_table,
    format_model_info,
    format_settings,
    format_error
)


class TestPhase12CLI(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_1_parser_subcommands(self):
        """Verify CLI parser resolves all required subcommands."""
        args_models = self.parser.parse_args(["models"])
        self.assertEqual(args_models.subcommand, "models")

        args_info = self.parser.parse_args(["info", "distilgpt2-lora"])
        self.assertEqual(args_info.subcommand, "info")
        self.assertEqual(args_info.model, "distilgpt2-lora")

        args_gen = self.parser.parse_args(["generate", "--model", "minigpt-programming", "--prompt", "Hello"])
        self.assertEqual(args_gen.subcommand, "generate")
        self.assertEqual(args_gen.model, "minigpt-programming")
        self.assertEqual(args_gen.prompt, "Hello")

        args_chat = self.parser.parse_args(["chat", "--model", "distilgpt2"])
        self.assertEqual(args_chat.subcommand, "chat")
        self.assertEqual(args_chat.model, "distilgpt2")

    def test_2_model_registry_contents(self):
        """Verify model registry contains the 4 expected models with valid metadata."""
        models = list_models()
        self.assertGreaterEqual(len(models), 3)
        self.assertTrue(is_valid_model("distilgpt2-lora"))
        self.assertTrue(is_valid_model("minigpt-programming"))
        self.assertTrue(is_valid_model("distilgpt2"))
        self.assertFalse(is_valid_model("unknown-model-xyz"))

        lora_meta = get_model_info("distilgpt2-lora")
        self.assertEqual(lora_meta.parameter_count, 82060032)
        self.assertEqual(lora_meta.trainable_parameters, 147456)
        self.assertEqual(lora_meta.tokenizer_type, "bpe")

    def test_3_invalid_model_id_error(self):
        """Verify requesting an invalid model raises a clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_model_info("non_existent_model")
        self.assertIn("Unknown model", str(ctx.exception))
        self.assertIn("Available models", str(ctx.exception))

    def test_4_generation_settings_validation(self):
        """Verify validation rejects invalid generation hyperparameters."""
        # Negative / zero max tokens
        with self.assertRaises(ValueError):
            s = GenerationSettings(max_new_tokens=0)
            s.validate()

        with self.assertRaises(ValueError):
            s = GenerationSettings(max_new_tokens=-10)
            s.validate()

        # Non-positive temperature
        with self.assertRaises(ValueError):
            s = GenerationSettings(temperature=0.0)
            s.validate()

        with self.assertRaises(ValueError):
            s = GenerationSettings(temperature=-0.5)
            s.validate()

        # Non-positive top_k
        with self.assertRaises(ValueError):
            s = GenerationSettings(top_k=0)
            s.validate()

        with self.assertRaises(ValueError):
            s = GenerationSettings(top_k=-5)
            s.validate()

        # Valid settings pass cleanly
        valid_s = GenerationSettings(max_new_tokens=50, temperature=0.7, top_k=20, seed=42)
        valid_s.validate()

    def test_5_empty_prompt_rejection(self):
        """Verify empty prompts are rejected with a clear error."""
        meta = get_model_info("distilgpt2")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            generate_text(
                model=mock_model,
                tokenizer=mock_tokenizer,
                metadata=meta,
                prompt="   ",
                settings=GenerationSettings()
            )
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_6_character_oov_detection(self):
        """Verify check_character_oov detects missing characters."""
        class DummyTokenizer:
            stoi = {"a": 0, "b": 1, " ": 2}
        tok = DummyTokenizer()
        missing = check_character_oov(tok, "abc @")
        self.assertEqual(missing, ["@", "c"])

    def test_7_missing_checkpoint_handling(self):
        """Verify ModelSession raises FileNotFoundError if checkpoint file is missing."""
        session = ModelSession()
        fake_meta = ModelMetadata(
            id="fake_model",
            display_name="Fake Model",
            model_type="from_scratch",
            checkpoint_path="non/existent/checkpoint.pt",
            tokenizer_type="character",
            parameter_count=1000,
            trainable_parameters=1000,
            frozen_parameters=0,
            trainable_percentage=100.0,
            context_length=64,
            description="Fake"
        )
        with patch("phase12_cli.src.model_loader.get_model_info", return_value=fake_meta):
            with self.assertRaises(FileNotFoundError) as ctx:
                session.get_or_load("fake_model")
            self.assertIn("Checkpoint not found", str(ctx.exception))

    def test_8_formatting_helpers(self):
        """Verify banner, tables, and settings formatters produce valid strings."""
        banner = format_banner()
        self.assertIn("MiniGPT", banner)

        models_table = format_models_table(list_models())
        self.assertIn("distilgpt2-lora", models_table)
        self.assertIn("minigpt-programming", models_table)

        settings_str = format_settings(GenerationSettings())
        self.assertIn("max_new_tokens", settings_str)
        self.assertIn("temperature", settings_str)

        error_str = format_error("Sample error message")
        self.assertIn("[Error]", error_str)
        self.assertIn("Sample error message", error_str)

    def test_9_session_caching(self):
        """Verify ModelSession avoids re-loading when requesting already cached model."""
        session = ModelSession()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_meta = get_model_info("distilgpt2")
        
        session.current_model_id = "distilgpt2"
        session.model = mock_model
        session.tokenizer = mock_tokenizer
        session.metadata = mock_meta

        m, t, meta = session.get_or_load("distilgpt2")
        self.assertIs(m, mock_model)
        self.assertIs(t, mock_tokenizer)
        self.assertEqual(meta.id, "distilgpt2")

        session.clear()
        self.assertIsNone(session.model)
        self.assertIsNone(session.current_model_id)


if __name__ == "__main__":
    unittest.main()
