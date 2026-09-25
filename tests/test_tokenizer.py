"""Comprehensive unit tests for MiniGPT Character-Level Tokenizer (Phase 2)."""

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tokenizer import CharTokenizer, Tokenizer


class TestCharTokenizer(unittest.TestCase):
    """Test suite covering vocabulary, encoding, decoding, round-trip, and error handling."""

    def setUp(self):
        self.sample_text = "Hello, MiniGPT! 123 \n"
        self.tokenizer = CharTokenizer(self.sample_text)

    def test_1_vocabulary_creation(self):
        """Test 1: Verify that vocabulary contains all unique characters from training text."""
        expected_chars = set(self.sample_text)
        actual_chars = set(self.tokenizer.stoi.keys())
        self.assertEqual(expected_chars, actual_chars)
        self.assertEqual(set(self.tokenizer.itos.values()), expected_chars)

        # Ensure bijection between stoi and itos
        for ch, idx in self.tokenizer.stoi.items():
            self.assertEqual(self.tokenizer.itos[idx], ch)

    def test_2_vocabulary_size(self):
        """Test 2: Verify tokenizer.vocab_size equals len(tokenizer.stoi)."""
        expected_size = len(set(self.sample_text))
        self.assertEqual(self.tokenizer.vocab_size, expected_size)
        self.assertEqual(self.tokenizer.vocab_size, len(self.tokenizer.stoi))
        self.assertEqual(self.tokenizer.vocab_size, len(self.tokenizer.itos))

    def test_3_encoding(self):
        """Test 3: Verify encoding produces integer IDs."""
        text = "Hello"
        tokens = self.tokenizer.encode(text)

        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), len(text))
        for token_id in tokens:
            self.assertIsInstance(token_id, int)
            self.assertNotIsInstance(token_id, bool)
            self.assertTrue(0 <= token_id < self.tokenizer.vocab_size)

    def test_4_decoding(self):
        """Test 4: Verify decoding valid token IDs reproduces original text."""
        tokens = [self.tokenizer.stoi[ch] for ch in "MiniGPT"]
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, "MiniGPT")

    def test_5_round_trip(self):
        """Test 5: Verify decode(encode(text)) == text for known characters."""
        test_strings = [
            self.sample_text,
            "H",
            "123",
            "\n",
            "Hello, MiniGPT!",
            "MiniGPT 123",
            "",  # Empty string edge case
        ]
        for s in test_strings:
            encoded = self.tokenizer.encode(s)
            decoded = self.tokenizer.decode(encoded)
            self.assertEqual(decoded, s, f"Round trip failed for string: {repr(s)}")

    def test_6_unknown_character(self):
        """Test 6: Encoding an unknown character raises a clear ValueError."""
        # '@' is not in self.sample_text
        self.assertNotIn("@", self.tokenizer.stoi)
        with self.assertRaises(ValueError) as ctx:
            self.tokenizer.encode("Hello @ World")
        self.assertIn("Unknown character '@'", str(ctx.exception))
        self.assertIn("not present in vocabulary", str(ctx.exception))

    def test_7_invalid_token_id(self):
        """Test 7: Decoding an invalid token ID raises a clear ValueError."""
        invalid_id = self.tokenizer.vocab_size + 999
        with self.assertRaises(ValueError) as ctx:
            self.tokenizer.decode([invalid_id])
        self.assertIn(f"Invalid token ID {invalid_id}", str(ctx.exception))

        # Negative token ID
        with self.assertRaises(ValueError) as ctx_neg:
            self.tokenizer.decode([-1])
        self.assertIn("Invalid token ID -1", str(ctx_neg.exception))

    def test_8_determinism(self):
        """Test 8: Two tokenizers from the same text produce identical mappings."""
        tok1 = CharTokenizer(self.sample_text)
        tok2 = CharTokenizer(self.sample_text)

        self.assertEqual(tok1.stoi, tok2.stoi)
        self.assertEqual(tok1.itos, tok2.itos)
        self.assertEqual(tok1.vocab_size, tok2.vocab_size)

        test_phrase = "Hello 123"
        self.assertEqual(tok1.encode(test_phrase), tok2.encode(test_phrase))

    def test_alias_tokenizer(self):
        """Verify Tokenizer alias points to CharTokenizer."""
        tok = Tokenizer(self.sample_text)
        self.assertIsInstance(tok, CharTokenizer)
        self.assertEqual(tok.encode("Hello"), self.tokenizer.encode("Hello"))

    def test_input_validation(self):
        """Verify proper validation on invalid argument types."""
        # Non-string to init
        with self.assertRaises(TypeError):
            CharTokenizer(12345)

        # Empty string to init
        with self.assertRaises(ValueError):
            CharTokenizer("")

        # Non-string to encode
        with self.assertRaises(TypeError):
            self.tokenizer.encode(None)
        with self.assertRaises(TypeError):
            self.tokenizer.encode(["H", "e"])

        # Non-iterable to decode
        with self.assertRaises(TypeError):
            self.tokenizer.decode(123)

        # Non-integer in decode sequence
        with self.assertRaises(TypeError):
            self.tokenizer.decode(["0", "1"])
        with self.assertRaises(TypeError):
            self.tokenizer.decode([True])  # bool should be rejected


if __name__ == "__main__":
    unittest.main()
