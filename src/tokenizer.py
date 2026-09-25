"""Character-Level Tokenizer for MiniGPT.

Phase 2: Character-Level Tokenizer
-----------------------------------
A clean, deterministic character-level tokenizer implemented from scratch.
Converts raw strings into discrete numerical token IDs (encode) and reconstructs
the original strings from token IDs (decode).

Zero third-party tokenizers, zero pretrained weights.
"""

from typing import List, Sequence, Union


class CharTokenizer:
    """Deterministic character-level tokenizer built from training text.

    Attributes:
        stoi (dict[str, int]): Mapping from character to integer token ID.
        itos (dict[int, str]): Mapping from integer token ID to character.
    """

    def __init__(self, text: str = None) -> None:
        """Initializes tokenizer and optionally builds vocabulary from text.

        Args:
            text: Optional raw text corpus used to construct vocabulary.
        """
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text: str) -> None:
        """Constructs deterministic vocabulary mappings (stoi and itos) from text.

        Args:
            text: Raw training text.

        Raises:
            TypeError: If input text is not a string.
            ValueError: If input text is empty.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Training text must be a string, got {type(text).__name__}."
            )
        if len(text) == 0:
            raise ValueError("Training text cannot be empty.")

        # Deterministically extract unique characters by sorting
        unique_chars = sorted(list(set(text)))

        # Construct bijective mapping tables
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        """Encodes an input string into a list of integer token IDs.

        Args:
            text: Input string to encode.

        Returns:
            List[int]: Sequence of integer token IDs.

        Raises:
            TypeError: If input text is not a string.
            ValueError: If vocabulary has not been built or contains an unknown character.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Input text must be a string, got {type(text).__name__}."
            )
        if not self.stoi:
            raise ValueError(
                "Tokenizer vocabulary is empty. Provide training text during initialization "
                "or call build_vocab(text) first."
            )

        encoded_ids: List[int] = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(
                    f"Unknown character '{ch}' (Unicode code point {ord(ch)}) "
                    f"not present in vocabulary."
                )
            encoded_ids.append(self.stoi[ch])

        return encoded_ids

    def decode(self, ids: Union[Sequence[int], List[int]]) -> str:
        """Decodes a sequence of integer token IDs back into the original string.

        Args:
            ids: Sequence of integer token IDs.

        Returns:
            str: Reconstructed text string.

        Raises:
            TypeError: If ids is not an iterable sequence of integers.
            ValueError: If an ID is out of range or not found in itos.
        """
        if not hasattr(ids, "__iter__") or isinstance(ids, (str, bytes)):
            raise TypeError(
                f"Input token IDs must be an iterable of integers, got {type(ids).__name__}."
            )

        decoded_chars: List[str] = []
        for token_id in ids:
            # Reject bools (which are technically subclasses of int in Python)
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(
                    f"Token ID must be an integer, got {type(token_id).__name__}: {token_id}."
                )
            if token_id not in self.itos:
                max_id = self.vocab_size - 1
                valid_range = f"0 to {max_id}" if max_id >= 0 else "empty"
                raise ValueError(
                    f"Invalid token ID {token_id} not found in vocabulary "
                    f"(valid range: {valid_range})."
                )
            decoded_chars.append(self.itos[token_id])

        return "".join(decoded_chars)


# Alias for backward compatibility and general usage
Tokenizer = CharTokenizer
