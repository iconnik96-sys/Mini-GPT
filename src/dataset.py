"""Dataset and Batching module for MiniGPT.

Phase 3: Dataset and Batching
-----------------------------
Responsible for:
1. Loading raw text corpus from local files (data/).
2. Tokenizing corpus into token IDs using Phase 2 CharTokenizer.
3. Partitioning token IDs into deterministic train and validation splits (e.g. 90/10).
4. Generating training batches of input sequences (x) and next-token target sequences (y)
   where y is shifted by exactly 1 token relative to x:
       x[t] -> y[t] = x[t + 1]
5. Placing batch tensors on the configured hardware device (CUDA if available, else CPU).

No third-party datasets or pretrained data pipelines are used.
"""

import os
import random
from typing import Any, Dict, Optional, Tuple, Union

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    import numpy as np

import config
from src.tokenizer import CharTokenizer


class _FallbackTensor:
    """Lightweight tensor implementation used when PyTorch is not yet installed."""

    def __init__(self, data, device: str = "cpu", dtype: str = "int64") -> None:
        if isinstance(data, _FallbackTensor):
            self._arr = np.array(data._arr, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self._arr = data.astype(dtype)
        else:
            self._arr = np.array(data, dtype=dtype)
        self.device = str(device)
        self.dtype = self._arr.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._arr.shape

    def to(self, device: Union[str, any]) -> "_FallbackTensor":
        return _FallbackTensor(self._arr, device=str(device), dtype=self.dtype)

    def tolist(self):
        return self._arr.tolist()

    def __getitem__(self, idx):
        sub = self._arr[idx]
        if isinstance(sub, np.ndarray):
            return _FallbackTensor(sub, device=self.device, dtype=self.dtype)
        return sub

    def __len__(self) -> int:
        return len(self._arr)

    def __repr__(self) -> str:
        return f"tensor({self._arr.tolist()}, device='{self.device}')"

    def __eq__(self, other):
        if isinstance(other, _FallbackTensor):
            return _FallbackTensor(self._arr == other._arr, device=self.device, dtype="bool")
        return _FallbackTensor(self._arr == other, device=self.device, dtype="bool")


def _create_tensor(data, device: str = "cpu", dtype=None):
    """Creates a tensor using PyTorch if available, otherwise _FallbackTensor."""
    if _HAS_TORCH:
        target_dtype = dtype or torch.long
        t = torch.tensor(data, dtype=target_dtype)
        return t.to(device)
    else:
        return _FallbackTensor(data, device=device, dtype="int64")


class TextDataset:
    """Manages raw text loading, tokenization, train/val partitioning, and batch extraction.

    Attributes:
        tokenizer (CharTokenizer): Tokenizer built from corpus vocabulary.
        train_data: 1D sequence of token IDs for training split.
        val_data: 1D sequence of token IDs for validation split.
        block_size (int): Context window length.
        batch_size (int): Number of sequences per batch.
        device (str): Execution device for tensor placement ('cuda' or 'cpu').
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        raw_text: Optional[str] = None,
        block_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        train_split: float = 0.9,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes dataset from file or raw text.

        Args:
            file_path: Path to raw text file. If None and raw_text is None, defaults to data/input.txt.
            raw_text: Optional string containing corpus directly.
            block_size: Sequence context length. Defaults to config.block_size.
            batch_size: Sequences per batch. Defaults to config.batch_size.
            train_split: Fraction of tokens assigned to training (e.g. 0.9 for 90%).
            device: Hardware device ('cuda' or 'cpu'). Defaults to config.device.
            seed: Optional random seed for reproducible sampling.
        """
        self.block_size = block_size if block_size is not None else config.block_size
        self.batch_size = batch_size if batch_size is not None else config.batch_size
        self.device = device if device is not None else config.device
        self.train_split = train_split

        if seed is not None:
            self.set_seed(seed)

        # 1. Load raw text
        if raw_text is not None:
            self.raw_text = raw_text
        else:
            path = file_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "input.txt",
            )
            self.raw_text = self._load_file(path)

        if not isinstance(self.raw_text, str) or len(self.raw_text) == 0:
            raise ValueError("Corpus text cannot be empty.")

        # Clean/normalize line endings for consistent cross-platform tokens
        self.raw_text = self.raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Tokenize using Phase 2 CharTokenizer
        self.tokenizer = CharTokenizer(self.raw_text)
        token_ids = self.tokenizer.encode(self.raw_text)

        # 3. Partition train and validation splits
        if not (0.0 < self.train_split < 1.0):
            raise ValueError(
                f"train_split must be between 0.0 and 1.0 exclusive, got {self.train_split}."
            )

        split_idx = int(len(token_ids) * self.train_split)
        train_tokens = token_ids[:split_idx]
        val_tokens = token_ids[split_idx:]

        # Validate that both splits contain enough tokens for block_size + 1 (needed for x and y)
        min_required = self.block_size + 1
        if len(train_tokens) < min_required:
            raise ValueError(
                f"Training split length ({len(train_tokens)}) is too small for "
                f"block_size={self.block_size}. Minimum required is {min_required} tokens."
            )
        if len(val_tokens) < min_required:
            raise ValueError(
                f"Validation split length ({len(val_tokens)}) is too small for "
                f"block_size={self.block_size}. Minimum required is {min_required} tokens."
            )

        self.train_data = _create_tensor(train_tokens, device="cpu")
        self.val_data = _create_tensor(val_tokens, device="cpu")

    @staticmethod
    def _load_file(file_path: str) -> str:
        """Reads corpus from disk with clear error handling."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) == 0:
            raise ValueError(f"Dataset file at {file_path} is empty.")
        return content

    def set_seed(self, seed: int) -> None:
        """Sets random seed for reproducible batch generation."""
        random.seed(seed)
        if _HAS_TORCH:
            torch.manual_seed(seed)
        else:
            np.random.seed(seed)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size from underlying tokenizer."""
        return self.tokenizer.vocab_size

    def get_batch(
        self,
        split: str,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        block_size: Optional[int] = None,
    ):
        """Extracts a batch of (x, y) training pairs for the specified split.

        Args:
            split: 'train' or 'val'.
            batch_size: Optional override for batch size.
            device: Optional override for tensor device.
            block_size: Optional override for context length.

        Returns:
            x: Input tensor of shape (batch_size, block_size).
            y: Target tensor of shape (batch_size, block_size) shifted by 1 token.

        Raises:
            ValueError: If split is not 'train' or 'val'.
        """
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'val'.")

        data = self.train_data if split == "train" else self.val_data
        bsz = batch_size if batch_size is not None else self.batch_size
        target_device = device if device is not None else self.device
        blk_size = block_size if block_size is not None else self.block_size

        if blk_size >= len(data):
            raise ValueError(
                f"Requested block_size ({blk_size}) is too large for split '{split}' with length {len(data)}."
            )

        # Number of possible starting points: index range [0, len(data) - blk_size - 1]
        max_start = len(data) - blk_size

        if _HAS_TORCH:
            ix = torch.randint(0, max_start, (bsz,))
            x = torch.stack([data[i : i + blk_size] for i in ix])
            y = torch.stack([data[i + 1 : i + blk_size + 1] for i in ix])
            return x.to(target_device), y.to(target_device)
        else:
            # Fallback path using numpy / _FallbackTensor
            ix = [random.randint(0, max_start - 1) for _ in range(bsz)]
            x_list = [data[i : i + blk_size].tolist() for i in ix]
            y_list = [data[i + 1 : i + blk_size + 1].tolist() for i in ix]
            x_tensor = _FallbackTensor(x_list, device=target_device)
            y_tensor = _FallbackTensor(y_list, device=target_device)
            return x_tensor, y_tensor


# Global cached dataset instance for direct get_batch() function calls
_DEFAULT_DATASET: Optional[TextDataset] = None


def get_dataset() -> TextDataset:
    """Returns or lazily creates the default TextDataset instance."""
    global _DEFAULT_DATASET
    if _DEFAULT_DATASET is None:
        _DEFAULT_DATASET = TextDataset()
    return _DEFAULT_DATASET


def get_batch(split: str, batch_size: Optional[int] = None, device: Optional[str] = None):
    """Module-level batch extraction function adhering to project structure.

    Args:
        split: 'train' or 'val'.
        batch_size: Optional batch size override.
        device: Optional target device override.

    Returns:
        (x, y): Tensors of shape (batch_size, block_size) on target device.
    """
    return get_dataset().get_batch(split=split, batch_size=batch_size, device=device)


_PROGRAMMING_DATASET: Optional[TextDataset] = None


def get_programming_dataset(
    file_path: Optional[str] = None,
    block_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> TextDataset:
    """Returns or lazily creates the programming-domain TextDataset instance.

    Defaults to loading from data/programming.txt.
    """
    global _PROGRAMMING_DATASET
    if file_path is not None or _PROGRAMMING_DATASET is None:
        target_path = file_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "programming.txt",
        )
        ds = TextDataset(
            file_path=target_path,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        if file_path is None:
            _PROGRAMMING_DATASET = ds
        return ds
    return _PROGRAMMING_DATASET


def get_dataset_statistics(
    file_path: Optional[str] = None,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """Computes comprehensive statistics for a text corpus.

    Args:
        file_path: Path to dataset file.
        text: Optional raw text string.

    Returns:
        Dict[str, Any] containing character counts, lines, vocabulary, token splits,
        and top character frequencies.
    """
    from collections import Counter

    if text is not None:
        content = text
    else:
        path = file_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "programming.txt",
        )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

    # Normalize newlines
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    chars = sorted(list(set(content)))
    total_chars = len(content)
    lines = content.splitlines()

    # Identify section count if annotated with '// Section'
    sections = [line for line in lines if line.strip().startswith("// Section")]

    train_tokens = int(total_chars * 0.9)
    val_tokens = total_chars - train_tokens

    char_counts = Counter(content)
    top_chars = [
        {
            "char": repr(ch) if ch in ("\n", "\t", " ") else ch,
            "count": count,
            "percentage": round((count / total_chars) * 100, 2),
        }
        for ch, count in char_counts.most_common(15)
    ]

    return {
        "total_characters": total_chars,
        "total_lines": len(lines),
        "num_sections": len(sections),
        "section_titles": sections,
        "vocab_size": len(chars),
        "vocabulary": "".join(chars),
        "train_split": 0.9,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "top_characters": top_chars,
        "source": "Educational Open-Source Reference Architecture & Backend Guides",
        "license": "CC0 1.0 Universal / Public Domain (Educational Use)",
    }
