"""MiniGPT Configuration Module.

Phase 1: Project Foundation
Hyperparameters configured for local development and experimentation on an
NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM), with CPU fallback.

NOTE: Do not modify to load pretrained models. All architectures will be
trained from randomly initialized weights in future phases.
"""

from dataclasses import dataclass


@dataclass
class MiniGPTConfig:
    """Configuration class holding model architecture and training hyperparameters."""

    # -------------------------------------------------------------------------
    # Batch and Sequence Dimensions
    # -------------------------------------------------------------------------
    batch_size: int = 32
    """Number of independent sequences processed in parallel per forward/backward pass.
    32 fits comfortably within 6GB VRAM while providing stable gradient estimates."""

    block_size: int = 256
    """Maximum context length (time dimension) for predictions.
    256 tokens captures adequate context for character- or subword-level experiments."""

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    vocab_size: int = 65
    """Vocabulary size. Default placeholder is 65 (typical for character-level TinyShakespeare).
    This value will be dynamically set or confirmed by the tokenizer in Phase 2."""

    n_embd: int = 384
    """Embedding dimension (hidden layer size).
    Must be divisible by n_head (384 / 6 = 64 dimensions per attention head)."""

    n_head: int = 6
    """Number of parallel attention heads in Multi-Head Causal Self-Attention."""

    n_layer: int = 6
    """Number of Transformer blocks (layers) stacked sequentially."""

    dropout: float = 0.2
    """Dropout probability for regularization (applied to embeddings, attention, and residual connections)."""

    # -------------------------------------------------------------------------
    # Optimization and Training Loop
    # -------------------------------------------------------------------------
    learning_rate: float = 3e-4
    """Peak learning rate for the AdamW optimizer."""

    max_iters: int = 5000
    """Total number of training iterations / steps."""

    eval_interval: int = 500
    """Frequency (in iterations) at which validation loss is computed and logged."""

    eval_iters: int = 200
    """Number of batches to sample and average across when evaluating train/val loss."""

    # -------------------------------------------------------------------------
    # System and Hardware Execution
    # -------------------------------------------------------------------------
    try:
        import torch
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        # Fallback before torch is installed in the virtual environment
        device: str = "cuda"
    """Execution device: automatically selects CUDA if an NVIDIA GPU is detected, else CPU."""


# Default instance for quick access across scripts
config = MiniGPTConfig()

# Top-level hyperparameter bindings for direct script access
batch_size: int = config.batch_size
block_size: int = config.block_size
vocab_size: int = config.vocab_size
n_embd: int = config.n_embd
n_head: int = config.n_head
n_layer: int = config.n_layer
dropout: float = config.dropout
learning_rate: float = config.learning_rate
max_iters: int = config.max_iters
eval_interval: int = config.eval_interval
eval_iters: int = config.eval_iters
device: str = config.device
