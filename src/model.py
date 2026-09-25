"""Transformer Model Architecture for MiniGPT.

Phase 4: Transformer Architecture
---------------------------------
Implements a GPT-style decoder-only Transformer from scratch using PyTorch:
- Token and learnable positional embeddings
- Causal multi-head self-attention with lower-triangular causal masking
- Position-wise feed-forward networks (MLP) with GELU activation
- Pre-LayerNorm residual connections
- Final LayerNorm and language-model projection head (logits)
- Random weight initialization from scratch (zero pretrained weights)

Adheres strictly to the architectural constraints:
- Zero pretrained models or weights (GPT-2, LLaMA, etc.)
- All tensor operations built from scratch using torch.nn modules
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MiniGPTConfig, config as default_config


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with lower-triangular attention mask.

    Ensures tokens can only attend to prior tokens in the sequence (autoregressive property).
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Key, Query, Value linear projections for all heads
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Regularization dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of ones
        # Registered as buffer so it moves with model device without being a trainable parameter
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for multi-head causal self-attention.

        Args:
            x: Input tensor of shape (B, T, C).
            return_attention: If True, returns attention probability weights.

        Returns:
            Output tensor of shape (B, T, C), optionally with attention weights (B, n_head, T, T).
        """
        B, T, C = x.size()

        # Project and reshape: (B, T, C) -> (B, n_head, T, head_size)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention: Q @ K.T / sqrt(head_size)
        # Shape: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # Apply causal mask: mask future positions with -inf so attention probability is 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over key sequence length
        att_weights = F.softmax(att, dim=-1)
        att_dropped = self.attn_dropout(att_weights)

        # Aggregate values: (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        y = att_dropped @ v

        # Re-assemble multi-head outputs: (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection with residual dropout
        out = self.resid_dropout(self.out_proj(y))

        if return_attention:
            return out, att_weights
        return out


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (MLP) with GELU activation."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies position-wise linear transformations and non-linearity.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Transformed tensor of shape (B, T, C).
        """
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-LayerNorm residual connections."""

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pre-LayerNorm forward pass with residual skip connections.

        Args:
            x: Input tensor of shape (B, T, C).
            return_attention: If True, also returns attention weights from CausalSelfAttention.

        Returns:
            Residual output of shape (B, T, C).
        """
        if return_attention:
            attn_out, att_weights = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, att_weights
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x


class MiniGPT(nn.Module):
    """GPT-style decoder-only Transformer language model built from scratch.

    Transforms sequences of discrete token IDs into next-token logits over the vocabulary.
    All weights are initialized randomly from scratch.
    """

    def __init__(self, config: Optional[MiniGPTConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else default_config

        assert (
            self.config.n_embd % self.config.n_head == 0
        ), f"n_embd ({self.config.n_embd}) must be divisible by n_head ({self.config.n_head})"

        # Token embedding and learnable positional embedding tables
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_emb = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.dropout)

        # Stack of Transformer decoder blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layer)]
        )

        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(self.config.n_embd)

        # Language modeling head: linear projection from n_embd to vocab_size (logits)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Initialize all model weights from scratch
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Custom Gaussian weight initialization (mean 0.0, std 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass transforming token IDs to unnormalized logits.

        Args:
            idx: Tensor of token IDs of shape (B, T).
            targets: Optional ground-truth next-token targets of shape (B, T).
            return_attention: If True, returns list of attention weight matrices per layer.

        Returns:
            logits: Next-token logits of shape (B, T, vocab_size).
            loss (optional): Cross-entropy loss scalar when targets are provided.
        """
        B, T = idx.size()

        # Reject sequences exceeding maximum context length
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}; exceeds maximum block_size ({self.config.block_size})."
            )

        # Positional indices [0, 1, ..., T - 1]
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Lookup embeddings and add positional information
        tok_emb = self.tok_emb(idx)  # (B, T, n_embd)
        pos_emb = self.pos_emb(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # Pass through sequential Transformer blocks
        attention_maps: List[torch.Tensor] = []
        for block in self.blocks:
            if return_attention:
                x, att_w = block(x, return_attention=True)
                attention_maps.append(att_w)
            else:
                x = block(x)

        # Apply final LayerNorm
        x = self.ln_f(x)

        # Project representations to vocabulary logits: (B, T, vocab_size)
        logits = self.lm_head(x)

        # Compute optional cross-entropy loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        if return_attention:
            return logits, attention_maps

        return logits
