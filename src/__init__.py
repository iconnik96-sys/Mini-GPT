"""MiniGPT source package.

Phase 1: Project Foundation.
Contains modular components for the scratch-built GPT implementation:
- tokenizer: Encoding and decoding text to token IDs
- dataset: Loading data, splitting train/val, and generating batches
- model: GPT-style Transformer architecture (Embeddings, Attention, Blocks)
- train: Training loop, loss computation, AdamW optimizer, checkpointing
- generate: Autoregressive text generation using trained weights
"""

__version__ = "0.1.0"
