"""Inference and Autoregressive Text Generation for MiniGPT.

Phase 7: Text Generation
-------------------------
Implements manual autoregressive text generation from scratch:
1. Encodes a user prompt into token IDs using CharTokenizer.
2. Context window management: crops running sequence to block_size if length exceeds context window.
3. Model evaluation under torch.no_grad() without gradient computation or weight modification.
4. Final position logit extraction.
5. Deterministic greedy decoding via torch.argmax.
6. Stochastic sampling with temperature scaling (sharper vs flatter distributions).
7. Top-k filtering to restrict candidate token pool.
8. Decodes output token sequence back to text string.
9. Checkpoint restoration: loads architecture, weights, and tokenizer vocabulary directly from .pt files.
10. CLI entry point for command-line inference.

Zero pretrained models, zero high-level generation frameworks.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MiniGPTConfig
from src.checkpoint import load_checkpoint
from src.model import MiniGPT
from src.tokenizer import CharTokenizer


def generate(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    greedy: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
) -> str:
    """Generates text autoregressively starting from a prompt string.

    Args:
        model: MiniGPT language model.
        tokenizer: Character-level tokenizer with vocabulary matching the model.
        prompt: Initial string to condition generation on.
        max_new_tokens: Number of tokens to generate beyond the prompt.
        temperature: Logit scaling factor (>0). Lower values make distribution sharper.
        top_k: Optional positive integer to filter distribution to top-k highest logits.
        greedy: If True, uses deterministic argmax decoding instead of probabilistic sampling.
        device: Device to run inference on. Defaults to model's current device or CPU.
        seed: Optional random seed for reproducible sampling.

    Returns:
        str: The full generated text string (prompt + generated tokens).

    Raises:
        ValueError: If max_new_tokens < 0, temperature <= 0 (when greedy=False),
                    top_k is invalid, or prompt contains characters not in tokenizer vocabulary.
    """
    # 1. Parameter validation
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")

    if not greedy and temperature <= 0.0:
        raise ValueError(
            f"Temperature must be strictly positive (> 0), got {temperature}. "
            f"For deterministic generation, set greedy=True."
        )

    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer or None, got {top_k}.")

    # 2. Setup reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # 3. Resolve target device
    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 4. Resolve context length (block_size)
    block_size = getattr(getattr(model, "config", None), "block_size", 256)

    # 5. Encode prompt into token IDs
    if prompt:
        try:
            prompt_ids = tokenizer.encode(prompt)
        except ValueError as err:
            raise ValueError(f"Prompt validation failed: {err}") from err
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=target_device)
    else:
        # If prompt is empty and no tokens requested, return empty string
        if max_new_tokens == 0:
            return ""
        # Default starting token (token ID 0)
        idx = torch.zeros((1, 1), dtype=torch.long, device=target_device)

    # Fast return if no tokens requested
    if max_new_tokens == 0:
        return prompt

    # 6. Set inference mode and preserve training state
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context to model's maximum block_size if sequence is longer
                idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

                # Forward pass: obtain unnormalized logits (shape: (1, T_cond, vocab_size))
                logits = model(idx_cond)

                # Pluck the logits at the final position (shape: (1, vocab_size))
                logits = logits[:, -1, :]

                if greedy:
                    # Deterministic greedy selection: pick highest logit
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    # Apply temperature scaling
                    logits = logits / temperature

                    # Optional Top-k filtering: set all logits below k-th highest to -infinity
                    if top_k is not None:
                        vocab_size = logits.size(-1)
                        k = min(top_k, vocab_size)
                        v, _ = torch.topk(logits, k)
                        logits[logits < v[:, [-1]]] = -float("Inf")

                    # Convert logits to probabilities
                    probs = F.softmax(logits, dim=-1)

                    # Sample next token from categorical distribution
                    idx_next = torch.multinomial(probs, num_samples=1)

                # Append sampled token to running sequence
                idx = torch.cat((idx, idx_next), dim=1)

        # 7. Decode complete token sequence back to text string
        generated_ids = idx[0].tolist()
        return tokenizer.decode(generated_ids)

    finally:
        # Restore model's prior training mode
        model.train(was_training)


# Alias for backward compatibility
generate_text = generate


def load_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[MiniGPT, CharTokenizer]:
    """Loads and instantiates a MiniGPT model and CharTokenizer from a checkpoint.

    Extracts configuration parameters and tokenizer vocabulary from checkpoint metadata,
    builds the model architecture, and restores trained weights.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to place model on. Defaults to CUDA if available, else CPU.

    Returns:
        Tuple[MiniGPT, CharTokenizer]: Ready-to-use model and tokenizer.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format is invalid.
    """
    target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint_path, device=target_device, validate_compatibility=False)

    # 1. Reconstruct MiniGPTConfig from checkpoint metadata
    ckpt_cfg = ckpt.get("config", {})
    known_fields = {k: v for k, v in ckpt_cfg.items() if hasattr(MiniGPTConfig, k)}
    cfg = MiniGPTConfig(**known_fields)
    cfg.device = str(target_device)

    # 2. Instantiate and restore model
    model = MiniGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(target_device)
    model.eval()

    # 3. Reconstruct CharTokenizer from checkpoint metadata
    tokenizer = CharTokenizer()
    tok_vocab = ckpt.get("tokenizer_vocab")
    if tok_vocab and "stoi" in tok_vocab and "itos" in tok_vocab:
        tokenizer.stoi = dict(tok_vocab["stoi"])
        # Ensure integer keys for itos mapping
        tokenizer.itos = {int(k): v for k, v in tok_vocab["itos"].items()}
    else:
        # Fallback to corpus dataset tokenizer if vocabulary metadata was not serialized
        from src.dataset import get_dataset
        ds = get_dataset()
        tokenizer = ds.tokenizer

    return model, tokenizer


def generate_from_checkpoint(
    checkpoint_path: Union[str, Path],
    prompt: str = "The ",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    greedy: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
) -> str:
    """Convenience helper to load a checkpoint and generate text in a single call.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        prompt: Initial prompt text.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (>0).
        top_k: Optional top-k threshold.
        greedy: If True, uses greedy argmax decoding.
        device: Device to run generation on.
        seed: Optional random seed for reproducible sampling.

    Returns:
        str: Generated text string.
    """
    model, tokenizer = load_from_checkpoint(checkpoint_path, device=device)
    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
        device=device,
        seed=seed,
    )


def main() -> None:
    """CLI entry point for command-line text generation."""
    parser = argparse.ArgumentParser(description="MiniGPT Autoregressive Text Generation CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The ",
        help="Initial prompt string to condition generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (>0)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k filtering threshold (positive integer or omit)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic greedy decoding (argmax)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cpu' or 'cuda')",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MiniGPT Text Generation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Max New Tokens: {args.max_new_tokens} | Temp: {args.temperature} | Top-k: {args.top_k} | Greedy: {args.greedy}")
    print("=" * 60)

    try:
        output_text = generate_from_checkpoint(
            checkpoint_path=args.checkpoint,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            device=args.device,
            seed=args.seed,
        )
        print("\n--- Generated Output ---\n")
        print(output_text)
        print("\n------------------------")
    except Exception as exc:
        print(f"Error during generation: {exc}")


if __name__ == "__main__":
    main()
