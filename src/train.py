"""Training Loop, Evaluation, and Optimization for MiniGPT.

Phase 5: Training Loop
Phase 6: Validation and Checkpoints
-----------------------------------
Implements the training and validation workflow from scratch:
1. Cross-entropy next-token loss computation from raw unnormalized logits.
2. AdamW optimizer updating randomly initialized Transformer weights.
3. Periodic train and validation loss estimation across eval_iters batches under torch.no_grad().
4. Clean gradient zeroing, backpropagation, and gradient norm clipping.
5. Explicit device placement (CUDA if available, else CPU).
6. Automatic alignment of model vocab_size with dataset/tokenizer vocabulary.
7. Atomic checkpoint saving (latest.pt and best.pt) and seamless resume support.

Zero pretrained weights, zero external LLMs.
"""

from copy import deepcopy
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MiniGPTConfig, config as default_config
from src.checkpoint import load_checkpoint, save_checkpoint
from src.dataset import TextDataset, get_dataset
from src.model import MiniGPT


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss for next-token prediction.

    Flattens spatial dimensions (B, T) to evaluate unnormalized logits against target indices.

    Args:
        logits: Unnormalized logits of shape (B, T, vocab_size).
        targets: Target token IDs of shape (B, T).

    Returns:
        torch.Tensor: Scalar cross-entropy loss.
    """
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    targets_flat = targets.view(B * T)
    return F.cross_entropy(logits_flat, targets_flat)


def estimate_loss(
    model: nn.Module,
    dataset: TextDataset,
    eval_iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    block_size: Optional[int] = None,
) -> Dict[str, float]:
    """Estimates average cross-entropy loss on train and validation splits.

    Switches model to eval mode and uses torch.no_grad() to ensure no gradients are tracked.
    Restores the model's previous training state before returning.

    Args:
        model: MiniGPT model to evaluate.
        dataset: Dataset providing train/val splits.
        eval_iters: Number of batches to sample per split. Defaults to config.eval_iters.
        batch_size: Batch size for evaluation. Defaults to model.config.batch_size.
        device: Device to run evaluation on. Defaults to model.config.device.
        block_size: Sequence context length. Defaults to model.config.block_size.

    Returns:
        Dict[str, float]: Dictionary with 'train' and 'val' average loss values.
    """
    iters = eval_iters if eval_iters is not None else default_config.eval_iters
    bsz = batch_size if batch_size is not None else getattr(model, "config", default_config).batch_size
    target_device = device if device is not None else getattr(model, "config", default_config).device
    blk_size = block_size if block_size is not None else getattr(model, "config", default_config).block_size

    was_training = model.training
    model.eval()

    out: Dict[str, float] = {}

    with torch.no_grad():
        for split in ("train", "val"):
            losses = []
            for _ in range(iters):
                x, y = dataset.get_batch(
                    split, batch_size=bsz, device=target_device, block_size=blk_size
                )
                logits = model(x)
                loss = compute_loss(logits, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses) if losses else 0.0

    model.train(was_training)
    return out


def train(
    model: Optional[MiniGPT] = None,
    dataset: Optional[TextDataset] = None,
    cfg: Optional[MiniGPTConfig] = None,
    max_iters: Optional[int] = None,
    eval_interval: Optional[int] = None,
    eval_iters: Optional[int] = None,
    learning_rate: Optional[float] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    grad_clip: Optional[float] = 1.0,
    verbose: bool = True,
    checkpoint_dir: str = "checkpoints",
    save_checkpoints: bool = True,
    save_interval: Optional[int] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes or resumes the MiniGPT pretraining loop from scratch.

    Args:
        model: Optional pre-constructed MiniGPT model.
        dataset: Optional TextDataset. Defaults to default corpus.
        cfg: Optional configuration override.
        max_iters: Total training iterations. Defaults to config.max_iters.
        eval_interval: Steps between evaluation logging. Defaults to config.eval_interval.
        eval_iters: Batches to sample during evaluation. Defaults to config.eval_iters.
        learning_rate: Optimizer learning rate. Defaults to config.learning_rate.
        device: Target execution device. Defaults to config.device.
        seed: Optional random seed for reproducible initialization and sampling.
        grad_clip: Maximum gradient norm for clipping. Defaults to 1.0. Set to None to disable.
        verbose: If True, prints periodic training logs to stdout.
        checkpoint_dir: Directory where checkpoints are stored.
        save_checkpoints: If True, periodically saves latest.pt and best.pt.
        save_interval: Frequency (steps) for saving latest checkpoint. Defaults to eval_interval.
        resume_from: Optional path to a checkpoint file to resume training from.

    Returns:
        Dict containing trained model, loss history, best validation loss, and final metrics.
    """
    # 1. Setup reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # 2. Setup dataset and determine actual vocabulary size
    ds = dataset if dataset is not None else get_dataset()
    actual_vocab_size = ds.vocab_size

    # 3. Setup configuration matching actual dataset vocabulary or resumed checkpoint
    resumed_ckpt = None
    if resume_from is not None:
        # Load raw metadata from checkpoint to configure model structure
        resumed_ckpt = load_checkpoint(resume_from, device=device or "cpu", validate_compatibility=False)
        start_step = int(resumed_ckpt.get("step", 0))
        best_val_loss = resumed_ckpt.get("best_val_loss", resumed_ckpt.get("val_loss", float("inf")))
        if best_val_loss is None:
            best_val_loss = float("inf")
        history = deepcopy(resumed_ckpt.get("history", {}))

        ckpt_cfg_dict = resumed_ckpt.get("config", {})
        if cfg is None and ckpt_cfg_dict:
            # Reconstruct configuration from checkpoint
            known_fields = {k: v for k, v in ckpt_cfg_dict.items() if hasattr(MiniGPTConfig, k)}
            run_cfg = MiniGPTConfig(**known_fields)
        elif cfg is not None:
            run_cfg = deepcopy(cfg)
        elif model is not None:
            run_cfg = deepcopy(model.config)
        else:
            run_cfg = deepcopy(default_config)
    else:
        start_step = 0
        best_val_loss = float("inf")
        history = {}
        if cfg is not None:
            run_cfg = deepcopy(cfg)
        elif model is not None:
            run_cfg = deepcopy(model.config)
        else:
            run_cfg = deepcopy(default_config)

    # Guarantee vocab_size matches the tokenizer
    run_cfg.vocab_size = actual_vocab_size

    # Resolve loop parameters
    total_iters = max_iters if max_iters is not None else run_cfg.max_iters
    eval_step = eval_interval if eval_interval is not None else run_cfg.eval_interval
    save_step = save_interval if save_interval is not None else eval_step
    eval_batches = eval_iters if eval_iters is not None else run_cfg.eval_iters
    lr = learning_rate if learning_rate is not None else run_cfg.learning_rate
    target_device = device if device is not None else run_cfg.device

    # 4. Instantiate model if not provided
    if model is None:
        m = MiniGPT(run_cfg)
    else:
        m = model
        # Check vocab consistency
        if m.config.vocab_size != actual_vocab_size:
            raise ValueError(
                f"Model vocab_size ({m.config.vocab_size}) does not match dataset vocab_size ({actual_vocab_size})."
            )

    m = m.to(target_device)
    m.train()

    # 5. Setup AdamW optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

    # If resuming, load model weights and optimizer state into instances
    if resume_from is not None:
        load_checkpoint(
            resume_from,
            model=m,
            optimizer=optimizer,
            device=target_device,
            validate_compatibility=True,
        )

    # Ensure checkpoint directory exists if saving
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    latest_ckpt_path: Optional[str] = None
    best_ckpt_path: Optional[str] = None

    if verbose:
        param_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print("=" * 60)
        print("MiniGPT Training Initialized" if resume_from is None else "MiniGPT Training Resumed")
        print(f"Device: {target_device} | Trainable Parameters: {param_count:,}")
        print(f"Vocab Size: {actual_vocab_size} | Block Size: {run_cfg.block_size} | Batch Size: {run_cfg.batch_size}")
        print(f"Start Step: {start_step} | Max Iters: {total_iters} | Eval Interval: {eval_step} | LR: {lr}")
        if resume_from is not None:
            print(f"Resumed from: {resume_from} (best val loss: {best_val_loss:.4f})")
        print("=" * 60)

    # 6. Check if already complete
    if start_step >= total_iters:
        if verbose:
            print(f"Start step ({start_step}) is already >= max_iters ({total_iters}). No steps to execute.")
        final_metrics = history.get(start_step, {})
        return {
            "model": m,
            "history": history,
            "final_train_loss": final_metrics.get("train", 0.0),
            "final_val_loss": final_metrics.get("val", 0.0),
            "best_val_loss": best_val_loss,
            "last_checkpoint": None,
            "best_checkpoint": None,
        }

    # 7. Main training loop
    start_time = time.time()

    for step in range(start_step, total_iters + 1):
        # Periodic evaluation (at start_step, every eval_step, or final step)
        if step == start_step or step % eval_step == 0 or step == total_iters:
            losses = estimate_loss(
                m,
                ds,
                eval_iters=eval_batches,
                batch_size=run_cfg.batch_size,
                device=target_device,
                block_size=run_cfg.block_size,
            )
            history[step] = losses
            train_loss = losses["train"]
            val_loss = losses["val"]

            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"step {step:5d} / {total_iters:5d} | "
                    f"train loss: {train_loss:.4f} | "
                    f"val loss: {val_loss:.4f} | "
                    f"elapsed: {elapsed:.2f}s"
                )

            # Best validation checkpoint tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_checkpoints:
                    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
                    save_checkpoint(
                        filepath=best_ckpt_path,
                        model=m,
                        optimizer=optimizer,
                        step=step,
                        config=run_cfg,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        best_val_loss=best_val_loss,
                        history=history,
                        dataset=ds,
                    )
                    if verbose:
                        print(f"       * New best validation loss: {val_loss:.4f} -> saved to {best_ckpt_path}")

            # Periodic latest checkpoint saving
            if save_checkpoints and (step % save_step == 0 or step == total_iters):
                latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
                save_checkpoint(
                    filepath=latest_ckpt_path,
                    model=m,
                    optimizer=optimizer,
                    step=step,
                    config=run_cfg,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    history=history,
                    dataset=ds,
                )

        if step == total_iters:
            break

        # Fetch training batch: x is context, y is shifted targets
        x, y = ds.get_batch(
            "train",
            batch_size=run_cfg.batch_size,
            device=target_device,
            block_size=run_cfg.block_size,
        )

        # Forward pass: compute logits
        logits = m(x)

        # Cross-entropy loss computation
        loss = compute_loss(logits, y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=grad_clip)

        optimizer.step()

    elapsed = time.time() - start_time
    total_steps_run = total_iters - start_step
    tokens_processed = total_steps_run * run_cfg.batch_size * run_cfg.block_size
    tokens_per_sec = tokens_processed / max(elapsed, 1e-6)

    final_metrics = history.get(total_iters, {})
    if verbose:
        print("=" * 60)
        print("Training Complete")
        print(f"Final Train Loss: {final_metrics.get('train', 0.0):.4f}")
        print(f"Final Val Loss:   {final_metrics.get('val', 0.0):.4f}")
        print(f"Best Val Loss:    {best_val_loss:.4f}")
        print(f"Duration:         {elapsed:.2f}s ({tokens_per_sec:.1f} tokens/s)")
        if latest_ckpt_path:
            print(f"Latest Checkpoint: {latest_ckpt_path}")
        if best_ckpt_path:
            print(f"Best Checkpoint:   {best_ckpt_path}")
        print("=" * 60)

    return {
        "model": m,
        "history": history,
        "final_train_loss": final_metrics.get("train", 0.0),
        "final_val_loss": final_metrics.get("val", 0.0),
        "best_val_loss": best_val_loss,
        "last_checkpoint": latest_ckpt_path,
        "best_checkpoint": best_ckpt_path,
        "total_duration_sec": elapsed,
        "tokens_processed": tokens_processed,
        "tokens_per_sec": tokens_per_sec,
    }


if __name__ == "__main__":
    train()
