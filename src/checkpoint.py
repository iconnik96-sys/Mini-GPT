"""Checkpoint Management for MiniGPT.

Phase 6: Validation and Checkpoints
-----------------------------------
Implements robust, atomic checkpoint saving and loading for MiniGPT:
1. Complete state preservation: model weights, optimizer state, training step,
   hyperparameters, loss metrics, loss history, tokenizer vocabulary, dataset info, and RNG state.
2. Atomic saving: writes to a temporary file before renaming to prevent corrupted files on interruption.
3. Strict safety checks: file existence, integrity verification, vocabulary alignment,
   architecture compatibility, and NaN/Inf validation.
4. Seamless device mapping: transparent CPU/CUDA fallback via map_location.
5. Best-model checkpoint tracking based on validation loss.

Zero pretrained weights, zero external LLMs.
"""

from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from config import MiniGPTConfig


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    config: Optional[Union[MiniGPTConfig, Dict[str, Any]]] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    history: Optional[Dict[int, Dict[str, float]]] = None,
    tokenizer: Optional[Any] = None,
    dataset: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Atomically saves a complete training checkpoint to disk.

    Writes to a temporary file in the same directory first, then atomically renames
    it via os.replace to guarantee atomic persistence and avoid corrupted files
    if saving is interrupted.

    Args:
        filepath: Target destination path for the checkpoint (e.g. 'checkpoints/latest.pt').
        model: MiniGPT model whose state_dict is saved.
        optimizer: Optional optimizer whose state_dict is saved.
        step: Current training iteration / step.
        config: Model configuration (MiniGPTConfig or dict). Defaults to model.config if present.
        train_loss: Current training loss.
        val_loss: Current validation loss.
        best_val_loss: Best validation loss observed so far.
        history: Dictionary mapping step -> loss metrics.
        tokenizer: Optional tokenizer to serialize vocabulary mappings (stoi/itos/vocab_size).
        dataset: Optional TextDataset to extract dataset info and tokenizer vocab.
        extra_metadata: Optional arbitrary dictionary for user metadata.

    Returns:
        str: Absolute path of the saved checkpoint.
    """
    target_path = Path(filepath).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve configuration metadata
    cfg_obj = config if config is not None else getattr(model, "config", None)
    if is_dataclass(cfg_obj):
        cfg_dict = asdict(cfg_obj)
    elif isinstance(cfg_obj, dict):
        cfg_dict = dict(cfg_obj)
    else:
        cfg_dict = {}

    # 2. Resolve tokenizer vocabulary metadata
    tok_obj = tokenizer
    if tok_obj is None and dataset is not None and hasattr(dataset, "tokenizer"):
        tok_obj = dataset.tokenizer

    tokenizer_vocab = None
    if tok_obj is not None:
        tokenizer_vocab = {
            "vocab_size": getattr(tok_obj, "vocab_size", None),
            "stoi": getattr(tok_obj, "stoi", {}),
            "itos": getattr(tok_obj, "itos", {}),
        }

    # 3. Resolve dataset info
    dataset_info = None
    if dataset is not None:
        dataset_info = {
            "train_len": len(getattr(dataset, "train_data", [])),
            "val_len": len(getattr(dataset, "val_data", [])),
            "vocab_size": getattr(dataset, "vocab_size", None),
        }

    # 4. Capture RNG state for reproducibility
    rng_state = {
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass

    # 5. Build full checkpoint payload
    checkpoint_payload: Dict[str, Any] = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg_dict,
        "train_loss": float(train_loss) if train_loss is not None else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "history": history or {},
        "tokenizer_vocab": tokenizer_vocab,
        "dataset_info": dataset_info,
        "rng_state": rng_state,
        "extra_metadata": extra_metadata or {},
        "version": "1.0",
    }

    # 6. Atomic save via temporary file
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    try:
        with open(temp_path, "wb") as f:
            torch.save(checkpoint_payload, f)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return str(target_path)


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    validate_compatibility: bool = True,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    """Loads and validates a training checkpoint from disk.

    Supports CPU and CUDA mapping, parameter compatibility checks, and weight restoration.

    Args:
        filepath: Path to the checkpoint file to load.
        model: Optional model instance into which weights are loaded.
        optimizer: Optional optimizer into which state is loaded.
        device: Device to map tensors to. Defaults to CUDA if available, else CPU.
        strict: Whether to strictly enforce key matching in model.load_state_dict.
        validate_compatibility: If True, checks configuration and vocabulary compatibility with model.
        restore_rng: If True, restores saved PyTorch RNG states.

    Returns:
        Dict[str, Any]: The loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint file is corrupted or cannot be unpickled.
        ValueError: If checkpoint format is invalid, weights contain NaNs, or configuration is incompatible.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: '{filepath}'")

    # 1. Resolve map_location
    if device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = str(device)

    # 2. Load serialized checkpoint payload
    try:
        with open(path, "rb") as f:
            checkpoint = torch.load(f, map_location=target_device)
    except Exception as exc:
        raise RuntimeError(f"Corrupted or invalid checkpoint file at '{filepath}': {exc}") from exc

    # 3. Validate root payload schema
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Invalid checkpoint format at '{filepath}'. Expected dict, got {type(checkpoint).__name__}."
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format at '{filepath}'. Missing required key 'model_state_dict'."
        )

    # 4. Check for corrupted NaN/Inf values in weights
    state_dict = checkpoint["model_state_dict"]
    for param_name, tensor in state_dict.items():
        if torch.is_tensor(tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(
                    f"Corrupted weights detected in checkpoint '{filepath}': parameter '{param_name}' contains NaN or Inf."
                )

    # 5. Validate compatibility with target model if provided
    if model is not None and validate_compatibility:
        ckpt_cfg = checkpoint.get("config", {})
        model_cfg = getattr(model, "config", None)

        if model_cfg is not None and ckpt_cfg:
            # Check vocabulary size
            ckpt_vocab = ckpt_cfg.get("vocab_size")
            model_vocab = getattr(model_cfg, "vocab_size", None)
            if ckpt_vocab is not None and model_vocab is not None and ckpt_vocab != model_vocab:
                raise ValueError(
                    f"Incompatible vocabulary size: checkpoint has vocab_size={ckpt_vocab}, "
                    f"but target model has vocab_size={model_vocab}."
                )

            # Check core architecture dimensions
            for dim_name in ("n_embd", "n_head", "n_layer", "block_size"):
                ckpt_dim = ckpt_cfg.get(dim_name)
                model_dim = getattr(model_cfg, dim_name, None)
                if ckpt_dim is not None and model_dim is not None and ckpt_dim != model_dim:
                    raise ValueError(
                        f"Incompatible architecture parameter '{dim_name}': checkpoint has {ckpt_dim}, "
                        f"but target model has {model_dim}."
                    )

    # 6. Restore model state_dict
    if model is not None:
        try:
            model.load_state_dict(state_dict, strict=strict)
        except Exception as exc:
            raise ValueError(f"Failed to load model state_dict from '{filepath}': {exc}") from exc

    # 7. Restore optimizer state_dict
    if optimizer is not None:
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as exc:
                raise ValueError(f"Failed to load optimizer state_dict from '{filepath}': {exc}") from exc

    # 8. Optionally restore RNG state
    if restore_rng and "rng_state" in checkpoint:
        rng_state = checkpoint["rng_state"]
        if "torch" in rng_state and rng_state["torch"] is not None:
            try:
                torch.set_rng_state(rng_state["torch"])
            except Exception:
                pass

    return checkpoint
