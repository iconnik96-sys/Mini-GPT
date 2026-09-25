"""
Lazy Model Loader for Phase 12 CLI.

Ensures models are only loaded on-demand when generation or detailed inspection
is explicitly requested. Maintains an active model cache for interactive chat sessions.
"""

import os
from typing import Tuple, Any, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from phase12_cli.src.model_registry import get_model_info, ModelMetadata
from src.generate import load_from_checkpoint as load_minigpt_from_checkpoint
from phase10_lora.src.train import load_fine_tuned_model as load_lora_from_checkpoint


class ModelSession:
    """Manages currently loaded model and tokenizer in memory to avoid redundant reloads."""
    def __init__(self):
        self.current_model_id: Optional[str] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.metadata: Optional[ModelMetadata] = None
        self.device: str = "cpu"

    def get_or_load(
        self,
        model_id: str,
        device: str = "cpu"
    ) -> Tuple[Any, Any, ModelMetadata]:
        """Load requested model if not already cached in memory."""
        if self.current_model_id == model_id and self.model is not None:
            return self.model, self.tokenizer, self.metadata

        meta = get_model_info(model_id)

        # Checkpoint validation
        if meta.model_type in ["from_scratch", "peft_lora"]:
            if not os.path.exists(meta.checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found for model '{model_id}' at path: {meta.checkpoint_path}\n"
                    f"Please ensure prior phase training completed successfully."
                )

        if meta.model_type == "from_scratch":
            model, tokenizer = load_minigpt_from_checkpoint(meta.checkpoint_path, device=device)
            model.eval()

        elif meta.model_type == "causal_lm":
            tokenizer = AutoTokenizer.from_pretrained(meta.checkpoint_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(meta.checkpoint_path).to(device)
            model.eval()

        elif meta.model_type == "peft_lora":
            model, tokenizer = load_lora_from_checkpoint(
                checkpoint_dir=meta.checkpoint_path,
                base_model_name=meta.base_model_name or "distilbert/distilgpt2",
                device=device
            )
            model.eval()

        else:
            raise ValueError(f"Unsupported model type: {meta.model_type}")

        self.current_model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = meta
        self.device = device

        return model, tokenizer, meta

    def clear(self) -> None:
        """Release currently loaded model from memory."""
        self.current_model_id = None
        self.model = None
        self.tokenizer = None
        self.metadata = None
