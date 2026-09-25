"""
Model Registry for Phase 12 MiniGPT CLI.

Provides catalog of available models and static metadata without loading
heavy neural network weights into memory.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelMetadata:
    id: str
    display_name: str
    model_type: str  # "from_scratch", "causal_lm", "peft_lora"
    checkpoint_path: str
    tokenizer_type: str  # "character", "bpe"
    parameter_count: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percentage: float
    context_length: int
    description: str
    base_model_name: Optional[str] = None


# Static registry catalog (No models loaded on import)
REGISTRY: Dict[str, ModelMetadata] = {
    "distilgpt2-lora-v2": ModelMetadata(
        id="distilgpt2-lora-v2",
        display_name="DistilGPT-2 + LoRA v2 (Improved)",
        model_type="peft_lora",
        checkpoint_path="phase10_lora/checkpoints/distilgpt2_lora_programming_v2",
        tokenizer_type="bpe",
        parameter_count=82_060_032,
        trainable_parameters=147_456,
        frozen_parameters=81_912_576,
        trainable_percentage=0.1797,
        context_length=1024,
        description="Improved DistilGPT-2 fine-tuned with LoRA on 94 diverse Java, Spring, REST, and SQL instructions.",
        base_model_name="distilbert/distilgpt2"
    ),
    "distilgpt2-lora": ModelMetadata(
        id="distilgpt2-lora",
        display_name="DistilGPT-2 + LoRA (Phase 10 v1)",
        model_type="peft_lora",
        checkpoint_path="phase10_lora/checkpoints/distilgpt2_lora_programming",
        tokenizer_type="bpe",
        parameter_count=82_060_032,
        trainable_parameters=147_456,
        frozen_parameters=81_912_576,
        trainable_percentage=0.1797,
        context_length=1024,
        description="Pretrained DistilGPT-2 fine-tuned with LoRA (r=8, alpha=16) on Java & Spring Boot instructions (v1 baseline).",
        base_model_name="distilbert/distilgpt2"
    ),
    "minigpt-programming": ModelMetadata(
        id="minigpt-programming",
        display_name="MiniGPT Programming (Phase 9)",
        model_type="from_scratch",
        checkpoint_path="checkpoints/phase9_longer/best.pt",
        tokenizer_type="character",
        parameter_count=114_944,
        trainable_parameters=114_944,
        frozen_parameters=0,
        trainable_percentage=100.0,
        context_length=64,
        description="From-scratch decoder-only Transformer (2L, 4H, 64D) trained on Java, Spring Boot, and SQL."
    ),
    "distilgpt2": ModelMetadata(
        id="distilgpt2",
        display_name="DistilGPT-2 Base (Pretrained)",
        model_type="causal_lm",
        checkpoint_path="distilbert/distilgpt2",
        tokenizer_type="bpe",
        parameter_count=81_912_576,
        trainable_parameters=0,
        frozen_parameters=81_912_576,
        trainable_percentage=0.0,
        context_length=1024,
        description="Pretrained 82M causal language model by Hugging Face on WebText without task adaptation."
    ),
    "minigpt-baseline": ModelMetadata(
        id="minigpt-baseline",
        display_name="MiniGPT Baseline (Phase 8)",
        model_type="from_scratch",
        checkpoint_path="checkpoints/exp1_baseline/best.pt",
        tokenizer_type="character",
        parameter_count=110_336,
        trainable_parameters=110_336,
        frozen_parameters=0,
        trainable_percentage=100.0,
        context_length=64,
        description="From-scratch decoder-only Transformer (2L, 4H, 64D) trained on Shakespeare (Coriolanus)."
    )
}

# User-friendly case-insensitive aliases
ALIASES: Dict[str, str] = {
    "distilgpt2-lora-v2": "distilgpt2-lora-v2",
    "distilgpt2_lora_v2": "distilgpt2-lora-v2",
    "lora-v2": "distilgpt2-lora-v2",
    "v2": "distilgpt2-lora-v2",
    "distilgpt2-lora": "distilgpt2-lora",
    "distilgpt2_lora": "distilgpt2-lora",
    "distilgpt-2 + lora": "distilgpt2-lora-v2",
    "distilgpt-2+lora": "distilgpt2-lora-v2",
    "minigpt-programming": "minigpt-programming",
    "minigpt_programming": "minigpt-programming",
    "programming": "minigpt-programming",
    "distilgpt2": "distilgpt2",
    "minigpt-baseline": "minigpt-baseline",
    "minigpt_baseline": "minigpt-baseline",
    "baseline": "minigpt-baseline",
}


def resolve_model_id(model_id: str) -> str:
    """Resolve aliases or case variations to canonical model ID."""
    cleaned = model_id.strip().lower()
    return ALIASES.get(cleaned, model_id)


def list_models() -> List[ModelMetadata]:
    """Return list of all registered models."""
    return list(REGISTRY.values())


def is_valid_model(model_id: str) -> bool:
    """Check if model ID is registered."""
    resolved = resolve_model_id(model_id)
    return resolved in REGISTRY


def get_model_info(model_id: str) -> ModelMetadata:
    """Retrieve metadata for a registered model, or raise ValueError."""
    resolved = resolve_model_id(model_id)
    if resolved not in REGISTRY:
        valid_ids = ", ".join(f"'{k}'" for k in REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_id}'. Available models: {valid_ids}")
    return REGISTRY[resolved]

