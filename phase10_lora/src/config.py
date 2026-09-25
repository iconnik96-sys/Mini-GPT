"""
Configuration for Phase 10: Pretrained LLM LoRA Fine-Tuning.

Encapsulates all hyperparameters for:
- Pretrained base model selection
- LoRA adapter configuration (rank, alpha, dropout, target modules)
- SFT dataset paths and tokenization limits
- Training hyperparameters (batch size, learning rate, epochs, seed)
- Checkpoint and experiment tracking paths
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Phase10Config:
    # Model
    model_name: str = "distilbert/distilgpt2"
    device: str = "cpu"
    
    # LoRA hyperparameters
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["c_attn"])
    fan_in_fan_out: bool = True
    bias: str = "none"
    
    # Dataset & Training
    dataset_path: str = os.path.join("phase10_lora", "data", "programming_instructions.json")
    train_val_split: float = 0.8
    seed: int = 42
    max_seq_len: int = 256
    
    # Optimization
    batch_size: int = 4
    learning_rate: float = 5e-4
    num_train_epochs: int = 5
    weight_decay: float = 0.01
    
    # Storage
    checkpoint_dir: str = os.path.join("phase10_lora", "checkpoints")
    experiment_dir: str = os.path.join("phase10_lora", "experiments")
    adapter_checkpoint_name: str = "distilgpt2_lora_programming"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase10Config":
        return cls(**data)

    @classmethod
    def get_v2_config(cls) -> "Phase10Config":
        """Return optimized configuration for Phase 10 v2 LoRA model."""
        return cls(
            dataset_path=os.path.join("phase10_lora", "data", "programming_instructions_v2.json"),
            adapter_checkpoint_name="distilgpt2_lora_programming_v2",
            learning_rate=3e-4,
            num_train_epochs=6,
            batch_size=4,
            seed=42
        )
