"""
Configuration for Phase 11: Cross-System Evaluation and Comparison.

Defines:
- Four target systems and checkpoint paths
- Output artifact directories
- Deterministic and sampling generation parameters
- Benchmark prompt paths
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class ModelTargetConfig:
    system_id: str
    display_name: str
    category: str  # "from_scratch", "domain_scratch", "pretrained_base", "peft_lora"
    checkpoint_path: str
    tokenizer_type: str  # "character", "bpe"
    base_model_name: str = ""  # For LoRA adapters
    is_peft: bool = False


@dataclass
class Phase11Config:
    # Evaluation prompt file
    prompts_path: str = os.path.join("phase11_evaluation", "prompts", "evaluation_prompts.json")
    
    # Storage directories
    raw_results_dir: str = os.path.join("phase11_evaluation", "results", "raw")
    final_results_dir: str = os.path.join("phase11_evaluation", "results", "final")
    reports_dir: str = os.path.join("phase11_evaluation", "reports")
    
    # Reproducibility
    seed: int = 42
    device: str = "cpu"
    
    # Deterministic generation
    deterministic_max_new_tokens: int = 80
    deterministic_temperature: float = 0.0
    
    # Stochastic sampling generation
    sampling_max_new_tokens: int = 80
    sampling_temperature: float = 0.7
    sampling_top_k: int = 50
    
    # Four target models
    models: List[ModelTargetConfig] = field(default_factory=lambda: [
        ModelTargetConfig(
            system_id="minigpt_baseline",
            display_name="MiniGPT Baseline (Phase 8)",
            category="from_scratch",
            checkpoint_path=os.path.join("checkpoints", "exp1_baseline", "best.pt"),
            tokenizer_type="character"
        ),
        ModelTargetConfig(
            system_id="minigpt_programming",
            display_name="MiniGPT Programming (Phase 9)",
            category="domain_scratch",
            checkpoint_path=os.path.join("checkpoints", "phase9_longer", "best.pt"),
            tokenizer_type="character"
        ),
        ModelTargetConfig(
            system_id="distilgpt2_base",
            display_name="DistilGPT-2 Base (Pretrained)",
            category="pretrained_base",
            checkpoint_path="distilbert/distilgpt2",
            tokenizer_type="bpe"
        ),
        ModelTargetConfig(
            system_id="distilgpt2_lora",
            display_name="DistilGPT-2 + LoRA (Phase 10)",
            category="peft_lora",
            checkpoint_path=os.path.join("phase10_lora", "checkpoints", "distilgpt2_lora_programming"),
            tokenizer_type="bpe",
            base_model_name="distilbert/distilgpt2",
            is_peft=True
        )
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompts_path": self.prompts_path,
            "raw_results_dir": self.raw_results_dir,
            "final_results_dir": self.final_results_dir,
            "reports_dir": self.reports_dir,
            "seed": self.seed,
            "device": self.device,
            "deterministic_max_new_tokens": self.deterministic_max_new_tokens,
            "deterministic_temperature": self.deterministic_temperature,
            "sampling_max_new_tokens": self.sampling_max_new_tokens,
            "sampling_temperature": self.sampling_temperature,
            "sampling_top_k": self.sampling_top_k,
            "models": [asdict(m) for m in self.models]
        }
