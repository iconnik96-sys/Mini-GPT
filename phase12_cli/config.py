"""
Configuration and Generation Settings for Phase 12 CLI.

Defines:
- GenerationSettings with strict validation (temperature, top_k, max_new_tokens, seed)
- CLIConfig holding default values and runtime options
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class GenerationSettings:
    max_new_tokens: int = 80
    temperature: float = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    seed: Optional[int] = 42
    stream: bool = True

    def validate(self) -> None:
        """Validate generation parameters and raise ValueError with clear guidance if invalid."""
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError(
                f"Invalid max_new_tokens ({self.max_new_tokens}). Must be a positive integer greater than 0."
            )
        if not isinstance(self.temperature, (int, float)) or self.temperature <= 0.0:
            raise ValueError(
                f"Invalid temperature ({self.temperature}). Must be a positive float greater than 0.0 (e.g. 0.7 or 1.0)."
            )
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or self.top_k <= 0:
                raise ValueError(
                    f"Invalid top_k ({self.top_k}). Must be a positive integer (e.g. 20 or 50), or None."
                )
        if self.top_p is not None:
            if not isinstance(self.top_p, (int, float)) or not (0.0 < self.top_p <= 1.0):
                raise ValueError(
                    f"Invalid top_p ({self.top_p}). Must be a float between 0.0 and 1.0, or None."
                )
        if self.repetition_penalty is not None:
            if not isinstance(self.repetition_penalty, (int, float)) or self.repetition_penalty < 1.0:
                raise ValueError(
                    f"Invalid repetition_penalty ({self.repetition_penalty}). Must be a float >= 1.0."
                )
        if self.no_repeat_ngram_size is not None:
            if not isinstance(self.no_repeat_ngram_size, int) or self.no_repeat_ngram_size < 0:
                raise ValueError(
                    f"Invalid no_repeat_ngram_size ({self.no_repeat_ngram_size}). Must be an integer >= 0."
                )
        if self.seed is not None:
            if not isinstance(self.seed, int):
                raise ValueError(
                    f"Invalid seed ({self.seed}). Must be an integer or None."
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CLIConfig:
    default_model: str = "distilgpt2-lora"
    device: str = "cpu"
    default_settings: GenerationSettings = field(default_factory=GenerationSettings)
