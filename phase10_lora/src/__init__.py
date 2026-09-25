"""
Phase 10: Pretrained LLM Fine-Tuning with LoRA / QLoRA.
"""

from phase10_lora.src.config import Phase10Config
from phase10_lora.src.dataset import (
    PROMPT_TEMPLATE,
    format_instruction,
    load_dataset,
    split_dataset,
    SFTInstructionDataset,
    create_dataloaders
)
from phase10_lora.src.train import (
    count_parameters,
    setup_model_and_tokenizer,
    train_lora,
    load_fine_tuned_model
)
from phase10_lora.src.generate import generate_response
from phase10_lora.src.evaluate import BENCHMARK_PROMPTS, run_benchmark, run_evaluation_comparison

__all__ = [
    "Phase10Config",
    "PROMPT_TEMPLATE",
    "format_instruction",
    "load_dataset",
    "split_dataset",
    "SFTInstructionDataset",
    "create_dataloaders",
    "count_parameters",
    "setup_model_and_tokenizer",
    "train_lora",
    "load_fine_tuned_model",
    "generate_response",
    "BENCHMARK_PROMPTS",
    "run_benchmark",
    "run_evaluation_comparison"
]
