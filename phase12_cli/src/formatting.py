"""
Terminal Formatting Utilities for Phase 12 CLI.

Produces clean, readable ASCII cards, tables, and statistics.
"""

from typing import List
from phase12_cli.src.model_registry import ModelMetadata
from phase12_cli.src.generator import GenerationResult
from phase12_cli.config import GenerationSettings


def format_banner() -> str:
    return (
        "======================================================================\n"
        "                  MiniGPT Local Terminal Assistant                    \n"
        "======================================================================"
    )


def format_models_table(models: List[ModelMetadata]) -> str:
    """Format available models in a structured ASCII table."""
    lines = [
        format_banner(),
        "\nAvailable Models in Registry:\n",
        f"{'Model ID':<22} | {'Architecture':<24} | {'Parameters':<10} | {'Tokenizer':<9}",
        "-" * 72
    ]
    for m in models:
        lines.append(
            f"{m.id:<22} | {m.display_name[:24]:<24} | {m.parameter_count:<10,d} | {m.tokenizer_type:<9}"
        )
    lines.append("-" * 72)
    lines.append("\nRun 'python phase12_cli/cli.py info <model_id>' for detailed inspection.")
    lines.append("Run 'python phase12_cli/cli.py generate --model <model_id> --prompt \"...\"' to generate.")
    return "\n".join(lines)


def format_model_info(meta: ModelMetadata, device: str = "cpu") -> str:
    """Format detailed model inspection view."""
    lines = [
        "=" * 60,
        f" Model Information: {meta.display_name}",
        "=" * 60,
        f"ID:                   {meta.id}",
        f"Type:                 {meta.model_type}",
        f"Architecture:         {meta.display_name}",
        f"Tokenizer:            {meta.tokenizer_type.upper()}",
        f"Context Length:       {meta.context_length} tokens",
        f"Total Parameters:     {meta.parameter_count:,}",
        f"Trainable Parameters: {meta.trainable_parameters:,}",
        f"Frozen Parameters:    {meta.frozen_parameters:,}",
        f"Trainable Ratio:      {meta.trainable_percentage:.4f}%",
        f"Checkpoint Path:      {meta.checkpoint_path}",
        f"Device:               {device.upper()}",
        "-" * 60,
        f"Description:\n  {meta.description}",
        "=" * 60
    ]
    if meta.model_type == "peft_lora":
        lines.insert(10, f"PEFT Efficiency:      {meta.trainable_parameters:,} trainable | {100.0 - meta.trainable_percentage:.2f}% frozen")
    return "\n".join(lines)


def format_settings(settings: GenerationSettings) -> str:
    """Format current generation settings."""
    return (
        "------------------------------------\n"
        " Current Generation Settings:\n"
        "------------------------------------\n"
        f"  max_new_tokens : {settings.max_new_tokens}\n"
        f"  temperature    : {settings.temperature}\n"
        f"  top_k          : {settings.top_k}\n"
        f"  seed           : {settings.seed}\n"
        f"  stream         : {settings.stream}\n"
        "------------------------------------"
    )


def format_generation_stats(res: GenerationResult) -> str:
    """Format performance metadata summary."""
    return (
        f"\n[Stats] Generated {res.tokens_generated} {res.token_unit}-tokens in "
        f"{res.elapsed_seconds:.3f}s ({res.tokens_per_second:.1f} tok/s)"
    )


def format_chat_header(meta: ModelMetadata, device: str) -> str:
    """Display welcome header for interactive chat session."""
    return (
        "======================================================================\n"
        "                  MiniGPT Interactive Local Chat                      \n"
        "======================================================================\n"
        f"Model:  {meta.display_name}\n"
        f"ID:     {meta.id}\n"
        f"Device: {device.upper()}\n"
        "\nType /help for command list.\n"
        "Type /exit or press Ctrl+C to quit.\n"
        "======================================================================\n"
    )


def format_error(msg: str) -> str:
    """Format error message clearly without stacktrace."""
    return f"\n[Error] {msg}\n"
