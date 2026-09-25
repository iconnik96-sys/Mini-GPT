"""
Phase 12: MiniGPT Local CLI Application.
"""

from phase12_cli.config import GenerationSettings, CLIConfig
from phase12_cli.src.model_registry import ModelMetadata, list_models, get_model_info, is_valid_model
from phase12_cli.src.model_loader import ModelSession
from phase12_cli.src.generator import GenerationResult, generate_text
from phase12_cli.src.commands import handle_models, handle_info, handle_generate, handle_chat
from phase12_cli.src.formatting import (
    format_banner,
    format_models_table,
    format_model_info,
    format_settings,
    format_generation_stats,
    format_chat_header,
    format_error
)

__all__ = [
    "GenerationSettings",
    "CLIConfig",
    "ModelMetadata",
    "list_models",
    "get_model_info",
    "is_valid_model",
    "ModelSession",
    "GenerationResult",
    "generate_text",
    "handle_models",
    "handle_info",
    "handle_generate",
    "handle_chat",
    "format_banner",
    "format_models_table",
    "format_model_info",
    "format_settings",
    "format_generation_stats",
    "format_chat_header",
    "format_error"
]
