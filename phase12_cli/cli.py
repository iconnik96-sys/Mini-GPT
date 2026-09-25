"""
Main Command-Line Interface for MiniGPT (Phase 12).

Usage Examples:
    python phase12_cli/cli.py models
    python phase12_cli/cli.py info distilgpt2-lora
    python phase12_cli/cli.py generate --model distilgpt2-lora --prompt "Explain dependency injection"
    python phase12_cli/cli.py chat --model distilgpt2-lora
"""

import sys
import os
import argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from phase12_cli.src.commands import (
    handle_models,
    handle_info,
    handle_generate,
    handle_chat
)
from phase12_cli.src.formatting import format_banner
from phase12_cli.config import CLIConfig


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands."""
    default_cfg = CLIConfig()

    parser = argparse.ArgumentParser(
        prog="minigpt",
        description="MiniGPT Local Terminal Assistant & Language Model Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Command: models
    subparsers.add_parser(
        "models",
        help="List all registered models without loading weights"
    )

    # Command: info
    parser_info = subparsers.add_parser(
        "info",
        help="Display detailed metadata, parameters, and architecture for a model"
    )
    parser_info.add_argument(
        "model",
        type=str,
        help="Model ID to inspect (e.g. distilgpt2-lora, minigpt-programming)"
    )
    parser_info.add_argument(
        "--device",
        type=str,
        default=default_cfg.device,
        help="Execution device (default: cpu)"
    )

    # Command: generate
    parser_gen = subparsers.add_parser(
        "generate",
        help="Run one-shot text completion on a prompt"
    )
    parser_gen.add_argument(
        "--model", "-m",
        type=str,
        default=default_cfg.default_model,
        help=f"Target model ID (default: {default_cfg.default_model})"
    )
    parser_gen.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Prompt text for text generation"
    )
    parser_gen.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_cfg.default_settings.max_new_tokens,
        help=f"Maximum new tokens to generate (default: {default_cfg.default_settings.max_new_tokens})"
    )
    parser_gen.add_argument(
        "--temperature", "-t",
        type=float,
        default=default_cfg.default_settings.temperature,
        help=f"Sampling temperature (default: {default_cfg.default_settings.temperature})"
    )
    parser_gen.add_argument(
        "--top-k", "-k",
        type=int,
        default=default_cfg.default_settings.top_k,
        help=f"Top-k filtering limit (default: {default_cfg.default_settings.top_k})"
    )
    parser_gen.add_argument(
        "--seed", "-s",
        type=int,
        default=default_cfg.default_settings.seed,
        help="Random seed for reproducible generation (default: 42)"
    )
    parser_gen.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        default=True,
        help="Disable real-time token streaming"
    )
    parser_gen.add_argument(
        "--device",
        type=str,
        default=default_cfg.device,
        help="Execution device (default: cpu)"
    )

    # Command: chat
    parser_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session with live model interaction"
    )
    parser_chat.add_argument(
        "--model", "-m",
        type=str,
        default=default_cfg.default_model,
        help=f"Target model ID (default: {default_cfg.default_model})"
    )
    parser_chat.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_cfg.default_settings.max_new_tokens,
        help=f"Max tokens per turn (default: {default_cfg.default_settings.max_new_tokens})"
    )
    parser_chat.add_argument(
        "--temperature", "-t",
        type=float,
        default=default_cfg.default_settings.temperature,
        help=f"Sampling temperature (default: {default_cfg.default_settings.temperature})"
    )
    parser_chat.add_argument(
        "--top-k", "-k",
        type=int,
        default=default_cfg.default_settings.top_k,
        help=f"Top-k filtering (default: {default_cfg.default_settings.top_k})"
    )
    parser_chat.add_argument(
        "--seed", "-s",
        type=int,
        default=default_cfg.default_settings.seed,
        help="Random seed (default: 42)"
    )
    parser_chat.add_argument(
        "--device",
        type=str,
        default=default_cfg.device,
        help="Execution device (default: cpu)"
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand is None:
        print(format_banner())
        parser.print_help()
        print("\nQuick Start:")
        print("  python phase12_cli/cli.py models")
        print("  python phase12_cli/cli.py generate --model distilgpt2-lora --prompt \"Explain OOP\"\n")
        return 0

    if args.subcommand == "models":
        return handle_models(args)
    elif args.subcommand == "info":
        return handle_info(args)
    elif args.subcommand == "generate":
        return handle_generate(args)
    elif args.subcommand == "chat":
        return handle_chat(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
