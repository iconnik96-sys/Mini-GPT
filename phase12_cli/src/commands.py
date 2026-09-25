"""
Command Handlers for Phase 12 CLI.

Dispatches:
- `models`: list all registered models
- `info`: inspect model architecture and parameters
- `generate`: one-shot text generation
- `chat`: interactive REPL session with slash commands
"""

import sys
import os
from typing import Optional, Any

from phase12_cli.src.model_registry import list_models, get_model_info
from phase12_cli.src.model_loader import ModelSession
from phase12_cli.src.generator import generate_text
from phase12_cli.config import GenerationSettings
from phase12_cli.src.formatting import (
    format_models_table,
    format_model_info,
    format_settings,
    format_generation_stats,
    format_chat_header,
    format_error
)


def handle_models(args: Any = None) -> int:
    """Handle 'models' command: list all models without loading weights."""
    models = list_models()
    print(format_models_table(models))
    return 0


def handle_info(args: Any, session: Optional[ModelSession] = None) -> int:
    """Handle 'info' command: display metadata and parameter counts."""
    model_id = args.model
    try:
        meta = get_model_info(model_id)
        device = getattr(args, "device", "cpu")
        print(format_model_info(meta, device=device))
        return 0
    except Exception as e:
        print(format_error(str(e)), file=sys.stderr)
        return 1


def handle_generate(args: Any, session: Optional[ModelSession] = None) -> int:
    """Handle 'generate' command: run one-shot text completion."""
    sess = session or ModelSession()
    device = getattr(args, "device", "cpu")
    
    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        stream=getattr(args, "stream", True)
    )

    try:
        settings.validate()
        print(f"\nLoading '{args.model}' on {device.upper()}...")
        model, tokenizer, meta = sess.get_or_load(args.model, device=device)
        
        print("\n" + "=" * 60)
        print(f"Model: {meta.display_name}")
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        print("Response:\n")
        
        res = generate_text(
            model=model,
            tokenizer=tokenizer,
            metadata=meta,
            prompt=args.prompt,
            settings=settings,
            device=device
        )
        
        if not settings.stream:
            print(res.output)
            
        print("\n" + "-" * 60)
        print(format_generation_stats(res))
        print("=" * 60 + "\n")
        return 0
    except Exception as e:
        print(format_error(str(e)), file=sys.stderr)
        return 1


def handle_chat(args: Any, session: Optional[ModelSession] = None) -> int:
    """Handle 'chat' command: interactive REPL session with slash commands."""
    sess = session or ModelSession()
    device = getattr(args, "device", "cpu")
    current_model_id = args.model
    
    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        stream=True
    )

    try:
        settings.validate()
        print(f"\nInitializing chat session with '{current_model_id}'...")
        model, tokenizer, meta = sess.get_or_load(current_model_id, device=device)
    except Exception as e:
        print(format_error(str(e)), file=sys.stderr)
        return 1

    print("\n" + format_chat_header(meta, device))

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd in ["/exit", "/quit", "/q"]:
                    print("\nExiting MiniGPT CLI. Goodbye!")
                    break

                elif cmd in ["/help", "/h"]:
                    print("\nSupported Commands:")
                    print("  /help                     Show this help message")
                    print("  /model <model_id>         Switch active model")
                    print("  /info                     Inspect current model properties")
                    print("  /settings                 Show current generation settings")
                    print("  /set <key> <val>          Update setting (e.g. /set temperature 0.5)")
                    print("  /clear                    Clear terminal screen")
                    print("  /exit                     Exit chat session")

                elif cmd == "/info":
                    print("\n" + format_model_info(meta, device=device))

                elif cmd == "/settings":
                    print("\n" + format_settings(settings))

                elif cmd == "/set":
                    if len(parts) < 3:
                        print("\nUsage: /set <temperature|max_new_tokens|top_k|seed> <value>")
                        continue
                    key, val = parts[1].lower(), parts[2]
                    try:
                        if key == "temperature":
                            settings.temperature = float(val)
                        elif key == "max_new_tokens":
                            settings.max_new_tokens = int(val)
                        elif key == "top_k":
                            settings.top_k = int(val) if val.lower() != "none" else None
                        elif key == "seed":
                            settings.seed = int(val) if val.lower() != "none" else None
                        else:
                            print(f"\n[Error] Unknown setting '{key}'.")
                            continue
                        settings.validate()
                        print(f"\nUpdated {key} to {val}.")
                    except ValueError as ve:
                        print(format_error(str(ve)))

                elif cmd == "/clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    print(format_chat_header(meta, device))

                elif cmd == "/model":
                    if len(parts) < 2:
                        print(f"\nCurrent active model: '{current_model_id}'.")
                        print("To switch: /model <model_id>. (Use 'models' command to view catalog)")
                        continue
                    new_id = parts[1]
                    try:
                        print(f"\nSwitching model to '{new_id}'...")
                        model, tokenizer, meta = sess.get_or_load(new_id, device=device)
                        current_model_id = new_id
                        print(f"Switched successfully to {meta.display_name}.")
                    except Exception as e:
                        print(format_error(str(e)))

                else:
                    print(f"\nUnknown command '{cmd}'. Type /help for available commands.")

                continue

            # Standard prompt generation
            print(f"\n{meta.id} > ", end="")
            res = generate_text(
                model=model,
                tokenizer=tokenizer,
                metadata=meta,
                prompt=user_input,
                settings=settings,
                device=device
            )
            print(f" {format_generation_stats(res)}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession interrupted by user. Exiting MiniGPT CLI. Goodbye!")
            break
        except Exception as e:
            print(format_error(str(e)))

    return 0
