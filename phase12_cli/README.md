# Phase 12 — MiniGPT Local CLI Application

A lightweight, professional command-line interface for inspecting and interacting with all language models developed throughout the MiniGPT project.

---

## 1. Purpose

The Phase 12 CLI provides a developer-friendly terminal interface to explore, benchmark, and interact with both from-scratch and parameter-efficient fine-tuned models on a local machine.
It bridges the gap between raw Python training scripts and local generative AI tools, providing:
- Dynamic model inspection without loading weights into memory
- One-shot generation with latency and token throughput metrics
- Interactive chat REPL with runtime configuration updates and slash commands
- Real-time token streaming for responsive generation feedback
- Robust error handling for out-of-vocabulary characters and missing checkpoints

---

## 2. Architecture & Design Principles

```text
phase12_cli/
├── README.md                      # Phase 12 documentation and command guide
├── cli.py                         # Argparse CLI entry point and dispatch
├── config.py                      # GenerationSettings & CLIConfig with validation
├── src/
│   ├── __init__.py                # Package exports
│   ├── model_registry.py          # Static catalog of models and parameters
│   ├── model_loader.py            # Lazy model loader with memory session caching
│   ├── generator.py               # Streaming & non-streaming text generation engine
│   ├── commands.py                # Command handlers for models, info, generate, chat
│   └── formatting.py              # Clean ASCII cards, banners, and statistics formatters
├── tests/
│   └── test_cli.py                # Unit and integration test suite
└── examples/
    └── example_session.txt        # Verified end-to-end terminal recording
```

### Core Design Rules
1. **Lazy Loading:** Running `models` or `--help` never initializes PyTorch weights. Neural networks are loaded only when `generate` or `chat` is invoked.
2. **Session Persistence:** During an interactive chat session, the active model is kept in memory across multiple turns rather than reloading per prompt.
3. **Graceful Error Handling:** Expected user errors (unknown model ID, negative temperature, empty prompts, missing characters) yield informative human-readable alerts rather than unhandled Python tracebacks.

---

## 3. Supported Models in Registry

| Model ID | Display Name | Architecture | Tokenizer | Total Params | Trainable % | Checkpoint Location |
|---|---|---|---|---:|---:|---|
| `distilgpt2-lora` | DistilGPT-2 + LoRA (Phase 10) | DistilGPT-2 + LoRA (r=8, alpha=16) | Byte-level BPE (50,257) | 82,060,032 | **0.1797%** | `phase10_lora/checkpoints/distilgpt2_lora_programming` |
| `minigpt-programming` | MiniGPT Programming (Phase 9) | Custom Transformer (2L, 4H, 64D) | Character (88) | 114,944 | 100.0000% | `checkpoints/phase9_longer/best.pt` |
| `distilgpt2` | DistilGPT-2 Base (Pretrained) | DistilGPT-2 (6L, 12H, 768D) | Byte-level BPE (50,257) | 81,912,576 | 0.0000% | Hugging Face cache (`distilbert/distilgpt2`) |
| `minigpt-baseline` | MiniGPT Baseline (Phase 8) | Custom Transformer (2L, 4H, 64D) | Character (52) | 110,336 | 100.0000% | `checkpoints/exp1_baseline/best.pt` |

---

## 4. Installation & Setup

All dependencies are standard components already installed in the local environment:
- Python 3.10+
- PyTorch (`torch`)
- Hugging Face `transformers`
- Hugging Face `peft`

No additional external services or paid APIs are required.

---

## 5. Usage & Available Commands

The CLI provides four primary subcommands:
```powershell
python phase12_cli/cli.py models
python phase12_cli/cli.py info <model_id>
python phase12_cli/cli.py generate --model <model_id> --prompt "<prompt>" [options]
python phase12_cli/cli.py chat --model <model_id> [options]
```

### 1. `models` — List Available Models
```powershell
python phase12_cli/cli.py models
```
Displays a clean ASCII table of all registered models without allocating model memory.

### 2. `info` — Inspect Model Architecture & Parameters
```powershell
python phase12_cli/cli.py info distilgpt2-lora
```
Displays total parameters, trainable parameters, frozen parameters, tokenizer type, and checkpoint paths.

### 3. `generate` — One-Shot Text Completion
```powershell
python phase12_cli/cli.py generate `
  --model distilgpt2-lora `
  --prompt "Explain dependency injection in Spring Boot" `
  --max-new-tokens 60 `
  --temperature 0.8 `
  --top-k 20 `
  --seed 42
```
Outputs the response in real-time with streaming, followed by performance stats (tokens generated, latency, tokens/sec).

### 4. `chat` — Interactive REPL Session
```powershell
python phase12_cli/cli.py chat --model distilgpt2-lora
```
Enters a live chat loop. Supports runtime slash commands:
- `/help` — Display list of commands
- `/info` — View active model metadata
- `/settings` — Display active generation settings
- `/set <key> <val>` — Update a setting on the fly (e.g. `/set temperature 0.5`)
- `/model <model_id>` — Switch active model dynamically
- `/clear` — Clear terminal screen
- `/exit` — Cleanly exit session

---

## 6. Generation Settings & Validation

| Parameter | Type | Default | Description | Validation Rule |
|---|---|---|---|---|
| `--max-new-tokens` | int | 80 | Max tokens to generate beyond prompt | Must be $> 0$ |
| `--temperature` | float | 0.8 | Sampling entropy / randomness | Must be $> 0.0$ |
| `--top-k` | int | 20 | Filters candidate pool to top-k | Must be $> 0$ or `None` |
| `--seed` | int | 42 | Random seed for reproducibility | Must be an integer or `None` |
| `--no-stream` | flag | False | Disables live token streaming | Boolean flag |

Any invalid input (e.g. `--temperature -1` or `--max-new-tokens 0`) generates an immediate informative error without crashing.

---

## 7. Token Streaming Implementation

1. **Character-level MiniGPT:** Implemented in `generate_minigpt_stream()`, which generates character tokens sequentially and flushes each character directly to `sys.stdout` as it is sampled.
2. **BPE DistilGPT-2:** Leverages Hugging Face's native `TextStreamer(tokenizer, skip_prompt=True)`, rendering subwords incrementally as they are predicted.

---

## 8. Error Handling

- **Character Out-of-Vocabulary (OOV):** When passing modern code syntax (e.g. `@`, `{}`) to character models trained on Shakespeare, the CLI intercepts the missing characters and informs the user:
  ```text
  [Error] Prompt contains characters not supported by character model 'minigpt-baseline': ['@', '{', '}']
  Character models can only process characters present in their training corpus.
  Tip: Try using 'distilgpt2-lora' which supports full universal BPE subwords.
  ```
- **Missing Checkpoints:** If a checkpoint file is missing on disk, an alert instructs the user which prior training phase must be run.
- **Interrupts:** `Ctrl+C` (SIGINT) or EOF exits the interactive chat session cleanly with a goodbye message.

---

## 9. Performance & Measurements (CPU Execution)

Measurements performed on Intel 12-core CPU:
- **MiniGPT Programming:** ~327 to 650 char-tokens/second (~0.15s per prompt)
- **DistilGPT-2 + LoRA:** ~26 to 28 BPE-tokens/second (~0.8s per prompt)

---

## 10. Known Limitations

1. **No Long-Term Conversational Memory:** The chat REPL executes single-turn completions conditioned on the current prompt. It does not concatenate infinite dialogue turns because small context windows (64 chars for MiniGPT, 1024 tokens for DistilGPT-2) would rapidly overflow.
2. **Character Vocabulary Constraints:** The Phase 8 baseline cannot accept numbers or programming annotations.
3. **CPU Latency for Large Generation:** While generation at 27 tok/s is interactive, generating 250+ tokens on CPU takes ~8-10 seconds.

---

## 11. Testing

Run Phase 12 unit and integration tests:
```powershell
python -m unittest discover -s phase12_cli/tests -p "test_*.py" -v
```
All 9 test cases pass in under 0.05 seconds.
