# MiniGPT: From-Scratch GPT Implementation

MiniGPT is an educational, clean-room implementation of a **GPT-style decoder-only Transformer** built entirely from scratch in PyTorch.

> **Important Principle:**
> MiniGPT is designed purely for deep conceptual understanding of Large Language Models (LLMs).
> - **Zero Pretrained Weights:** We do **NOT** use, import, or fine-tune any pretrained weights or libraries (no LLaMA, GPT-2, GPT-3, GPT-4, Mistral, or Gemma checkpoints).
> - **Zero High-Level Black Boxes:** No HuggingFace Transformers, LangChain, or pre-built model pipelines.
> - **Scratch-Built:** Every tensor, tokenizer mapping, attention mechanism, residual block, and training loop will be coded explicitly and trained starting from **randomly initialized weights**.

---

## Table of Contents
1. [Fresh Installation & Quickstart](#fresh-installation--quickstart)
2. [Model Checkpoint & Weights Guide](#model-checkpoint--weights-guide)
3. [Goal of MiniGPT](#goal-of-minigpt)
4. [Pretraining vs. Fine-Tuning](#pretraining-vs-fine-tuning)
5. [The End-to-End Pipeline](#the-end-to-end-pipeline)
6. [Development Roadmap (Phases 1–13)](#development-roadmap-phases-113)
7. [Hardware & Memory Considerations](#hardware--memory-considerations)
8. [Project Structure](#project-structure)
9. [Phase Summaries & Deep-Dives](#phase-summaries--deep-dives)

---

## Fresh Installation & Quickstart

Follow these instructions to clone, install, and run MiniGPT from a fresh Windows, macOS, or Linux environment.

### System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux (Windows tested and verified)
- **Python Version:** Python 3.10 – 3.12 (Python 3.12 recommended)
- **Node.js & npm:** Node.js v18+ and npm (required only for Phase 13 Studio frontend)
- **Git:** Standard git client
- **Hardware:** **CPU fully supported out of the box**; GPU / CUDA is completely optional and **not** required.

---

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Mini-Gpt
```

---

### Step 2: Create and Activate Python Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Python Dependencies

Install the pinned, tested requirements:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Note on PyTorch CPU:**
> The `requirements.txt` file specifies `--extra-index-url https://download.pytorch.org/whl/cpu`, which automatically installs the official lightweight, CPU-optimized PyTorch build without downloading unnecessary multi-gigabyte CUDA binaries.
> If you prefer to install PyTorch manually before running `pip install -r requirements.txt`:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

### Step 4: Verify Installation

Verify that the CLI can discover and list all registered models:

```powershell
python phase12_cli/cli.py models
```

Expected output:
```text
Available Models in MiniGPT Registry:
----------------------------------------------------------------------------------------------------
Model ID               | Architecture             | Parameters | Tokenizer
------------------------------------------------------------------------
distilgpt2-lora-v2     | DistilGPT-2 + LoRA v2 (I | 82,060,032 | bpe      
distilgpt2-lora        | DistilGPT-2 + LoRA (Phas | 82,060,032 | bpe      
minigpt-programming    | MiniGPT Programming (Pha | 114,944    | character
distilgpt2             | DistilGPT-2 Base (Pretra | 81,912,576 | bpe      
minigpt-baseline       | MiniGPT Baseline (Phase  | 110,336    | character
------------------------------------------------------------------------
```

---

### Step 5: Run the MiniGPT CLI (Phase 12)

**Interactive Chat Session:**
```powershell
python phase12_cli/cli.py chat --model minigpt-programming
```
*(Type prompts directly; use `/settings`, `/model`, `/help`, or `/exit` to navigate.)*

**One-Shot Generation with Live Streaming:**
```powershell
python phase12_cli/cli.py generate --model distilgpt2-lora --prompt "Explain dependency injection in Spring Boot" --max-new-tokens 80 --temperature 0.8
```

---

### Step 6: Start Backend API (Phase 13)

The FastAPI backend provides REST endpoints and real-time Server-Sent Events (SSE) streaming at `http://127.0.0.1:8000`:

```powershell
# From repository root:
python -m uvicorn phase13_api_web.backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

### Step 7: Start Frontend Web Studio (Phase 13)

Open a second terminal to launch the React 18 + Vite development server:

```powershell
cd phase13_api_web/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to interact with **MiniGPT Studio**.

To create an optimized production build of the frontend:
```powershell
npm run build
```

---

### Step 8: Run Automated Test Suites

Run all 136 unit and integration tests across all phases:

```powershell
# 1. Core scratch model tests (Phases 1–9: 94 tests)
python -m unittest discover -s tests -p "test_*.py" -v

# 2. LoRA fine-tuning tests (Phase 10: 10 tests)
python -m unittest discover -s phase10_lora/tests -p "test_*.py" -v

# 3. Evaluation framework tests (Phase 11: 10 tests)
python -m unittest discover -s phase11_evaluation/tests -p "test_*.py" -v

# 4. CLI application tests (Phase 12: 9 tests)
python -m unittest discover -s phase12_cli/tests -p "test_*.py" -v

# 5. FastAPI backend tests (Phase 13: 13 tests)
python -m unittest discover -s phase13_api_web/backend/tests -p "test_*.py" -v
```

---

## Model Checkpoint & Weights Guide

| Model Identifier | Type | Required Checkpoint Files | Status in Repo | Download Required? |
|---|---|---|---|---|
| `minigpt-baseline` | Scratch Transformer | `checkpoints/exp1_baseline/best.pt` or `checkpoints/best.pt` (~1.4 MB) | **Included** | No |
| `minigpt-programming` | Scratch Transformer | `checkpoints/phase9_longer/best.pt` (~1.4 MB) | **Included** | No |
| `distilgpt2` | Pretrained Hugging Face | Base Hugging Face model weights (`distilbert/distilgpt2`, ~334 MB) | External HF Hub | **Automatic** on first run |
| `distilgpt2-lora-v2` | LoRA Adapter (v2 Improved) | Base HF weights + `phase10_lora/checkpoints/distilgpt2_lora_programming_v2/` (~591 KB) | **Adapter Included** | Base auto-downloaded on first run |
| `distilgpt2-lora` | LoRA Adapter (v1 Baseline) | Base HF weights + `phase10_lora/checkpoints/distilgpt2_lora_programming/` (~591 KB) | **Adapter Included** | Base auto-downloaded on first run |

### Details on Included vs. Auto-Downloaded Files:

1. **Included in Git Repository:**
   - `checkpoints/best.pt`: Phase 8 Shakespeare reference model.
   - `checkpoints/exp1_baseline/best.pt`: Phase 8 baseline model.
   - `checkpoints/phase9_longer/best.pt`: Phase 9 programming domain reference model.
   - `phase10_lora/checkpoints/distilgpt2_lora_programming_v2/`: Trained v2 LoRA adapter weights (591 KB) + configs (improved model).
   - `phase10_lora/checkpoints/distilgpt2_lora_programming/`: Trained v1 LoRA adapter weights (591 KB) + configs (legacy baseline).

2. **Files Downloaded Automatically:**
   - The base model `distilbert/distilgpt2` weights (~334 MB) are fetched directly from Hugging Face Hub using standard `transformers` caching in `~/.cache/huggingface/hub/` on the first time `distilgpt2`, `distilgpt2-lora`, or `distilgpt2-lora-v2` is loaded.
   - **No manual downloads, API tokens, or external setup steps are required.**

3. **Re-training Scratch Checkpoints (Optional):**
   - If any scratch checkpoint is removed or you wish to retrain from scratch:
     - Baseline: `python -m src.experiment --name baseline` (~8s on CPU)
     - Programming: `python -m src.experiment --phase9` (~24s on CPU)
     - LoRA v2: `python phase10_lora/train_v2.py` (~535s on CPU)
     - LoRA v1: `python phase10_lora/src/train_lora.py` (~178s on CPU)

---


## Goal of MiniGPT

Modern LLMs are often interacted with through high-level APIs or monolithic libraries that obscure the foundational mechanics of generative deep learning. The goal of MiniGPT is to demystify these models by building one step-by-step from mathematical first principles:

- Understand how raw text is broken down into numerical tokens.
- Grasp how positional and token embeddings represent sequence structure.
- Implement scaled dot-product causal self-attention and multi-head attention.
- Assemble residual Transformer blocks with Layer Normalization and Feed-Forward Networks.
- Train the resulting model on next-token prediction using the AdamW optimizer.
- Autoregressively sample from the trained model to generate original text.

---

## Pretraining vs. Fine-Tuning

Understanding the distinction between **Pretraining** and **Fine-Tuning** is foundational in modern NLP:

### 1. Pretraining (What MiniGPT Does)
- **Objective:** Self-supervised next-token prediction on a large raw text corpus.
- **Starting Point:** Randomly initialized weights (uniform / Gaussian noise with near-zero knowledge).
- **Process:** The model reads tokens sequentially ($x_1, x_2, \dots, x_t$) and predicts the probability distribution of the next token $x_{t+1}$. It computes cross-entropy loss against the true next token and updates all parameters via backpropagation.
- **Outcome:** The model learns the statistical structure of language, grammar, syntax, world knowledge, and context dependencies without human labeling.
- **MiniGPT Scope:** MiniGPT is a **pure pretraining project**. We train our Transformer from random weights on a training corpus (e.g., TinyShakespeare).

### 2. Fine-Tuning (Downstream Adaptation)
- **Objective:** Adapting an already pretrained model to perform a specific task (e.g., classification, instruction following, or chat).
- **Starting Point:** Pretrained weights that already understand language structure.
- **Process:** Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF) using specialized question-answer or prompt-completion datasets.
- **Outcome:** A conversational assistant or task-specific classifier.

MiniGPT focuses on the hardest, most fundamental part: **Pretraining from scratch**.

---

## The End-to-End Pipeline

Here is the complete data and computation flow planned for MiniGPT:

```
raw text
  │
  ▼
tokenizer
  │
  ▼
token IDs
  │
  ▼
batches  [shape: (batch_size, block_size)]
  │
  ▼
embeddings  (Token Embeddings + Positional Embeddings)
  │
  ▼
causal self-attention  (Scaled Dot-Product with lower-triangular mask)
  │
  ▼
Transformer blocks  (LayerNorm + Multi-Head Attention + Residual + MLP)
  │
  ▼
logits  [shape: (batch_size, block_size, vocab_size)]
  │
  ▼
loss  (Cross-Entropy between logits and shifted targets)
  │
  ▼
backpropagation  (Compute gradients with respect to all weights)
  │
  ▼
AdamW  (Update parameters with weight decay and learning rate)
  │
  ▼
checkpoint  (Serialize model weights and training state to disk)
  │
  ▼
text generation  (Autoregressive sampling with temperature and top-k)
```

---

## Development Roadmap (Phases 1–13)

Development proceeded across 13 structured, iterative phases:

| Phase | Title | Scope & Deliverables |
|---|---|---|
| **Phase 1** | **Project Foundation** | Project architecture, configuration (`config.py`), dependencies (`requirements.txt`), git tracking, and educational roadmap. *(Completed)* |
| **Phase 2** | **Tokenizer** | Custom character-level tokenizer (`CharTokenizer`, `encode`, `decode`, `vocab_size`) constructed deterministically from scratch without third-party models. *(Completed)* |
| **Phase 3** | **Dataset & Batching** | Data loading, train/validation split, tensor batch generation `(x, y)` with block shifting and device placement. *(Completed)* |
| **Phase 4** | **Transformer Architecture** | Token & positional embeddings, causal multi-head self-attention, feed-forward network, layer normalization, residual connections, and language modeling head. *(Completed)* |
| **Phase 5** | **Training Loop** | Forward pass, cross-entropy loss, backward pass, AdamW optimization, gradient clipping, periodic train/validation loss estimation, and parameter updates. *(Completed)* |
| **Phase 6** | **Validation & Checkpoints** | Periodic evaluation on train/val splits to monitor overfitting, atomic checkpoint saving (`latest.pt`, `best.pt`), compatibility verification, and training resumption. *(Completed)* |
| **Phase 7** | **Text Generation** | Autoregressive sampling script (`src/generate.py`) accepting prompts, supporting temperature scaling, top-k filtering, greedy decoding, and checkpoint loading. *(Completed)* |
| **Phase 8** | **Experiments & Improvements** | Controlled empirical experiments (baseline, longer training, model scaling, 7 sampling modes), metrics calculation (PPL, throughput, params), and JSON tracking. *(Completed)* |
| **Phase 9** | **Domain Customization** | Domain customization for Java, Spring Boot, REST APIs, OOP, and SQL from scratch. Benchmarks, 7-prompt evaluations, and comparative analysis against Phase 8. *(Completed)* |
| **Phase 10** | **LoRA Pretrained LLM Adaptation** | Parameter-efficient fine-tuning (PEFT/LoRA) on `distilbert/distilgpt2`, training only 0.1797% of parameters on CPU for domain instruction following. *(Completed)* |
| **Phase 11** | **Evaluation & Comparison** | Multi-system evaluation framework benchmarking all 4 model systems across 13 standardized domain prompts with latency, throughput, and repetition metrics. *(Completed)* |
| **Phase 12** | **CLI Application** | Professional command-line interface with unified model registry, real-time token streaming, interactive chat REPL, and generation settings. *(Completed)* |
| **Phase 13** | **API + Web Interface** | FastAPI asynchronous backend with SSE streaming and React 18 + Vite "MiniGPT Studio" UI with live generation and performance metrics. *(Completed)* |

---

## Hardware & Memory Considerations

MiniGPT is specifically configured to run smoothly on a modern laptop with an **NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)**, while also remaining 100% compatible with CPU-only environments.

- **Parameter Budget:** ~10M parameters (~40 MB for FP32 weights).
- **Optimizer Memory:** AdamW maintains two FP32 states per parameter (~80 MB).
- **Activation Footprint:** With `batch_size = 32` and `block_size = 256`, activations consume ~500 MB – 1.2 GB during backpropagation.
- **Total VRAM Headroom:** ~1.5 GB to 2.5 GB peak VRAM usage, leaving plenty of cushion within the 6 GB capacity to prevent Out-Of-Memory (OOM) errors.

---

## Project Structure

```
MiniGPT/
├── data/               # Raw text datasets
│   ├── .gitkeep
│   ├── input.txt       # Shakespeare's Coriolanus (Phase 1-8)
│   └── programming.txt # Java, Spring Boot, REST APIs, OOP, SQL corpus (Phase 9)
├── checkpoints/        # Saved model weights and training checkpoints
│   ├── best.pt         # Reference Shakespeare checkpoint
│   ├── latest.pt       # Most recent training checkpoint
│   ├── exp1_baseline/  # Phase 8 Baseline experiment checkpoints
│   ├── exp2_longer_training/ # Phase 8 Longer training checkpoints
│   ├── exp3_model_size/      # Phase 8 Scaled model checkpoints
│   ├── phase9_baseline/      # Phase 9 Domain baseline checkpoints
│   ├── phase9_longer/        # Phase 9 Domain longer training checkpoints (Reference model)
│   └── phase9_hyperparam/    # Phase 9 Scaled domain model checkpoints
├── experiments/        # Machine-readable experiment records
│   ├── exp1_baseline.json
│   ├── exp2_longer_training.json
│   ├── exp3_model_size.json
│   ├── exp4_sampling.json
│   ├── summary.json          # Phase 8 benchmark summary
│   ├── phase9_baseline.json  # Phase 9 Baseline experiment record
│   ├── phase9_longer.json    # Phase 9 Longer training experiment record
│   ├── phase9_hyperparam.json# Phase 9 Scaled model experiment record
│   ├── phase9_prompts.json   # 7-prompt evaluation outputs (Greedy & Top-k)
│   └── phase9_summary.json   # Phase 9 scoreboard & Phase 8 comparison
├── src/                # Scratch implementation modules (Phases 1–9)
│   ├── __init__.py     # Package initialization
│   ├── tokenizer.py    # Custom character-level tokenizer (Phase 2)
│   ├── dataset.py      # Dataset loading, normalization & stats (Phases 3 & 9)
│   ├── model.py        # GPT Transformer architecture (Phase 4)
│   ├── train.py        # Training loop, optimization & metrics (Phase 5)
│   ├── checkpoint.py   # Atomic checkpoint saving & loading (Phase 6)
│   ├── generate.py     # Autoregressive generation & sampling (Phase 7)
│   └── experiment.py   # Benchmarking suite & domain experiments (Phases 8 & 9)
├── tests/              # Unit tests for from-scratch MiniGPT (Phases 1–9, 94 tests)
├── phase10_lora/       # Pretrained DistilGPT-2 + LoRA fine-tuning module (Phase 10)
│   ├── checkpoints/    # Trained LoRA adapter weights (adapter_model.safetensors)
│   ├── data/           # 30 curated Java/Spring instruction-response pairs
│   ├── src/            # LoRA fine-tuning, training loop, dataset & generation
│   └── tests/          # LoRA unit tests (10 tests)
├── phase11_evaluation/ # Multi-system evaluation framework (Phase 11)
│   ├── prompts/        # 13 standardized domain benchmark prompts
│   ├── reports/        # Comprehensive scientific evaluation report
│   ├── results/        # Machine-readable evaluation JSONs
│   ├── src/            # Perplexity, generation, and comparison metrics engine
│   └── tests/          # Evaluation test suite (10 tests)
├── phase12_cli/        # Command-line interface application (Phase 12)
│   ├── cli.py          # Unified CLI entrypoint (models, info, generate, chat)
│   ├── config.py       # CLI default settings and validation constants
│   ├── session.py      # Interactive REPL chat session manager
│   ├── streaming.py    # Dual real-time character & subword streaming engine
│   └── tests/          # CLI unit tests (9 tests)
├── phase13_api_web/    # Full-stack API & Web interface (Phase 13)
│   ├── backend/        # FastAPI async backend with SSE streaming
│   │   ├── app.py      # FastAPI application & REST/SSE routes
│   │   ├── config.py   # Backend server configuration & model paths
│   │   ├── manager.py  # Model loader & inference manager
│   │   └── tests/      # Backend API test suite (13 tests)
│   └── frontend/       # React 18 + Vite "MiniGPT Studio" UI
│       ├── src/        # React components (ChatPanel, ModelInfo, Settings, Header)
│       ├── package.json# Frontend dependencies
│       └── vite.config.js # Vite dev server & build configuration
├── config.py           # Central hyperparameters & hardware configuration
├── requirements.txt    # Project Python dependencies (CPU-optimized PyTorch)
├── README.md           # Project documentation and roadmap
└── .gitignore          # Git exclusion rules
```

---

## Phase Summaries & Deep-Dives

### 1. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run All Test Suites (94 Tests)
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 4. Generate Text from a Trained Checkpoint (Phase 7)
```powershell
# Temperature sampling with top-k filtering
python -m src.generate --checkpoint checkpoints/best.pt --prompt "The " --max_new_tokens 100 --temperature 0.8 --top_k 10

# Deterministic greedy generation
python -m src.generate --checkpoint checkpoints/best.pt --prompt "The " --max_new_tokens 100 --greedy
```

### 5. Run Experiments & Metrics Suite (Phase 8)
```powershell
# Run all benchmark experiments (Exp 1 through Exp 4) and print scoreboard:
python -m src.experiment --all

# Run individual experiments:
python -m src.experiment --name baseline
python -m src.experiment --name longer
python -m src.experiment --name size
python -m src.experiment --name sampling
```

#### Measured Experimental Scoreboard (Phase 8 - Shakespeare Corpus)

| Experiment | Parameters | Steps | Train Loss | Val Loss | Best Val Loss | Val PPL | Time (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|---|
| **Baseline** (L=2, H=4, D=64) | 110,336 | 150 | 2.2380 | 2.6535 | 2.6535 | 14.20 | 8.37s | 18,349.4 |
| **Longer Training** (L=2, H=4, D=64) | 110,336 | 500 | 1.6894 | 2.7393 | 2.6364 | 15.48 | 25.96s | 19,725.2 |
| **Larger Model** (L=4, H=4, D=128) | 812,800 | 150 | 2.0748 | 2.6160 | 2.6160 | 13.68 | 28.06s | 5,473.9 |

---

### 6. Programming Domain Customization (Phase 9)

In Phase 9, MiniGPT was customized from scratch for software engineering, covering **Java, Spring Boot, REST APIs, OOP principles, and SQL databases**:

- **Corpus (`data/programming.txt`):** 20,257 characters across 11 modules (entities, DTOs, controllers, services, repositories, exception handlers, SQL DDL/DML, and testing).
- **Universal Character Vocabulary:** 88 unique characters providing 100% token coverage with zero out-of-vocabulary (OOV) tokens across all Java identifiers, Spring annotations, brackets, and SQL queries.
- **Zero Pretrained Weights:** The domain model is trained purely from random initialization on CPU.

#### Run Phase 9 Domain Experiments & Prompt Evaluations
```powershell
# Run all Phase 9 experiments (Baseline, Longer Training, Scaled Model, 7-Prompt Evaluation):
python -m src.experiment --phase9

# Generate code from the Phase 9 reference checkpoint:
python -m src.generate --checkpoint checkpoints/phase9_longer/best.pt --prompt "public class UserService {" --max_new_tokens 80 --temperature 0.8 --top_k 10

# Test SQL completion:
python -m src.generate --checkpoint checkpoints/phase9_longer/best.pt --prompt "SELECT * FROM users" --max_new_tokens 80 --temperature 0.8 --top_k 10
```

#### Phase 9 Measured Experimental Scoreboard

| Experiment | Parameters | Steps | Train Loss | Val Loss | Best Val Loss | Val PPL | Time (s) | Throughput (tok/s) | Reference Checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| **Phase 9 Baseline** (L=2, H=4, D=64) | 114,944 | 150 | 2.3764 | 2.7572 | 2.7572 | 15.76 | 7.32s | 20,970.4 | `checkpoints/phase9_baseline/best.pt` |
| **Phase 9 Longer Training** (L=2, H=4, D=64) | 114,944 | 450 | 1.7552 | 2.6549 | **2.5680** | **13.04** | 24.06s | 19,154.6 | `checkpoints/phase9_longer/best.pt` |
| **Phase 9 Scaled Model** (L=4, H=4, D=128) | 822,016 | 150 | 2.1746 | 2.7017 | 2.7017 | 14.90 | 24.84s | 6,182.6 | `checkpoints/phase9_hyperparam/best.pt` |

#### Comparison with Phase 8 Baseline
- **Vocabulary Growth:** 52 characters (Shakespeare) $\to$ 88 characters (Programming), adding +4,608 parameters to token embedding & `lm_head`.
- **Lower Validation Perplexity:** Longer training reached **13.04** (Best Val Loss: **2.5680** at step 400), improving upon Phase 8's best validation loss ($2.6364$, PPL $13.96$).
- **Learned Code Patterns:** Model learned 4-space indentation, semicolons `;`, camelCase methods, Spring annotations (`@RestController`), SQL clauses (`SELECT * FROM users`), and comment headers (`//`).

---

### 7. Pretrained LLM Fine-Tuning with LoRA (Phase 10)

> **Architectural Boundary:**
> - **Phases 1–9:** Our own decoder-only Transformer built and trained **entirely from scratch**.
> - **Phase 10:** Parameter-Efficient Fine-Tuning (**LoRA**) of an open-source **pretrained causal LLM** (`distilbert/distilgpt2`).
>
> All Phase 10-specific code, tests, datasets, checkpoints, and experiment results reside exclusively in [`phase10_lora/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase10_lora/). Root `src/` remains the pristine from-scratch MiniGPT implementation.

#### Core Concept: Low-Rank Adaptation (LoRA)
Instead of updating all 82M pretrained weights ($W' = W + \Delta W$), LoRA freezes the base model and decomposes weight updates into two trainable low-rank matrices:
$$W' = W + \frac{\alpha}{r}(B \cdot A)$$

#### Measured Parameter Efficiency
```
Total parameters:     82,060,032
Trainable parameters: 147,456 (LoRA rank r=8, alpha=16 on c_attn)
Frozen parameters:    81,912,576
Trainable percentage: 0.1797%
```
**Over 99.82% of the base model remains completely frozen.**

#### Phase 10 Results Summary
- **Base Model:** `distilbert/distilgpt2` (82M params, Apache 2.0 license, running locally on CPU).
- **Domain:** Supervised instruction-following on Java, Spring Boot, REST APIs, OOP, and SQL (30 curated examples).
- **Training Time:** 178.44 seconds on CPU (8 epochs, AdamW lr=5e-4, loss: 4.1090 $\to$ 3.5563, val loss: 3.7001 $\to$ 3.5418).
- **Behavior Shift:** The base model suffered from degenerate looping (endlessly repeating prompts/headers); the LoRA model eliminated repetition and produced domain-aligned answers for Spring Boot, REST, JPA, and `@Transactional`.
- **QLoRA Investigation:** Truthfully documented that `bitsandbytes` 4-bit CUDA quantization requires an NVIDIA GPU and is not supported in the current CPU-only Windows environment; standard FP32 LoRA remains the verified working result.

#### Run Phase 10
```powershell
# Run Phase 10 unit tests (10 tests)
python -m unittest discover -s phase10_lora/tests -p "test_*.py" -v

# Run Phase 10 end-to-end experiment
python phase10_lora/run_experiment.py
```
For complete technical documentation, mathematical derivations, and side-by-side prompt comparisons, see [`phase10_lora/README.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase10_lora/README.md).

---

### 8. Cross-System Evaluation & Comparison (Phase 11)

In Phase 11, we established a unified, reproducible evaluation framework comparing all four language modeling systems built across the project on a standardized 13-prompt benchmark covering Java OOP, Spring Boot, REST APIs, JPA, and SQL.

All Phase 11 code, evaluation prompts, raw outputs, summary JSONs, unit tests, and reports reside in [`phase11_evaluation/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase11_evaluation/).

#### Multi-System Scoreboard

| System | Architecture | Tokenizer | Total Params | Trainable Params | Trainable % | Checkpoint Size | Avg Latency | Avg Throughput | Repetition Rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MiniGPT Baseline** | Custom Transformer (2L, 4H, 64D) | Character (52) | 110,336 | 110,336 | 100.0000% | 1.33 MB | 0.359s | 222.7 tok/s | 0.4000 |
| **MiniGPT Programming** | Custom Transformer (2L, 4H, 64D) | Character (88) | 114,944 | 114,944 | 100.0000% | 1.39 MB | 0.193s | 424.6 tok/s | 0.2428 |
| **DistilGPT-2 Base** | DistilGPT-2 (6L, 12H, 768D) | BPE (50,257) | 81,912,576 | 0 | 0.0000% | 334.35 MB | 2.414s | 33.1 tok/s | 0.8559 |
| **DistilGPT-2 + LoRA** | DistilGPT-2 + LoRA (r=8) | BPE (50,257) | 82,060,032 | 147,456 | **0.1797%** | **3.96 MB** | 1.523s | 26.6 tok/s | **0.4064** |

#### Key Scientific Findings
1. **LoRA Efficiency:** Adapting only **147,456 parameters (0.1797%)** across attention layers eliminated degenerate prompt repetition loops (repetition dropped from 85.6% to 40.6%) and steered the model toward backend programming responses.
2. **Vocabulary Bounds:** Scratch character models are strictly bounded by their training alphabet. Phase 8 (Shakespeare) rejected prompts containing numbers or `@`, while Phase 9 (88 chars) completed all 13 prompts cleanly.
3. **Perplexity Incomparability:** Character perplexity ($\approx 13.0$) cannot be ranked against subword BPE perplexity ($\approx 34.5$) due to differing token entropy upper bounds ($\log 88$ vs $\log 50257$).

#### Run Phase 11
```powershell
# Run Phase 11 unit tests (10 tests)
python -m unittest discover -s phase11_evaluation/tests -p "test_*.py" -v

# Run Phase 11 full evaluation
python phase11_evaluation/run_evaluation.py
```
For the comprehensive 15-section scientific report, see [`phase11_evaluation/reports/phase11_report.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase11_evaluation/reports/phase11_report.md).

---

### 9. MiniGPT CLI Application (Phase 12)

In Phase 12, we developed a professional, modular command-line interface that allows interactive chat, one-shot text generation, model inspection, and generation settings configuration across all 4 project models.

All Phase 12 implementation, configuration, tests, session logs, and documentation reside strictly in [`phase12_cli/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase12_cli/).

#### Key Features
- **Unified Model Registry:** Access `minigpt-baseline`, `minigpt-programming`, `distilgpt2`, and `distilgpt2-lora` through a single interface.
- **Lazy Weight Loading:** Inspecting models or viewing lists allocates zero neural network weights; models are loaded into a cached `ModelSession` only when generation begins.
- **Dual Real-Time Streaming:** Direct character-by-character flushing for from-scratch MiniGPT models, and Hugging Face `TextStreamer` integration for DistilGPT-2/LoRA.
- **Interactive Chat REPL:** Multi-turn conversational interface with built-in slash commands (`/help`, `/settings`, `/set`, `/model`, `/clear`, `/exit`).
- **One-Shot Generation with Metrics:** Outputs generated text alongside token count, generation latency, and throughput (tokens/second).
- **Strict Parameter Validation:** Comprehensive bounds checking for temperature, top-$k$, max tokens, and seeds with user-friendly error messages (no raw Python tracebacks).

#### PowerShell Usage Examples

```powershell
# 1. List all available models
python phase12_cli/cli.py models

# 2. Inspect model architecture and parameter counts
python phase12_cli/cli.py info distilgpt2-lora

# 3. One-shot generation with live streaming
python phase12_cli/cli.py generate `
  --model distilgpt2-lora `
  --prompt "Explain dependency injection in Spring Boot" `
  --max-new-tokens 60 `
  --temperature 0.8 `
  --top-k 20 `
  --seed 42

# 4. Generate with from-scratch programming MiniGPT
python phase12_cli/cli.py generate `
  --model minigpt-programming `
  --prompt "Explain inheritance in Java" `
  --max-new-tokens 60 `
  --temperature 0.8 `
  --top-k 20 `
  --seed 42

# 5. Launch interactive chat mode
python phase12_cli/cli.py chat --model distilgpt2-lora

# 6. Run Phase 12 test suite
python -m unittest discover -s phase12_cli/tests -p "test_*.py" -v
```

For full architecture details, session examples, and documentation, see [`phase12_cli/README.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase12_cli/README.md) and [`phase12_cli/examples/example_session.txt`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase12_cli/examples/example_session.txt).

---

### 10. MiniGPT Studio — API + Web Interface (Phase 13)

In Phase 13, we built **MiniGPT Studio**, a local developer workspace and web interface allowing visual interaction, live streaming generation, model selection, parameter tuning, and performance analysis across all four project models.

All Phase 13 backend, frontend, test, and documentation files reside strictly in [`phase13_api_web/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase13_api_web/).

#### Key Capabilities
- **FastAPI REST & Streaming Engine:** Clean asynchronous backend with `/api/health`, `/api/models`, `/api/generate`, and real-time Server-Sent Events (SSE) via `/api/generate/stream`.
- **React 18 + Vite Studio Frontend:** Dark-themed UI with model selection cards, model architecture specifications, hyperparameter sliders (temperature, top-k, max tokens, seed), prompt chips, and a live response container with blinking typing cursor.
- **Unified Model Management:** Integrates `minigpt-baseline`, `minigpt-programming`, `distilgpt2`, and `distilgpt2-lora` with lazy loading and active model caching in CPU RAM.
- **Dual Streaming Mechanisms:** Character-by-character SSE streaming for scratch Transformer models and `TextIteratorStreamer` subword streaming for DistilGPT-2/LoRA.
- **Performance Profiling:** Live generation time, token counts, and throughput (tokens/sec) with clear disclaimers regarding character vs BPE token equivalence.

#### Quickstart Commands

```powershell
# 1. Start the FastAPI backend
python -m uvicorn phase13_api_web.backend.app:app --host 127.0.0.1 --port 8000 --reload

# 2. In another terminal, start the React frontend
cd phase13_api_web/frontend
npm install
npm run dev

# 3. Run backend automated test suite (13 tests)
python -m unittest discover -s phase13_api_web/backend/tests -p "test_*.py" -v

# 4. Run end-to-end integration verification
python -m phase13_api_web.test_live_server
```

For full setup documentation, API specifications, and JSON examples, see [`phase13_api_web/README.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase13_api_web/README.md) and [`phase13_api_web/examples/api_examples.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase13_api_web/examples/api_examples.md).




