# MiniGPT: From-Scratch GPT Implementation

MiniGPT is an educational, clean-room implementation of a **GPT-style decoder-only Transformer** built entirely from scratch in PyTorch.

> **Important Principle:**
> MiniGPT is designed purely for deep conceptual understanding of Large Language Models (LLMs).
> - **Zero Pretrained Weights:** We do **NOT** use, import, or fine-tune any pretrained weights or libraries (no LLaMA, GPT-2, GPT-3, GPT-4, Mistral, or Gemma checkpoints).
> - **Zero High-Level Black Boxes:** No HuggingFace Transformers, LangChain, or pre-built model pipelines.
> - **Scratch-Built:** Every tensor, tokenizer mapping, attention mechanism, residual block, and training loop will be coded explicitly and trained starting from **randomly initialized weights**.

---

## Table of Contents
1. [Goal of MiniGPT](#goal-of-minigpt)
2. [Pretraining vs. Fine-Tuning](#pretraining-vs-fine-tuning)
3. [The End-to-End Pipeline](#the-end-to-end-pipeline)
4. [Planned Development Phases](#planned-development-phases)
5. [Hardware & Memory Considerations](#hardware--memory-considerations)
6. [Project Structure](#project-structure)
7. [Getting Started (Phase 1)](#getting-started-phase-1)

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

## Planned Development Phases

Development proceeds across 8 structured, iterative phases:

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
├── src/                # Implementation modules
│   ├── __init__.py     # Package initialization
│   ├── tokenizer.py    # Custom character-level tokenizer (Phase 2 - Implemented)
│   ├── dataset.py      # Dataset loading, normalization & stats (Phase 3 & 9 - Implemented)
│   ├── model.py        # GPT Transformer architecture (Phase 4 - Implemented)
│   ├── train.py        # Training loop, optimization & metrics (Phase 5 - Implemented)
│   ├── checkpoint.py   # Atomic checkpoint saving & loading (Phase 6 - Implemented)
│   ├── generate.py     # Autoregressive generation & CLI (Phase 7 - Implemented)
│   └── experiment.py   # Benchmarking suite & domain experiments (Phase 8 & 9 - Implemented)
├── tests/              # Unit tests and sanity checks
│   ├── __init__.py
│   ├── test_structure.py            # Phase 1 verification
│   ├── test_tokenizer.py            # Phase 2 tokenizer verification
│   ├── test_dataset.py              # Phase 3 dataset verification
│   ├── test_model.py                # Phase 4 model verification
│   ├── test_train.py                # Phase 5 training verification
│   ├── test_checkpoint.py           # Phase 6 checkpoint verification
│   ├── test_generate.py             # Phase 7 generation verification
│   ├── test_experiments.py          # Phase 8 experiment verification
│   └── test_domain_customization.py # Phase 9 domain customization verification
├── config.py           # Central hyperparameters & hardware configuration
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation and roadmap
└── .gitignore          # Git exclusion rules
```

---

## Getting Started

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
