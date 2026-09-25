# Phase 11 — Evaluation & Comparison

## 1. Overview & Goal

Phase 11 establishes a unified, scientific evaluation framework comparing four distinct language modeling systems developed throughout the MiniGPT project:

1. **MiniGPT Baseline (Phase 8):** From-scratch decoder-only Transformer trained on Shakespeare (*Coriolanus*).
2. **MiniGPT Programming (Phase 9):** From-scratch decoder-only Transformer trained on software engineering text (Java, Spring Boot, SQL).
3. **DistilGPT-2 Base (Pretrained):** 82M causal language model pretrained on WebText by Hugging Face (zero fine-tuning).
4. **DistilGPT-2 + LoRA (Phase 10):** Pretrained DistilGPT-2 adapted to backend programming instructions using Low-Rank Adaptation (LoRA $r=8$).

### Key Objective
Understand the empirical trade-offs between:
- Building and training a Transformer from scratch vs using a pretrained LLM
- General-domain vs domain-specific training
- Full parameter updates vs Parameter-Efficient Fine-Tuning (LoRA)

---

## 2. Model Comparison Scoreboard

| System ID | Display Name | Architecture | Tokenizer | Total Params | Trainable Params | Trainable % | Checkpoint Size | Avg Throughput |
|---|---|---|---|---:|---:|---:|---:|---:|
| `minigpt_baseline` | MiniGPT Baseline | Custom Transformer (2L, 4H, 64D) | Character (52) | 110,336 | 110,336 | 100.0000% | 1.33 MB | 222.7 tok/s |
| `minigpt_programming` | MiniGPT Programming | Custom Transformer (2L, 4H, 64D) | Character (88) | 114,944 | 114,944 | 100.0000% | 1.39 MB | 424.6 tok/s |
| `distilgpt2_base` | DistilGPT-2 Base | DistilGPT-2 (6L, 12H, 768D) | Byte-level BPE (50,257) | 81,912,576 | 0 | 0.0000% | 334.35 MB | 33.1 tok/s |
| `distilgpt2_lora` | DistilGPT-2 + LoRA | DistilGPT-2 + LoRA (r=8, alpha=16) | Byte-level BPE (50,257) | 82,060,032 | 147,456 | **0.1797%** | **3.96 MB** | 26.6 tok/s |

---

## 3. Key Findings

1. **Parameter Efficiency of LoRA:** Updating only **147,456 adapter parameters (0.1797%)** across the attention projection layers (`c_attn`) steered the 82M model to follow instructions and speak Spring Boot/JPA terminology, dropping prompt repetition rates from **85.6% down to 40.6%**.
2. **Character Tokenizer Vocabulary Boundaries:** The Phase 8 baseline was trained strictly on Shakespeare and lacks numbers, `@`, and brackets. Prompts requiring modern code syntax were rejected before the forward pass, clearly proving how character vocabularies bound model capabilities.
3. **Perplexity Incomparability:** Character perplexity ($\approx 13.0$) and BPE perplexity ($\approx 34.5$) **cannot be compared directly** because predicting 1 of 88 characters has much lower theoretical entropy ($\log 88 \approx 4.47$) than predicting 1 of 50,257 subwords ($\log 50257 \approx 10.82$).
4. **Pedagogical vs Practical Trade-offs:** Scratch models teach fundamental Transformer tensor operations and gradient dynamics, but pretrained models adapted via LoRA offer drastically superior language coherence with minimal storage overhead.

---

## 4. Directory Structure

```text
phase11_evaluation/
├── README.md                                  # Phase 11 overview & instructions
├── run_evaluation.py                          # Master execution script
├── config.py                                  # Phase11Config and ModelTargetConfig dataclasses
├── src/
│   ├── __init__.py                            # Module exports
│   ├── loaders.py                             # Unified model & tokenizer loaders
│   ├── metrics.py                             # Lexical diversity, repetition, and syntax heuristics
│   ├── generation_eval.py                     # Single/multi prompt generation engine
│   └── report.py                              # JSON compiler and Markdown report generator
├── tests/
│   └── test_evaluation.py                     # 10 unit and integration tests
├── prompts/
│   └── evaluation_prompts.json                # 13 standardized programming benchmark prompts
├── results/
│   ├── raw/                                   # Raw per-system generation JSON outputs
│   └── final/
│       └── summary_comparison.json            # Structured multi-system comparison JSON
└── reports/
    └── phase11_report.md                      # Full 15-section scientific evaluation report
```

---

## 5. How to Run Phase 11

### Run Phase 11 Tests
```powershell
python -m unittest discover -s phase11_evaluation/tests -p "test_*.py" -v
```

### Run Full Multi-System Evaluation
```powershell
python phase11_evaluation/run_evaluation.py
```

### Run Complete Project Test Suite (Phase 1–11)
```powershell
# Phase 1-9 tests
python -m unittest discover -s tests -p "test_*.py" -v

# Phase 10 tests
python -m unittest discover -s phase10_lora/tests -p "test_*.py" -v

# Phase 11 tests
python -m unittest discover -s phase11_evaluation/tests -p "test_*.py" -v
```
For the complete technical report, see [`phase11_evaluation/reports/phase11_report.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase11_evaluation/reports/phase11_report.md).
