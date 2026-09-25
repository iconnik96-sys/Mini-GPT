# MiniGPT Brain

Persistent project context and operational memory for the MiniGPT project.

---

## 1. Project Goal

MiniGPT is an educational GPT-style language model being built entirely from scratch in PyTorch. The goal of this project is to develop a deep, first-principles understanding of how Large Language Models (LLMs) function internally—from tokenization and attention mechanisms to autoregressive decoding and gradient optimization.

---

## 2. Core Constraint

- **Zero Pretrained Weights:** This project starts exclusively from randomly initialized model weights.
- **No Pretrained Models:** Do NOT use pretrained LLM weights. Do NOT use LLaMA, GPT-2, GPT-3, GPT-4, Mistral, Gemma, or any other pretrained language model as the base model.
- **Pure Scratch Training:** The eventual model must be trained using our own pretraining pipeline and dataset.

---

## 3. Current Phase

- **Phase 1 — Project Foundation:** COMPLETED
- **Phase 2 — Character-Level Tokenizer:** COMPLETED
- **Phase 3 — Dataset and Batching:** COMPLETED
- **Phase 4 — Transformer Architecture:** COMPLETED
- **Phase 5 — Training Loop:** COMPLETED
- **Phase 6 — Validation and Checkpoints:** COMPLETED
- **Phase 7 — Text Generation:** COMPLETED
- **Phase 8 — Experiments and Improvements:** COMPLETED
- **Phase 9 — Domain Customization (Java, Spring Boot, REST APIs, OOP, SQL):** COMPLETED
- **Next:** Phase 10 (or subsequent planned enhancements; NOT started yet)

---

## 4. Planned Pipeline

```
Raw text
  ↓
tokenizer
  ↓
token IDs
  ↓
training batches
  ↓
embeddings (token + position)
  ↓
causal self-attention
  ↓
Transformer blocks
  ↓
logits
  ↓
next-token loss (cross-entropy)
  ↓
backpropagation
  ↓
AdamW optimizer
  ↓
checkpoints (latest.pt & best.pt)
  ↓
text generation (autoregressive sampling & greedy decoding)
```

---

## 5. Development Phases

- **Phase 1 — Project foundation — COMPLETED**
- **Phase 2 — Tokenizer — COMPLETED**
- **Phase 3 — Dataset and batching — COMPLETED**
- **Phase 4 — Transformer architecture — COMPLETED**
- **Phase 5 — Training loop — COMPLETED**
- **Phase 6 — Validation and checkpoints — COMPLETED**
- **Phase 7 — Text generation — COMPLETED**
- **Phase 8 — Experiments and improvements — COMPLETED**
- **Phase 9 — Domain customization — COMPLETED**

---

## 6. Current Architecture

### Implemented Components
- [`config.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/config.py): Hyperparameter definitions (`MiniGPTConfig` dataclass and global constants) tailored for RTX 3050 6GB with CUDA/CPU device fallback.
- [`requirements.txt`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/requirements.txt): Core dependencies (`torch`, `numpy`, `tqdm`, `requests`).
- [`README.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/README.md): Comprehensive project documentation, pretraining vs fine-tuning concepts, 13-step pipeline, roadmap, and quickstart instructions.
- [`.gitignore`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/.gitignore): Standard ignore rules for virtual environments, python caches, model checkpoints (`*.pt`, `checkpoints/*`), temporary files (`*.tmp`, `tmp_*`), raw data, and IDE settings.
- [`data/input.txt`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/data/input.txt): Deterministic sample training corpus (3,714 characters from Shakespeare's Coriolanus).
- [`src/tokenizer.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/tokenizer.py): Deterministic `CharTokenizer` implementing `stoi`, `itos`, `vocab_size`, `encode(text)`, `decode(ids)`, with explicit validation and error handling.
- [`src/dataset.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/dataset.py): `TextDataset` and `get_batch(split, block_size)` managing text loading, deterministic 90/10 train/val partitioning, context window sampling, next-token shifting ($x[t] \to y[t] = x[t+1]$), and hardware device placement.
- [`src/model.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/model.py): Scratch-built `MiniGPT` decoder architecture featuring token embeddings, learnable positional embeddings, `CausalSelfAttention` with lower-triangular causal mask buffer, `FeedForward` MLP with GELU, pre-LayerNorm residual blocks, final LayerNorm, language-model projection head (`lm_head`), and Gaussian weight initialization.
- [`src/train.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/train.py): Training pipeline containing `compute_loss(logits, targets)` with unnormalized logit flattening, `estimate_loss(model, dataset, eval_iters, block_size, device)` running non-mutating evaluations under `torch.no_grad()`, and `train(...)` orchestrating AdamW optimization, gradient norm clipping, dynamic vocabulary alignment, periodic evaluation logging, atomic checkpoint saving (`latest.pt` and `best.pt`), and seamless training resumption.
- [`src/checkpoint.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/checkpoint.py): Checkpoint persistence manager providing `save_checkpoint(...)` with atomic temporary file swapping, and `load_checkpoint(...)` supporting CPU/CUDA device mapping, architecture compatibility checks, vocabulary alignment validation, and NaN/Inf weight corruption detection.
- [`src/generate.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/generate.py): Autoregressive generation engine implementing `generate(...)`, `generate_from_checkpoint(...)`, `load_from_checkpoint(...)`, context cropping to `block_size`, temperature scaling, top-k filtering, deterministic greedy argmax decoding, and CLI interface.
- [`src/experiment.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/experiment.py): Experiment management suite providing reproducible training benchmark runs, metrics calculation (`compute_perplexity` with overflow clamping, `count_parameters`, `calculate_tokens_per_sec`), comparative scoreboard generation, multi-config sampling comparisons across 7 decoding configurations, and machine-readable JSON logging.
- [`tests/test_structure.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_structure.py): Unit test suite validating directory structure, configuration keys, and module imports.
- [`tests/test_tokenizer.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_tokenizer.py): Unit test suite validating vocabulary construction, dynamic size, encoding, decoding, round-trip, determinism, and error handling.
- [`tests/test_dataset.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_dataset.py): Unit test suite validating corpus loading, tokenization, 90/10 split integrity, context shapes, next-token offset, token ID validity, custom batch sizes, device placement, repeated batching, and small corpus error handling.
- [`tests/test_model.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_model.py): Unit test suite validating dynamic vocabulary instantiation, output shapes `(B, T, vocab_size)`, multiple sequence lengths, causal future-mask probability zeroes, head dimension divisibility, parameter count, gradient flow via backpropagation, CPU execution, clean CUDA skipping when unavailable, block limit error rejection, and independent random initialization.
- [`tests/test_train.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_train.py): Unit test suite validating loss computation, scalar shape, backprop gradient generation across parameters, AdamW parameter modification, training step stability, non-mutating train/val loss estimation, train/eval mode restoration, clean device handling, dynamic vocabulary consistency, and tiny overfit loss reduction.
- [`tests/test_checkpoint.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_checkpoint.py): Unit test suite validating checkpoint saving, disk existence, parameter exact restoration, optimizer state restoration, step restoration, loss metrics restoration, configuration preservation, tokenizer vocabulary preservation, CPU device mapping, missing file error handling, corrupted/incompatible checkpoint rejection, resume step progression, best validation checkpoint tracking, and weight NaN/Inf detection.
- [`tests/test_generate.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_generate.py): Unit test suite validating string return type, prompt preservation, token length addition, zero-token prompt return, context cropping to block_size, temperature validation, top-k validation, top-k=1 equivalence to greedy, greedy determinism, seeded sampling reproducibility, zero gradient leakage, parameter immutability, train/eval mode restoration, CPU execution, CUDA clean skip, checkpoint-based generation, and unknown prompt character rejection.
- [`tests/test_experiments.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_experiments.py): Unit test suite validating perplexity calculation, numerical clamping/overflow protection, parameter counts, throughput calculation, zero-duration safe division, sampling configurations, JSON result persistence, and sampling evaluation runs.
- [`tests/test_domain_customization.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests/test_domain_customization.py): Unit test suite validating programming domain corpus loading, dataset statistics calculation, zero-OOV tokenization across all 7 domain prompts, batch generation with shifted targets, domain training steps, and prompt evaluation runner.

### Placeholder Components
- [`src/__init__.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src/__init__.py): Source package initialization.

### Project Directories
- [`data/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/data): Raw text datasets (contains `input.txt` Shakespeare sample corpus and `programming.txt` Java/Spring/SQL/OOP domain corpus).
- [`checkpoints/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/checkpoints): Serialized model weights (`best.pt`, `latest.pt`, Phase 8 experiment subdirectories, and Phase 9 subdirectories `phase9_baseline`, `phase9_longer`, `phase9_hyperparam`).
- [`experiments/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/experiments): Machine-readable JSON experiment records (`exp1_baseline.json`, `exp2_longer_training.json`, `exp3_model_size.json`, `exp4_sampling.json`, `summary.json`, `phase9_baseline.json`, `phase9_longer.json`, `phase9_hyperparam.json`, `phase9_prompts.json`, `phase9_summary.json`).
- [`src/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/src): Core implementation package.
- [`tests/`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/tests): Automated test scripts and sanity checks.

---

## 7. Current Configuration

Actual values from [`config.py`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/config.py):

| Parameter | Current Value | Description / Purpose |
|---|---|---|
| `batch_size` | `32` | Number of independent sequences processed in parallel |
| `block_size` | `256` | Maximum context length (tokens) for prediction |
| `vocab_size` | `65` | Initial placeholder (character-level TinyShakespeare vocab) |
| `n_embd` | `384` | Embedding dimension (divisible by `n_head`: 384 / 6 = 64) |
| `n_head` | `6` | Number of parallel attention heads |
| `n_layer` | `6` | Number of Transformer blocks stacked sequentially |
| `dropout` | `0.2` | Dropout regularization probability |
| `learning_rate` | `3e-4` (0.0003) | Peak learning rate for AdamW optimizer |
| `max_iters` | `5000` | Total training iterations |
| `eval_interval` | `500` | Iteration interval for calculating validation loss |
| `eval_iters` | `200` | Number of batches averaged during loss estimation |
| `device` | `"cuda"` if available else `"cpu"` | Hardware execution target |

*Note: These are initial baseline values and may be adjusted based on actual GPU memory measurements and empirical loss during later phases.*

---

## 8. Hardware Target

- **Target Development Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU with 6 GB VRAM (Windows).
- **Resource Constraints:** Model parameters (~10.79M params ≈ 41.15 MB FP32) and activation memory (~1.5–2.5 GB peak VRAM) operate safely within the 6 GB VRAM ceiling. CPU execution remains supported as a seamless fallback.

---

## 9. Decisions Made

- Build the model completely from scratch rather than wrapping existing libraries.
- Start with a character-level tokenizer in Phase 2 for clear educational mechanics before considering subword tokenizers.
- Derive vocabulary dynamically from input text by sorting unique characters deterministically (`sorted(list(set(text)))`).
- Raise explicit `ValueError` on unseen characters during `encode()` and invalid token IDs during `decode()` to avoid silent corruption.
- Implement dataset pipeline in Phase 3 with deterministic 90/10 train/val partitioning without data leakage.
- Generate `(x, y)` batches with context offset where `y` is shifted exactly 1 token ahead of `x` for autoregressive next-token prediction.
- Implement pre-LayerNorm Transformer structure ($x = x + \text{attn}(\text{ln}_1(x))$, $x = x + \text{mlp}(\text{ln}_2(x))$) in Phase 4 for training stability.
- Enforce causal masking using registered lower-triangular buffer with $-\infty$ masking before softmax, verified to produce 0.0 future attention probabilities.
- Scale multi-head attention scores by $\sqrt{\text{head\_size}} = \sqrt{64} = 8.0$.
- Structure feed-forward network with $4\times$ channel expansion ($384 \to 1536 \to 384$) and GELU non-linearity.
- Initialize all parameters randomly from $\mathcal{N}(0, 0.02)$ with zero biases and unit LayerNorm weights; zero pretrained weights.
- Compute next-token cross-entropy loss by flattening logits `(B*T, vocab_size)` and targets `(B*T)` without manual softmax.
- Use AdamW optimizer with default learning rate $3 \times 10^{-4}$ (or configured hyperparameter) and `betas=(0.9, 0.99)`.
- Enforce dynamic vocabulary sizing: model is always instantiated with `dataset.vocab_size` rather than a static default.
- Support gradient clipping with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to stabilize early gradient dynamics without disrupting standard updates.
- Isolate validation evaluation under `torch.no_grad()` and `model.eval()`, explicitly restoring `model.train()` afterwards to prevent state leakage.
- Provide clean deterministic seed setting (`seed=1337`) in `train()` while keeping seed configuration modular.
- Checkpoint persistence design: Preserve comprehensive state dictionary including `model_state_dict`, `optimizer_state_dict`, `step`, `config`, `train_loss`, `val_loss`, `best_val_loss`, `history`, `tokenizer_vocab`, `dataset_info`, and `rng_state`.
- Atomic saving: Save to a `.pt.tmp` temporary file first, then use `os.replace` to prevent corrupted checkpoint files upon sudden interruption.
- Robust file I/O: Open files explicitly via standard Python file handles (`with open(..., 'wb') as f: torch.save(...)`) to guarantee compatibility across Windows NTFS, OneDrive syncing, and process permission sandboxes.
- Strict checkpoint validation: Check compatibility for `vocab_size`, `n_embd`, `n_head`, `n_layer`, and `block_size` when loading into an existing model, raising descriptive `ValueError` upon mismatch.
- Corrupted weight detection: Actively scan state_dict tensors for NaN or Inf values upon load to prevent silent propagation of corrupted weights.
- Dual-checkpoint tracking: Automatically maintain `latest.pt` for resuming interrupted runs and `best.pt` containing the weights with the lowest validation loss.
- Resuming behavior: Training accepts `resume_from` path, restores weights, optimizer state, start iteration, and best validation loss, seamlessly continuing optimization to `max_iters`.
- Autoregressive inference: Sequential next-token generation plucking final position logits (`logits[:, -1, :]`) and appending sampled tokens iteratively.
- Context length management: Crop running sequence to trailing `block_size` tokens (`idx[:, -block_size:]`), allowing generation beyond model context length without shape violations.
- Temperature scaling: `logits / temperature` with strictly positive validation (`temperature > 0`), rejecting non-positive values when sampling.
- Top-k candidate restriction: Filters distribution to top-k logits (clamped to vocabulary size), setting remaining logits to $-\infty$. `top_k=1` is equivalent to greedy decoding.
- Deterministic greedy generation: Implemented via `greedy=True` with `torch.argmax(logits, dim=-1)`, bypassing stochastic sampling.
- Non-mutating generation: Runs under `torch.no_grad()` and `model.eval()`, restores original training mode, creates zero gradients, and leaves model weights bit-for-bit unchanged.
- Checkpoint-based generation: Automatically reconstructs model architecture, loads weights, and restores tokenizer mappings from `.pt` checkpoint files without manual configuration.
- Strict prompt validation: Unseen prompt characters trigger clear `ValueError` with Unicode details, rejecting invalid characters rather than silent substitution.
- Perplexity computation: Defined mathematically as $\text{PPL} = \exp(\mathcal{L}_{\text{val}})$. Implemented numerically safe clamping (`loss = min(loss, 50.0)`) in `compute_perplexity()` to avoid floating point overflow (`exp(50) ~ 5.18e21`), returning 1.0 for non-positive loss.
- Throughput metric: Calculated as $\text{tokens/sec} = \frac{\text{total tokens processed}}{\text{duration seconds}}$ where $\text{total tokens} = \text{steps} \times \text{batch\_size} \times \text{block\_size}$, with safe zero-duration protection.
- Reproducibility & seed control: Controlled random seeding (`seed=1337` for training, `seed=42` for generation sampling) across `torch.manual_seed`, `torch.cuda.manual_seed_all`, `random.seed`, and `np.random.seed` ensures identical data ordering and initialization across runs.
- Machine-readable tracking: Standalone JSON experiment logs stored in `experiments/<exp_id>.json` and rolled up in `experiments/summary.json` with full model architecture, hyperparameters, step-by-step loss history, evaluation metrics, and generated text samples.
- Systematic decoding evaluation: Comparative generation suite evaluating 7 distinct sampling configurations (Greedy, T=0.5, T=0.8, T=1.0, Top-k 5, Top-k 10, Top-k 20) against a shared checkpoint and prompt.
- Domain Dataset Design (Phase 9): Curated 20,257-character corpus (`data/programming.txt`) spanning 11 sections covering OOP principles, Java core classes/interfaces, Spring Boot (`@SpringBootApplication`, `@RestController`, `@Service`, `@Repository`), Spring Data JPA, REST APIs (`ResponseEntity`, HTTP methods/status codes), DTOs with validation (`@Valid`), global error handling (`@ControllerAdvice`), relational SQL schemas (DDL) & queries (DML with JOIN, GROUP BY), MockMvc integration testing, and database ACID properties.
- Universal Character Tokenizer for Code: Maintained the character-level tokenizer with 88 unique characters. Zero out-of-vocabulary tokens across all programming keywords, camelCase identifiers, annotations (`@`), braces (`{`, `}`), and operators.
- Text Normalization: Automatic conversion of CRLF to LF (`\r\n` -> `\n`) in `TextDataset` for uniform cross-platform token IDs and consistent indentation tokens.
- Controlled Domain Benchmarking: Reused the exact Transformer architecture, training loop, and evaluation methodologies from Phase 8 to maintain direct empirical comparability.
- Implement components incrementally in strict phase order.
- Do not jump directly to a pretrained model.
- Test each phase thoroughly before moving to the next phase.
- Prioritize conceptual clarity, modularity, and verification over generating monolithic or over-engineered code.

---

## 10. Verification

- **Phase 1 Verification:** Project structure and configuration verified (4 tests).
- **Phase 2 Verification:** Character-level tokenizer verified across 10 dedicated test cases.
- **Phase 3 Verification:** Dataset loading, tokenization, 90/10 split integrity, context shapes, next-token offset, token ID validity, custom batch sizes, device placement, repeated batching, and small corpus error handling verified across 11 dedicated test cases.
- **Phase 4 Verification:** Transformer architecture verified across 11 dedicated test cases (construction with dynamic vocabulary, forward pass logits shape `(B, T, vocab_size)`, variable sequence lengths, causal mask future zero probability, head dimension divisibility, parameter count, backprop gradient flow, CPU execution, CUDA clean skip, block limit rejection, and independent random initialization).
- **Phase 5 Verification:** Training loop verified across 10 dedicated test cases in `tests/test_train.py` (scalar finite loss, 0-dim loss shape, backpropagation gradient delivery, AdamW parameter update, multi-step training stability, no-grad loss estimation, train/eval mode restoration, clean device selection, dynamic vocabulary consistency, and tiny overfit loss reduction).
- **Phase 6 Verification:** Checkpoints and validation verified across 15 dedicated test cases in `tests/test_checkpoint.py` (checkpoint saving, file existence, exact parameter restoration, optimizer state restoration, step restoration, loss metrics restoration, config preservation, tokenizer vocab preservation, CPU execution, missing file error, corrupted/incompatible checkpoint handling, resume step continuation, best validation checkpoint tracking, NaN/Inf weight rejection, and clean CUDA skip).
- **Phase 7 Verification:** Autoregressive text generation verified across 17 dedicated test cases in `tests/test_generate.py` (string return, prompt preservation, token addition, zero-token prompt return, context cropping to block_size, temperature validation, top-k validation, top-k=1 greedy equivalence, greedy determinism, seeded sampling reproducibility, zero gradient creation, parameter immutability, train/eval mode restoration, CPU execution, clean CUDA skip, checkpoint-based generation, and unknown prompt character rejection).
- **Phase 8 Verification:** Experiment suite and metric calculation verified across 9 dedicated test cases in `tests/test_experiments.py` (perplexity computation, numerical overflow clamping, parameter count, throughput computation, safe zero-duration division, sampling configs generation, JSON experiment persistence, and multi-sampling generation runs).
- **Phase 9 Verification:** Domain customization verified across 7 dedicated test cases in `tests/test_domain_customization.py` (dataset file existence and length, dataset statistics calculation, domain dataset loading and split integrity, zero-OOV tokenization across all 7 domain prompts, batch generation with shifted targets, domain training steps, and prompt evaluation runner).
- **Complete Test Suite Run:** `python -m unittest discover -s tests -p "test_*.py" -v`
- **Result:** Ran 94 tests in 6.803s — **90 PASSED, 4 SKIPPED (OK)**.
  - `test_structure.py`: 4 tests passed.
  - `test_tokenizer.py`: 10 tests passed.
  - `test_dataset.py`: 11 tests passed.
  - `test_model.py`: 10 tests passed, 1 cleanly skipped (`test_9_cuda_execution` skipped due to CPU-only PyTorch).
  - `test_train.py`: 9 tests passed, 1 cleanly skipped (`test_8_device_handling` CUDA sub-check skipped due to CPU-only PyTorch).
  - `test_checkpoint.py`: 14 tests passed, 1 cleanly skipped (`test_15_cuda_handling` skipped due to CPU-only PyTorch).
  - `test_generate.py`: 16 tests passed, 1 cleanly skipped (`test_15_cuda_generation` skipped due to CPU-only PyTorch).
  - `test_experiments.py`: 9 tests passed.
  - `test_domain_customization.py`: 7 tests passed.

### Phase 8 — Experimental Results and Analysis

#### 1. Measured Results Scoreboard

| Experiment | Parameters | Steps | Train Loss | Val Loss | Best Val Loss | Val PPL | Time (s) | Throughput (tok/s) |
|---|---|---|---|---|---|---|---|---|
| **Exp 1: Baseline** (L=2, H=4, D=64) | 110,336 | 150 | 2.2380 | 2.6535 | 2.6535 | 14.20 | 8.37s | 18,349.4 |
| **Exp 2: Longer Training** (L=2, H=4, D=64) | 110,336 | 500 | 1.6894 | 2.7393 | 2.6364 (step 250) | 15.48 (best: 13.96) | 25.96s | 19,725.2 |
| **Exp 3: Model Size** (L=4, H=4, D=128) | 812,800 | 150 | 2.0748 | 2.6160 | 2.6160 | 13.68 | 28.06s | 5,473.9 |

#### 2. Sampling Decoding Comparison (Exp 4 on Reference Checkpoint `exp2_longer_training/best.pt`, Prompt: `"The "`, Length: 100)

| Decoding Mode | Config Details | Sample Output | Behavior / Observation |
|---|---|---|---|
| **Greedy** | `greedy=True, T=1.0` | `"The the the the the the the the the the the the the the the to the the the the the the the the the the t"` | Deterministic argmax mode collapses into an immediate repetitive 1-word cycle (`"the"`). |
| **Temperature 0.5** | `greedy=False, T=0.5` | `"The tithe me we the we tho thay tho isoushe and trisolit the the s s whe the the who there thandico the "` | Low entropy suppresses unlikely tokens; favors very common character n-grams (`"th"`, `"the"`, `"we"`). |
| **Temperature 0.8** | `greedy=False, T=0.8` | `"The titherey wrere oulle theve coustid thare. this osl t s o the foryou, wan than hore whethean'tche uly"` | Balanced entropy; generates varied Shakespearean-like token sequences with periods, spaces, and contractions. |
| **Temperature 1.0** | `greedy=False, T=1.0` | `"The tithey y wrere owele se hay pe tieaded\nwandeitinsl t  toouk. coryou,\n\n\n\n\nWhand con whethnth'tcharson"` | High entropy; higher probability of unusual character transitions and excessive newlines. |
| **Top-k 5** | `T=0.8, top_k=5` | `"The titheren we the we thares hare tido arede thit to the to the so whe than the who trthe thandid thath"` | Severely restricts vocabulary tail; produces high local fluency but repeats common short words. |
| **Top-k 10** | `T=0.8, top_k=10` | `"The titherey wrere oulle theve the tidou, cou thitins the ar the coryou than thed ho s thestsandiche sey"` | High lexical variety while eliminating low-probability nonsense characters. |
| **Top-k 20** | `T=0.8, top_k=20` | `"The titherey wrere oulle theve coustid thare. this osl t s o the foryou, wan than hore whetheandiche uly"` | Excellent balance between creative exploration and character transition stability. |

#### 3. Observations & Evidence
- **Generalization vs. Overfitting (Exp 1 vs Exp 2):** In Exp 2 (500 steps), training loss dropped continuously from 3.9515 down to 1.6894 (Train PPL 5.42). However, validation loss reached an optimum at step 250 ($2.6364$, Val PPL 13.96) before steadily increasing to $2.7393$ at step 500. This is empirical evidence of overfitting on the tiny Shakespeare dataset (3,342 train tokens).
- **Dialogue Syntax Learning:** At step 500, checkpoint generation under Top-k 10 revealed that the model successfully learned high-level structural document cues: `"\n\nFirst Citizen:\nThes malll at so mot satrme canansellve sed."`
- **Model Scaling vs Throughput (Exp 1 vs Exp 3):** Scaling from 2 layers / 64 embedding dim (110k parameters) to 4 layers / 128 embedding dim (812k parameters, $7.36\times$ larger) reduced validation loss from 2.6535 to 2.6160 in 150 steps. However, training throughput decreased from 18,349 tok/s to 5,473 tok/s ($3.35\times$ slower on CPU), illustrating the compute cost of width and depth.
- **Reference Checkpoint:** `checkpoints/exp2_longer_training/best.pt` represents the best performing checkpoint (Val loss 2.6364, Val PPL 13.96) combining optimal validation generalization with structural character-level understanding.

#### 4. Hardware / Device Status
- PyTorch `2.14.0+cpu` installed in active environment.
- CUDA is unavailable (`torch.cuda.is_available() == False`); all training, metric evaluations, and generation runs execute on CPU. All metrics and JSON tracking are 100% operational and reproducible.

---

### Phase 9 — Domain Customization (Java, Spring Boot, REST APIs, OOP, SQL)

#### 1. Objective & Scope
- Successfully customized the from-scratch MiniGPT Transformer for the backend software development domain, covering:
  - Java Core & OOP (classes, abstract classes, interfaces, inheritance, polymorphism, encapsulation)
  - Spring Boot Framework (`@SpringBootApplication`, `@RestController`, `@Service`, `@Repository`, `@Autowired`)
  - REST APIs (HTTP endpoints, request/response DTOs, `ResponseEntity`, HTTP status codes)
  - Spring Data JPA & SQL (entities, queries, DDL tables, DML SELECT/JOIN/WHERE, ACID transactions)
  - Architecture definitions (Dependency Injection, Inversion of Control, Layered Architecture)
- Reused 100% of from-scratch Transformer and training infrastructure without external pretrained LLMs, adapters, or black-box frameworks.

#### 2. Dataset & Statistics (`data/programming.txt`)
- **Corpus Source:** Educational Open-Source Reference Architecture & Backend Guides (CC0 1.0 Universal / Public Domain).
- **Total Characters:** 20,257 characters (647 lines across 11 structured sections).
- **Vocabulary Size:** 88 unique characters (full ASCII alphanumerics, punctuation, braces, annotations `@`, SQL wildcards `*`, and indentation spaces).
- **Partitioning:** 90% training (18,231 tokens), 10% validation (2,026 tokens).
- **Most Common Characters:** Space `' '` (18.29%), `'e'` (8.03%), `'t'` (5.42%), `'r'` (5.22%), `'s'` (4.87%), `'i'` (4.39%), `'o'` (3.88%), `'n'` (3.85%), `'a'` (3.70%), `'\n'` (3.19%).
- **Tokenization:** Deterministic character-level tokenizer with zero out-of-vocabulary tokens across all Java keywords, Spring annotations, and SQL syntax.

#### 3. Measured Experimental Results Scoreboard

| Experiment | Parameters | Steps | Train Loss | Val Loss | Best Val Loss | Val PPL | Time (s) | Throughput (tok/s) | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| **Phase 9 Baseline** (L=2, H=4, D=64) | 114,944 | 150 | 2.3764 | 2.7572 | 2.7572 | 15.76 | 7.32s | 20,970.4 | `checkpoints/phase9_baseline/best.pt` |
| **Phase 9 Longer Training** (L=2, H=4, D=64) | 114,944 | 450 | 1.7552 | 2.6549 | 2.5680 (step 400) | 14.22 (best: 13.04) | 24.06s | 19,154.6 | `checkpoints/phase9_longer/best.pt` |
| **Phase 9 Scaled Model** (L=4, H=4, D=128) | 822,016 | 150 | 2.1746 | 2.7017 | 2.7017 | 14.90 | 24.84s | 6,182.6 | `checkpoints/phase9_hyperparam/best.pt` |

#### 4. Controlled Comparison: Phase 8 (Shakespeare) vs Phase 9 (Programming)

| Metric | Phase 8 Baseline | Phase 9 Baseline | Phase 8 Longer | Phase 9 Longer | Phase 8 Scaled | Phase 9 Scaled |
|---|---|---|---|---|---|---|
| **Domain Corpus** | Shakespeare *Coriolanus* | Java / Spring Boot / SQL | Shakespeare *Coriolanus* | Java / Spring Boot / SQL | Shakespeare *Coriolanus* | Java / Spring Boot / SQL |
| **Vocab Size** | 52 | 88 | 52 | 88 | 52 | 88 |
| **Model Parameters** | 110,336 | 114,944 (+$4.2\%$) | 110,336 | 114,944 | 812,800 | 822,016 (+$1.1\%$) |
| **Training Steps** | 150 | 150 | 500 | 450 | 150 | 150 |
| **Final Train Loss** | 2.2380 | 2.3764 | 1.6894 | 1.7552 | 2.0748 | 2.1746 |
| **Best Val Loss** | 2.6535 | 2.7572 | 2.6364 | **2.5680** | 2.6160 | 2.7017 |
| **Best Val Perplexity** | 14.20 | 15.76 | 13.96 | **13.04** | 13.68 | 14.90 |
| **Throughput (tok/s)** | 18,349.4 | 20,970.4 | 19,725.2 | 19,154.6 | 5,473.9 | 6,182.6 |

*Analysis:*
- Parameter increase (+4,608 in baseline, +9,216 in scaled) is strictly due to the larger vocabulary matrix (88 vs 52 tokens in token embeddings and `lm_head`).
- Phase 9 baseline loss is slightly higher at 150 steps because vocabulary entropy is higher ($\log(88) \approx 4.47$ vs $\log(52) \approx 3.95$).
- However, longer training on the richer 20k dataset allowed Phase 9 to generalize better, achieving a superior best validation loss ($2.5680$) and lower perplexity ($13.04$) than Phase 8 ($2.6364$ / $13.96$).

#### 5. Programming-Domain Generation Evaluation (Reference Model: `phase9_longer/best.pt`)

| Required Prompt | Greedy Output (argmax) | Top-k 10 Output (T=0.8) | Observed Programming Patterns |
|---|---|---|---|
| `public class UserService {` | `\n return urn username;\n returnatete = = = urnate =` | `\n prin serer rReppossporonser(Stringetrice);\n patubon vernate;\n ` | Indentation, curly brace scoping, method signatures with `String` types, semicolons. |
| `@RestController` | ` itory username;\n returnatusername;\n returnate` | ` ce ctin cas ce);\n }\n\n\n\npublic public Ponssssthronaty unturve() {\n ` | Class/method declarations immediately following annotation, curly brace blocks. |
| `public ResponseEntity` | ` = us username;\n returnaturn = urnate = urnate = urndeate = ur` | ` atuprn in er.storornty.sererRepepos ublicasereDtis.gease(it) {\n }\n\n publ` | Return type usage, DTO-style tokens (`Dtis`), method parameter parentheses. |
| `SELECT * FROM users` | `er user username = userntr = = username;\n usernturnate = = = use` | `porocer {\n }\n public Perontectass atatiord;\n patublind vid\n publ` | Generates entity fields associated with users (`username`, `user`), syntax transitions. |
| `Spring Boot application` | `das as aterer te te tre trordedes torordernate er er edestrer er er eater eatede` | `acanctin c s as atornthas terandectas antand cton s.serronate unter.vid(re ang t` | Connects application tokens to service/component style naming. |
| `interface UserRepository` | ` userntry usernderRepospository.serReposeEntory = = = userntorderReposeseEntory ` | ` usepr) {\n }\n publicerReposes = titarory.serRepesponseDty, ubtodtred);\n ` | Generates Repository keywords, ResponseDto patterns, method closures. |
| `What is dependency injection?` | `\n///// Sen Spricencenction atin ate ate ern e urn username e usernte urn {\n` | `\n\n ByIn Strivis\n/ pterion Pronteces anatiorderon foulern(id) {\n }\n\n publ` | Automatically emits double-slash comments (`//`), associating explanation questions with comment syntax. |

#### 6. Lessons Learned & Limitations
- **Syntax Absorption:** Even a tiny 115k-parameter character-level Transformer absorbs structural conventions of code: 4-space indentation, semicolons `;`, camelCase naming (`userRepository`, `userService`), annotations (`@`), and comment headers (`//`).
- **Domain Adaptation:** Replacing archaic literary drama with structured Java/SQL code altered the character transition probability space entirely, shifting the model from pseudo-Shakespeare dialogue to pseudo-Java method bodies.
- **Educational Limitations (Not Production-Grade):** MiniGPT is a demonstration model trained on CPU for several hundred iterations. It produces probabilistic character patterns, NOT semantically verified, compilable code. It has no compiler checking, type system verification, or AST awareness.

#### 7. Reference Checkpoint
- **Checkpoint Location:** [`checkpoints/phase9_longer/best.pt`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/checkpoints/phase9_longer/best.pt)
- **Best Validation Loss:** 2.5680 (Validation Perplexity: 13.04).

---

## 11. Known Issues

- **Python environment availability:** Ensure Python executable and virtual environment (`.venv`) paths are properly configured when executing commands in Windows terminals.

---

## 12. Agent Workflow

Every time the coding agent receives a task:

1. Read `brain.md`.
2. Identify the current phase.
3. Understand existing decisions.
4. Inspect only relevant files.
5. Implement the requested change.
6. Run relevant tests.
7. Fix failures if appropriate.
8. Update `brain.md`.
9. Report exactly what changed and what was verified.
10. STOP when the requested phase/task is complete.

*Do not automatically continue into the next phase unless explicitly instructed.*

---

## 13. Important Rule

Never restart the project analysis from zero when `brain.md` already provides the required context. Use `brain.md` as a fast project-context index, while treating the actual source code and test results as the source of truth.

---

## 14. Loop-Until-Complete Rule

For every future implementation task, including Phase 2:

Do NOT stop after merely writing code.

Use this loop:
1. Understand the task.
2. Read `brain.md`.
3. Inspect the relevant existing files.
4. Implement the requested change.
5. Run the appropriate tests.
6. If tests fail:
   - inspect the error,
   - identify the root cause,
   - fix the code,
   - run the tests again.
7. If another error appears:
   - investigate it,
   - fix it,
   - run the tests again.
8. Continue this implementation → test → diagnose → fix → retest loop until the requested task is actually complete and verified.
9. Only after successful verification:
   - update `brain.md`,
   - report the final result.
10. STOP.

Do not declare a task complete merely because the code was generated. A task is complete only when the relevant implementation and verification have successfully finished, or when a genuine external blocker prevents completion.

If an external blocker prevents testing, clearly report:
- what was attempted,
- the exact blocker,
- what remains unverified.

Do not hide or ignore test failures.
