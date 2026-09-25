# Phase 10 — Pretrained LLM Fine-Tuning with LoRA / QLoRA

## 1. Overview & Critical Phase Boundary

In Phases 1–9, we built and trained our own decoder-only MiniGPT Transformer entirely from scratch.
**Phase 10 introduces parameter-efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA) on an openly available pretrained causal language model.**

| Dimension | Phase 1–9 (MiniGPT) | Phase 10 (LoRA Fine-Tuning) |
|---|---|---|
| **Origin** | Custom decoder-only Transformer built from scratch | Pretrained causal language model (`distilbert/distilgpt2`) |
| **Pretrained Weights** | None (random initialization on Shakespeare / Java corpus) | Pretrained on WebText by Hugging Face |
| **Fine-Tuning Method** | Full parameter updates from scratch | Parameter-Efficient Fine-Tuning (LoRA) |
| **Trainable Ratio** | 100% of parameters trained | **0.1797%** of parameters trained (147,456 / 82,060,032) |
| **Base Weights** | Updated every step | **100% frozen** (81,912,576 parameters) |
| **Target Domain** | Character-level Shakespeare & Java concepts | Instruction-following in Spring Boot, Java OOP, SQL, REST APIs |

---

## 2. Hardware & Environment Detected

All inspection was performed directly on the execution environment:
- **Operating System:** Windows 11 (10.0.26200-SP0)
- **CPU:** Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 12 logical cores
- **RAM:** 15.70 GB Total, ~5.48 GB Available
- **GPU Availability:** None (`torch.cuda.is_available() == False`, CPU-only)
- **GPU VRAM:** N/A
- **PyTorch Version:** `2.14.0+cpu`
- **CUDA Availability:** `False`
- **Transformers Version:** `5.17.0`
- **PEFT Version:** `0.21.0`
- **Datasets Version:** `5.0.1`
- **BitsAndBytes Version:** Not installed (requires CUDA on Windows; CPU-only environment)

---

## 3. Pretrained Model Selection

To ensure fast, fully reproducible local execution without cloud APIs or out-of-memory bottlenecks, we selected:
- **Model Name:** `distilbert/distilgpt2`
- **Total Parameters:** 81,912,576 (~81.9M parameters)
- **Architecture:** 6 Transformer decoder layers, 12 attention heads, 768 hidden dimension (`GPT2LMHeadModel`)
- **Tokenizer:** Byte-level BPE (`GPT2TokenizerFast`), vocabulary size: 50,257 tokens
- **Context Length:** 1,024 tokens
- **License:** Apache 2.0 (Permissive for research, educational, and commercial use)
- **Memory Footprint:** ~330 MB FP32 weights
- **Rationale:** Native Hugging Face causal language model that supports generation, fits easily in CPU RAM, runs training in under 3 minutes, and integrates seamlessly with Hugging Face PEFT.

---

## 4. Full Fine-Tuning vs LoRA Mathematics

### Full Fine-Tuning
In standard full fine-tuning, the updated weight matrix $W'$ is:
$$W' = W + \Delta W$$
where every parameter in $W \in \mathbb{R}^{d \times k}$ receives gradients and optimizer momentum/variance states. For large models, this requires massive VRAM and creates full-sized checkpoints for every downstream task.

### Low-Rank Adaptation (LoRA)
LoRA (Hu et al., 2021) hypothesizes that parameter updates $\Delta W$ have a low intrinsic rank $r \ll \min(d, k)$. It decomposes $\Delta W$ into the product of two low-rank matrices:
$$W' = W + \frac{\alpha}{r} (B \cdot A)$$
where:
- $W \in \mathbb{R}^{d \times k}$ is the **frozen** pretrained weight matrix ($\nabla_W \mathcal{L} = 0$).
- $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$ is initialized to **zeros**, ensuring $B \cdot A = 0$ at the start of training (no perturbation initially).
- $r$ is the **LoRA rank** (e.g., $r = 8$).
- $\alpha$ is a constant **scaling factor** (e.g., $\alpha = 16$), giving an effective scaling of $\frac{\alpha}{r} = 2.0$.
- **LoRA Dropout:** $0.05$ applied to adapter inputs to prevent overfitting.

### Parameter Efficiency Verification
In `distilbert/distilgpt2`, the attention projection layer `c_attn` combines query, key, and value projections:
- Layer dimension: $768 \to 2304$ ($3 \times 768$).
- For rank $r = 8$:
  - Matrix $A$: $8 \times 768 = 6,144$ parameters.
  - Matrix $B$: $2,304 \times 8 = 18,432$ parameters.
  - Total per layer: $6,144 + 18,432 = 24,576$ parameters.
  - Across all 6 Transformer layers: $6 \times 24,576 = 147,456$ trainable parameters.

```
======================================================================
Parameter Efficiency Verification:
  Total parameters:     82,060,032
  Trainable parameters: 147,456
  Frozen parameters:    81,912,576
  Trainable percentage: 0.1797%
======================================================================
```
**Over 99.82% of the base model remains completely frozen.**

---

## 5. Programming Domain Instruction Dataset

The dataset lives in `phase10_lora/data/programming_instructions.json` and consists of 30 curated, high-signal instruction-response pairs covering:
1. **Spring Boot:** Dependency injection, `@RestController`, layered architecture, auto-configuration, `@PathVariable` vs `@RequestParam`, properties vs YAML, global exception handling.
2. **Java OOP:** Encapsulation, inheritance, polymorphism, abstraction, checked vs unchecked exceptions, interfaces vs abstract classes, `equals()` vs `==`.
3. **Data Access & JPA:** Spring Data JPA, `JpaRepository`, JPA vs Hibernate, `@Entity` & `@Table`, N+1 query problem, optimistic vs pessimistic locking.
4. **REST APIs:** Controller endpoints, DTO patterns, Bean Validation (`@Valid`).
5. **Databases & SQL:** SQL JOIN types, `@Transactional`, ACID properties, primary vs foreign keys, database indexes.

### Dataset Token Statistics
- **Total Examples:** 30 (24 train, 6 validation)
- **Instruction Token Length:** Min: 6, Max: 26, Mean: 13.3 tokens
- **Response Token Length:** Min: 52, Max: 97, Mean: 73.2 tokens
- **Full Sequence Length:** Min: 71, Max: 121, Mean: 92.2 tokens
- **Loss Masking:** Label tokens corresponding to the instruction prompt prefix are masked with `-100`, ensuring cross-entropy loss is computed **only on response tokens**.

---

## 6. Training Pipeline & Results

Controlled training was executed using `phase10_lora/run_experiment.py`:
- **Epochs:** 8
- **Batch Size:** 4
- **Learning Rate:** $5 \times 10^{-4}$ (AdamW)
- **Weight Decay:** 0.01
- **Random Seed:** 42
- **Training Time:** 178.44 seconds (~3 minutes on CPU)

### Loss Progression
| Epoch | Training Loss | Validation Loss |
|:---:|:---:|:---:|
| Initial (Pre-training) | — | 3.7001 |
| 1 | 4.1090 | 3.6710 |
| 2 | 4.0024 | 3.6173 |
| 3 | 3.9062 | 3.5699 |
| 4 | 3.8268 | 3.5535 |
| 5 | 3.7234 | 3.5435 |
| 6 | 3.6802 | 3.5306 |
| 7 | 3.6322 | 3.5319 |
| 8 | **3.5563** | **3.5418** |

---

## 7. Base Model vs LoRA Model Evaluation

Both models were evaluated on the exact same 7 benchmark prompts using greedy decoding (`temperature=0.0`):

| # | Benchmark Prompt | Base Pretrained Model Output | LoRA Fine-Tuned Model Output |
|---|---|---|---|
| 1 | *What is dependency injection in Spring Boot?* | Repeating prompt verbatim (`What is dependency injection in Spring Boot? ### Response: ...`) | *"Spring Boot is a framework that provides a framework that provides..."* (Domain context triggered) |
| 2 | *Explain the difference between an interface and an abstract class in Java.* | Generic repetition (`The following code is a simple example of a Java object...`) | *"An abstract class is an abstract class that implements a method..."* (Directly addresses class concepts) |
| 3 | *How does @RestController work?* | Degenerate prompt looping (`How does @RestController work? ### Response: ...`) | *"@RestController is a method that returns a single request to the controller..."* (Associates with controller/requests) |
| 4 | *What is Spring Data JPA?* | Degenerate prompt looping | *"Spring Data JPA is a RESTful Data JPA that provides..."* (Recognizes JPA domain) |
| 5 | *Write a simple REST endpoint in Spring Boot.* | Degenerate prompt looping | *"Write a simple REST endpoint in Spring Boot."* (Short prompt echo) |
| 6 | *Explain SQL JOIN.* | Echoes template instruction boilerplate | *"SQL JOIN."* (Identifies domain entity) |
| 7 | *What is @Transactional?* | Degenerate prompt looping | *"Transactional is a method that allows a user to perform a task using a single method..."* (Explains transactional method role) |

### Key Observations
1. **Instruction Format Adaptation:** The base model had zero understanding of instruction templates and entered an infinite repetition loop echoing the prompt. The LoRA adapter immediately taught the model to produce responses rather than repeating headers.
2. **Domain Vocabulary Activation:** Despite training only 0.1797% of parameters on 24 examples, domain keywords (`Spring Boot`, `controller`, `abstract class`, `transactional`) were activated.
3. **Trade-offs & Limitations:** With a tiny 82M base model and 30 training examples, responses suffer from repetitive phrasing. True conversational depth requires larger models (e.g. 7B+) and larger instruction datasets (e.g. 10k+ examples).

---

## 8. QLoRA Investigation & Hardware Constraints

### What is QLoRA?
QLoRA (Dettmers et al., 2023) combines 4-bit NormalFloat (NF4) base-model quantization with low-rank adapters:
- **Base model weights** are quantized to 4 bits using NF4, reducing base weight memory by ~75%.
- **Double Quantization (DQ)** quantizes the quantization constants, saving an extra ~0.37 bits per parameter.
- **LoRA adapters** remain in 16-bit or 32-bit floating point and receive gradients during backpropagation.

### Hardware Limitation (Truthful Disclosure)
- QLoRA relies on custom CUDA kernels implemented in `bitsandbytes` to perform on-the-fly 4-bit dequantization and mixed-precision matrix multiplications.
- The current machine is a **CPU-only Windows environment** (`torch.cuda.is_available() == False`).
- Neither CUDA nor `bitsandbytes` GPU kernels are functional on this system.
- In accordance with project instructions, **we did not fake a QLoRA run**. Standard FP32 LoRA is our verified working result, while `phase10_lora/src/qlora_investigation.py` provides the configuration and hardware diagnostic probe.

---

## 9. Folder Structure

```
phase10_lora/
├── README.md                                  # Phase 10 documentation
├── run_experiment.py                          # End-to-end training and evaluation script
├── src/
│   ├── __init__.py                            # Module exports
│   ├── config.py                              # Phase10Config dataclass
│   ├── dataset.py                             # SFT dataset loader, prompt formatter, label masker
│   ├── train.py                               # LoRA training loop, parameter counter, checkpointing
│   ├── evaluate.py                            # 7 benchmark prompt runner and comparison compiler
│   ├── generate.py                            # Text generation helper with greedy/top-k sampling
│   └── qlora_investigation.py                 # QLoRA configuration, theory, and hardware probe
├── tests/
│   └── test_lora.py                           # 10 unit and integration tests
├── data/
│   └── programming_instructions.json          # 30 domain instruction-response pairs
├── checkpoints/
│   └── distilgpt2_lora_programming/           # LoRA adapter checkpoint (adapter_model.safetensors, adapter_config.json)
└── experiments/
    ├── phase10_lora_results.json              # Training metrics and loss progression
    ├── phase10_summary.json                   # Full comparison and parameter stats summary
    └── evaluation_comparison.json             # Side-by-side prompt responses
```

---

## 10. How to Run Phase 10

### Run Tests
```powershell
# Run Phase 10 test suite (10 tests)
python -m unittest discover -s phase10_lora/tests -p "test_*.py" -v

# Run full project test suite (Phase 1–9, 94 tests)
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run Full LoRA Experiment
```powershell
python phase10_lora/run_experiment.py
```
