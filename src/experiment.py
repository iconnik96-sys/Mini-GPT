"""Experimental Tracking, Evaluation Metrics, and Model Comparisons for MiniGPT.

Phase 8: Experiments and Improvements
-------------------------------------
Provides structured experimental runners and metric tracking:
1. Standard language modeling metrics:
   - Cross-entropy loss (train & val)
   - Perplexity: exp(loss) with numerical overflow protection
   - Trainable parameter counts
   - Wall-clock training duration and token throughput (tokens/sec)
2. Reproducible experiment pipelines:
   - Exp 1: Baseline model (150 iters, 2 layers, 64 embd, 4 heads)
   - Exp 2: Longer training (500 iters)
   - Exp 3: Model scaling comparison (4 layers, 128 embd, 4 heads)
   - Exp 4: Sampling evaluation across 7 decoding configurations (Greedy, T=0.5, T=0.8, T=1.0, Top-k 5/10/20)
3. Structured machine-readable results saved to `experiments/*.json` and summary table.

Zero external pretrained models, zero black-box evaluation frameworks.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from config import MiniGPTConfig
from src.dataset import (
    TextDataset,
    get_dataset,
    get_programming_dataset,
    get_dataset_statistics,
)
from src.generate import generate
from src.model import MiniGPT
from src.train import train


def compute_perplexity(loss: float) -> float:
    """Computes perplexity from cross-entropy loss: exp(loss).

    Perplexity represents the effective branching factor / uncertainty of the model.
    Applies numerical clamping to avoid floating-point overflow.

    Args:
        loss: Cross-entropy loss in nats (natural logarithm base e).

    Returns:
        float: Model perplexity (strictly >= 1.0).
    """
    if loss <= 0.0:
        return 1.0
    # Clamping loss to 50.0 prevents OverflowError on exp(50) ~ 5.18e21
    safe_loss = min(loss, 50.0)
    return math.exp(safe_loss)


def count_parameters(model: nn.Module) -> int:
    """Counts the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_tokens_per_sec(tokens: int, duration_sec: float) -> float:
    """Calculates training throughput in tokens processed per second."""
    return float(tokens) / max(float(duration_sec), 1e-6)


def get_default_sampling_configs() -> List[Dict[str, Any]]:
    """Returns the 7 standard decoding configurations required for Phase 8 Experiment 4."""
    return [
        {"name": "Greedy (argmax)", "greedy": True, "temperature": 1.0, "top_k": None},
        {"name": "Temperature 0.5 (low entropy)", "greedy": False, "temperature": 0.5, "top_k": None},
        {"name": "Temperature 0.8 (balanced)", "greedy": False, "temperature": 0.8, "top_k": None},
        {"name": "Temperature 1.0 (standard)", "greedy": False, "temperature": 1.0, "top_k": None},
        {"name": "Top-k 5 (T=0.8)", "greedy": False, "temperature": 0.8, "top_k": 5},
        {"name": "Top-k 10 (T=0.8)", "greedy": False, "temperature": 0.8, "top_k": 10},
        {"name": "Top-k 20 (T=0.8)", "greedy": False, "temperature": 0.8, "top_k": 20},
    ]


def run_experiment(
    name: str,
    cfg: MiniGPTConfig,
    max_iters: int,
    eval_interval: int = 25,
    eval_iters: int = 10,
    seed: int = 1337,
    dataset: Optional[TextDataset] = None,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "experiments",
    generation_prompts: Optional[List[str]] = None,
    sampling_configs: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executes a measurable language modeling experiment and saves results to JSON.

    Args:
        name: Unique identifier for the experiment (e.g. 'exp1_baseline').
        cfg: Architecture and hyperparameter configuration.
        max_iters: Total training steps.
        eval_interval: Iteration interval for validation loss tracking.
        eval_iters: Batches sampled during evaluation.
        seed: Random seed for deterministic reproducibility.
        dataset: TextDataset instance. Defaults to default Shakespeare corpus.
        checkpoint_dir: Directory where checkpoints are saved.
        results_dir: Directory where JSON metrics are saved.
        generation_prompts: List of prompt strings to test generation on.
        sampling_configs: List of sampling parameter dictionaries.
        verbose: If True, prints progress logs to stdout.

    Returns:
        Dict[str, Any]: Detailed experiment results and metrics dictionary.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ds = dataset if dataset is not None else get_dataset()

    # Determine default prompt compatible with tokenizer vocabulary
    if generation_prompts is not None:
        prompts = generation_prompts
    else:
        can_encode_default = all(ch in ds.tokenizer.stoi for ch in "The ")
        prompts = ["The "] if can_encode_default else [list(ds.tokenizer.stoi.keys())[0]]

    sample_cfgs = sampling_configs if sampling_configs is not None else [
        {"name": "Greedy", "greedy": True, "temperature": 1.0, "top_k": None},
        {"name": "Temp 0.8 Top-k 10", "greedy": False, "temperature": 0.8, "top_k": 10},
    ]

    exp_ckpt_dir = os.path.join(checkpoint_dir, name)
    os.makedirs(exp_ckpt_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print(f"STARTING EXPERIMENT: {name}")
        print(f"Architecture: L={cfg.n_layer} H={cfg.n_head} D={cfg.n_embd} | Context: {cfg.block_size} | Batch: {cfg.batch_size}")
        print(f"Steps: {max_iters} | Eval Interval: {eval_interval} | LR: {cfg.learning_rate} | Seed: {seed}")
        print("=" * 70)

    # 1. Execute Training
    train_res = train(
        dataset=ds,
        cfg=cfg,
        max_iters=max_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        checkpoint_dir=exp_ckpt_dir,
        save_checkpoints=True,
        seed=seed,
        verbose=verbose,
    )

    model = train_res["model"]
    param_count = count_parameters(model)
    final_train_loss = float(train_res["final_train_loss"])
    final_val_loss = float(train_res["final_val_loss"])
    best_val_loss = float(train_res["best_val_loss"])
    duration_sec = float(train_res["total_duration_sec"])
    tokens_processed = int(train_res["tokens_processed"])
    tokens_per_sec = float(train_res["tokens_per_sec"])

    # 2. Compute Perplexity
    train_ppl = compute_perplexity(final_train_loss)
    val_ppl = compute_perplexity(final_val_loss)
    best_val_ppl = compute_perplexity(best_val_loss)

    # 3. Generate Evaluation Samples
    generations: List[Dict[str, Any]] = []
    for prompt in prompts:
        for sc in sample_cfgs:
            gen_text = generate(
                model=model,
                tokenizer=ds.tokenizer,
                prompt=prompt,
                max_new_tokens=100,
                temperature=sc.get("temperature", 1.0),
                top_k=sc.get("top_k", None),
                greedy=sc.get("greedy", False),
                seed=seed,
            )
            generations.append({
                "config_name": sc.get("name", "sample"),
                "prompt": prompt,
                "greedy": sc.get("greedy", False),
                "temperature": sc.get("temperature", 1.0),
                "top_k": sc.get("top_k", None),
                "generated_text": gen_text,
            })

    # 4. Compile Structured Machine-Readable Record
    record: Dict[str, Any] = {
        "experiment_name": name,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "model_architecture": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "dropout": cfg.dropout,
            "trainable_parameters": param_count,
        },
        "training_hyperparameters": {
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "device": cfg.device,
        },
        "dataset_info": {
            "vocab_size": ds.vocab_size,
            "train_tokens": len(ds.train_data),
            "val_tokens": len(ds.val_data),
        },
        "metrics": {
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "train_perplexity": round(train_ppl, 4),
            "val_perplexity": round(val_ppl, 4),
            "best_val_perplexity": round(best_val_ppl, 4),
            "training_duration_sec": round(duration_sec, 2),
            "tokens_processed": tokens_processed,
            "tokens_per_second": round(tokens_per_sec, 1),
        },
        "loss_history": {str(k): v for k, v in train_res["history"].items()},
        "checkpoint_paths": {
            "best": train_res.get("best_checkpoint"),
            "latest": train_res.get("last_checkpoint"),
        },
        "generations": generations,
    }

    # 5. Persist JSON artifact
    json_path = os.path.join(results_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    if verbose:
        print("\n" + "-" * 70)
        print(f"EXPERIMENT {name} COMPLETE")
        print(f"Train Loss: {final_train_loss:.4f} (PPL: {train_ppl:.2f})")
        print(f"Val Loss:   {final_val_loss:.4f} (PPL: {val_ppl:.2f})")
        print(f"Best Val:   {best_val_loss:.4f} (PPL: {best_val_ppl:.2f})")
        print(f"Throughput: {tokens_per_sec:.1f} tokens/s | Duration: {duration_sec:.2f}s")
        print(f"Result Saved to: {json_path}")
        print("-" * 70)

    return record


def run_sampling_experiment(
    checkpoint_path: str,
    prompt: str = "The ",
    max_new_tokens: int = 100,
    results_dir: str = "experiments",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluates text generation across 7 decoding configurations on a single checkpoint.

    Configurations:
    1. Greedy (deterministic argmax)
    2. Temperature 0.5 (low entropy, conservative)
    3. Temperature 0.8 (moderate diversity)
    4. Temperature 1.0 (unscaled standard sampling)
    5. Top-k 5 (T=0.8, strict candidate cutoff)
    6. Top-k 10 (T=0.8, standard candidate cutoff)
    7. Top-k 20 (T=0.8, broad candidate cutoff)

    Args:
        checkpoint_path: Path to .pt checkpoint.
        prompt: Initial prompt text.
        max_new_tokens: Tokens to generate per sample.
        results_dir: Directory where JSON output is stored.
        seed: Random seed for stochastic samplers.
        verbose: If True, prints generated samples.

    Returns:
        Dict[str, Any]: Structured results containing outputs for each decoding setup.
    """
    os.makedirs(results_dir, exist_ok=True)
    from src.generate import load_from_checkpoint

    model, tokenizer = load_from_checkpoint(checkpoint_path, device="cpu")
    sampling_configs = get_default_sampling_configs()

    results: List[Dict[str, Any]] = []

    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: SAMPLING CONFIGURATIONS COMPARISON")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Prompt: {repr(prompt)} | Length: {max_new_tokens} tokens | Seed: {seed}")
        print("=" * 70)

    for sc in sampling_configs:
        out_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=sc.get("temperature", 1.0),
            top_k=sc.get("top_k", None),
            greedy=sc.get("greedy", False),
            seed=seed,
            device="cpu",
        )
        sample_record = {
            "setting": sc["name"],
            "greedy": sc.get("greedy", False),
            "temperature": sc.get("temperature", 1.0),
            "top_k": sc.get("top_k", None),
            "output": out_text,
        }
        results.append(sample_record)

        if verbose:
            print(f"\n[{sc['name']}]")
            print(out_text)

    record = {
        "experiment_name": "exp4_sampling",
        "timestamp": datetime.now().isoformat(),
        "checkpoint_evaluated": checkpoint_path,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "samples": results,
    }

    json_path = os.path.join(results_dir, "exp4_sampling.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record


def run_all_experiments(
    dataset: Optional[TextDataset] = None,
    results_dir: str = "experiments",
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs all 4 core Phase 8 experiments and creates a comparative summary.json."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ds = dataset if dataset is not None else get_dataset()

    # -------------------------------------------------------------------------
    # Experiment 1 — Baseline
    # -------------------------------------------------------------------------
    cfg_baseline = MiniGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=64,
        batch_size=16,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        learning_rate=1e-3,
        device="cpu",
    )
    exp1 = run_experiment(
        name="exp1_baseline",
        cfg=cfg_baseline,
        max_iters=150,
        eval_interval=25,
        eval_iters=10,
        seed=1337,
        dataset=ds,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # Experiment 2 — Longer Training (500 steps)
    # -------------------------------------------------------------------------
    cfg_longer = MiniGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=64,
        batch_size=16,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        learning_rate=1e-3,
        device="cpu",
    )
    exp2 = run_experiment(
        name="exp2_longer_training",
        cfg=cfg_longer,
        max_iters=500,
        eval_interval=50,
        eval_iters=10,
        seed=1337,
        dataset=ds,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # Experiment 3 — Model Size (4 layers, 128 embedding dim, 4 heads)
    # -------------------------------------------------------------------------
    cfg_larger = MiniGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=64,
        batch_size=16,
        n_embd=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        learning_rate=1e-3,
        device="cpu",
    )
    exp3 = run_experiment(
        name="exp3_model_size",
        cfg=cfg_larger,
        max_iters=150,
        eval_interval=25,
        eval_iters=10,
        seed=1337,
        dataset=ds,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # Experiment 4 — Sampling Evaluation on Best Checkpoint from Exp 2
    # -------------------------------------------------------------------------
    best_ckpt = exp2["checkpoint_paths"]["best"]
    exp4 = run_sampling_experiment(
        checkpoint_path=best_ckpt,
        prompt="The ",
        max_new_tokens=100,
        results_dir=results_dir,
        seed=42,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # Comparative Summary
    # -------------------------------------------------------------------------
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [
            {
                "id": "exp1_baseline",
                "name": "Baseline (150 iters, L=2, D=64)",
                "parameters": exp1["model_architecture"]["trainable_parameters"],
                "iterations": 150,
                "train_loss": exp1["metrics"]["final_train_loss"],
                "val_loss": exp1["metrics"]["final_val_loss"],
                "best_val_loss": exp1["metrics"]["best_val_loss"],
                "val_perplexity": exp1["metrics"]["val_perplexity"],
                "best_val_perplexity": exp1["metrics"]["best_val_perplexity"],
                "duration_sec": exp1["metrics"]["training_duration_sec"],
                "tokens_per_sec": exp1["metrics"]["tokens_per_second"],
            },
            {
                "id": "exp2_longer_training",
                "name": "Longer Training (500 iters, L=2, D=64)",
                "parameters": exp2["model_architecture"]["trainable_parameters"],
                "iterations": 500,
                "train_loss": exp2["metrics"]["final_train_loss"],
                "val_loss": exp2["metrics"]["final_val_loss"],
                "best_val_loss": exp2["metrics"]["best_val_loss"],
                "val_perplexity": exp2["metrics"]["val_perplexity"],
                "best_val_perplexity": exp2["metrics"]["best_val_perplexity"],
                "duration_sec": exp2["metrics"]["training_duration_sec"],
                "tokens_per_sec": exp2["metrics"]["tokens_per_second"],
            },
            {
                "id": "exp3_model_size",
                "name": "Larger Model (150 iters, L=4, D=128)",
                "parameters": exp3["model_architecture"]["trainable_parameters"],
                "iterations": 150,
                "train_loss": exp3["metrics"]["final_train_loss"],
                "val_loss": exp3["metrics"]["final_val_loss"],
                "best_val_loss": exp3["metrics"]["best_val_loss"],
                "val_perplexity": exp3["metrics"]["val_perplexity"],
                "best_val_perplexity": exp3["metrics"]["best_val_perplexity"],
                "duration_sec": exp3["metrics"]["training_duration_sec"],
                "tokens_per_sec": exp3["metrics"]["tokens_per_second"],
            },
        ],
        "sampling_evaluated_on": exp4["checkpoint_evaluated"],
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\n" + "=" * 90)
        print("PHASE 8 EXPERIMENT COMPARATIVE SCOREBOARD")
        print("=" * 90)
        header = f"{'Experiment':<32} | {'Params':<8} | {'Iters':<6} | {'Train Loss':<10} | {'Val Loss':<8} | {'Val PPL':<8} | {'Time (s)':<8} | {'Tok/s':<7}"
        print(header)
        print("-" * 90)
        for e in summary["experiments"]:
            row = (
                f"{e['name']:<32} | "
                f"{e['parameters']:<8,d} | "
                f"{e['iterations']:<6d} | "
                f"{e['train_loss']:<10.4f} | "
                f"{e['val_loss']:<8.4f} | "
                f"{e['val_perplexity']:<8.2f} | "
                f"{e['duration_sec']:<8.2f} | "
                f"{e['tokens_per_sec']:<7.1f}"
            )
            print(row)
        print("=" * 90)
        print(f"Summary JSON saved to: {summary_path}\n")

    return {
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
        "exp4": exp4,
        "summary": summary,
    }


def get_phase9_prompts() -> List[str]:
    """Returns the 7 domain-specific prompts required for Phase 9 evaluation."""
    return [
        "public class UserService {",
        "@RestController",
        "public ResponseEntity",
        "SELECT * FROM users",
        "Spring Boot application",
        "interface UserRepository",
        "What is dependency injection?",
    ]


def run_phase9_prompts_evaluation(
    checkpoint_path: str,
    results_dir: str = "experiments",
    seed: int = 42,
    max_new_tokens: int = 80,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluates all 7 required Phase 9 programming prompts across Greedy and Top-k sampling."""
    os.makedirs(results_dir, exist_ok=True)
    from src.generate import load_from_checkpoint

    model, tokenizer = load_from_checkpoint(checkpoint_path, device="cpu")
    prompts = get_phase9_prompts()

    modes = [
        {"name": "Greedy", "greedy": True, "temperature": 1.0, "top_k": None},
        {"name": "Top-k 10 (T=0.8)", "greedy": False, "temperature": 0.8, "top_k": 10},
    ]

    records: List[Dict[str, Any]] = []

    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 9: PROGRAMMING-DOMAIN PROMPT EVALUATION")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 80)

    for p in prompts:
        prompt_entry: Dict[str, Any] = {"prompt": p, "evaluations": []}
        if verbose:
            print(f"\n>>> PROMPT: {p}")
        for m in modes:
            out = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=p,
                max_new_tokens=max_new_tokens,
                temperature=m.get("temperature", 1.0),
                top_k=m.get("top_k", None),
                greedy=m.get("greedy", False),
                seed=seed,
                device="cpu",
            )
            prompt_entry["evaluations"].append({
                "mode": m["name"],
                "greedy": m.get("greedy", False),
                "temperature": m.get("temperature", 1.0),
                "top_k": m.get("top_k", None),
                "output": out,
            })
            if verbose:
                print(f"  [{m['name']}]\n  {repr(out)}")
        records.append(prompt_entry)

    result_payload = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "results": records,
    }
    json_path = os.path.join(results_dir, "phase9_prompts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload


def run_phase9_experiments(
    dataset: Optional[TextDataset] = None,
    results_dir: str = "experiments",
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executes the full suite of Phase 9 programming domain experiments."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ds = dataset if dataset is not None else get_programming_dataset()
    stats = get_dataset_statistics()

    # 1. Phase 9 Baseline (150 steps, L=2, H=4, D=64)
    cfg_base = MiniGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=64,
        batch_size=16,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.1,
        learning_rate=1e-3,
        device="cpu",
    )
    exp1 = run_experiment(
        name="phase9_baseline",
        cfg=cfg_base,
        max_iters=150,
        eval_interval=25,
        eval_iters=10,
        seed=1337,
        dataset=ds,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        generation_prompts=["public class UserService {", "SELECT * FROM users"],
        verbose=verbose,
    )

    # 2. Phase 9 Longer Training (450 steps, L=2, H=4, D=64)
    exp2 = run_experiment(
        name="phase9_longer",
        cfg=cfg_base,
        max_iters=450,
        eval_interval=50,
        eval_iters=10,
        seed=1337,
        dataset=ds,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        generation_prompts=["public class UserService {", "SELECT * FROM users"],
        verbose=verbose,
    )

    # 3. Phase 9 Scaled Model (150 steps, L=4, H=4, D=128)
    cfg_larger = MiniGPTConfig(
        vocab_size=ds.vocab_size,
        block_size=64,
        batch_size=16,
        n_embd=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        learning_rate=1e-3,
        device="cpu",
    )
    exp3 = run_experiment(
        name="phase9_hyperparam",
        cfg=cfg_larger,
        max_iters=150,
        eval_interval=25,
        eval_iters=10,
        seed=1337,
        dataset=ds,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        generation_prompts=["public class UserService {", "SELECT * FROM users"],
        verbose=verbose,
    )

    # 4. Prompt evaluation on best checkpoint (from exp2 or exp1)
    best_ckpt = exp2["checkpoint_paths"]["best"]
    prompt_res = run_phase9_prompts_evaluation(
        checkpoint_path=best_ckpt,
        results_dir=results_dir,
        seed=42,
        verbose=verbose,
    )

    # 5. Comparative summary (Phase 8 vs Phase 9)
    phase8_summary_path = os.path.join(results_dir, "summary.json")
    phase8_data = None
    if os.path.isfile(phase8_summary_path):
        try:
            with open(phase8_summary_path, "r", encoding="utf-8") as f:
                phase8_data = json.load(f)
        except Exception:
            pass

    summary = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 9 — Domain Customization (Java, Spring Boot, REST APIs, OOP, SQL)",
        "dataset_statistics": stats,
        "phase9_experiments": [
            {
                "id": "phase9_baseline",
                "name": "Phase 9 Baseline (150 iters, L=2, D=64)",
                "parameters": exp1["model_architecture"]["trainable_parameters"],
                "iterations": 150,
                "train_loss": exp1["metrics"]["final_train_loss"],
                "val_loss": exp1["metrics"]["final_val_loss"],
                "best_val_loss": exp1["metrics"]["best_val_loss"],
                "val_perplexity": exp1["metrics"]["val_perplexity"],
                "best_val_perplexity": exp1["metrics"]["best_val_perplexity"],
                "duration_sec": exp1["metrics"]["training_duration_sec"],
                "tokens_per_sec": exp1["metrics"]["tokens_per_second"],
            },
            {
                "id": "phase9_longer",
                "name": "Phase 9 Longer Training (450 iters, L=2, D=64)",
                "parameters": exp2["model_architecture"]["trainable_parameters"],
                "iterations": 450,
                "train_loss": exp2["metrics"]["final_train_loss"],
                "val_loss": exp2["metrics"]["final_val_loss"],
                "best_val_loss": exp2["metrics"]["best_val_loss"],
                "val_perplexity": exp2["metrics"]["val_perplexity"],
                "best_val_perplexity": exp2["metrics"]["best_val_perplexity"],
                "duration_sec": exp2["metrics"]["training_duration_sec"],
                "tokens_per_sec": exp2["metrics"]["tokens_per_second"],
            },
            {
                "id": "phase9_hyperparam",
                "name": "Phase 9 Scaled Model (150 iters, L=4, D=128)",
                "parameters": exp3["model_architecture"]["trainable_parameters"],
                "iterations": 150,
                "train_loss": exp3["metrics"]["final_train_loss"],
                "val_loss": exp3["metrics"]["final_val_loss"],
                "best_val_loss": exp3["metrics"]["best_val_loss"],
                "val_perplexity": exp3["metrics"]["val_perplexity"],
                "best_val_perplexity": exp3["metrics"]["best_val_perplexity"],
                "duration_sec": exp3["metrics"]["training_duration_sec"],
                "tokens_per_sec": exp3["metrics"]["tokens_per_second"],
            },
        ],
        "comparison_with_phase8": phase8_data.get("experiments") if phase8_data else None,
        "best_checkpoint": best_ckpt,
        "prompt_evaluations_file": "experiments/phase9_prompts.json",
    }

    summary_path = os.path.join(results_dir, "phase9_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("\n" + "=" * 90)
        print("PHASE 9 EXPERIMENT COMPARATIVE SCOREBOARD")
        print("=" * 90)
        header = f"{'Experiment':<36} | {'Params':<8} | {'Iters':<6} | {'Train Loss':<10} | {'Val Loss':<8} | {'Val PPL':<8} | {'Time (s)':<8} | {'Tok/s':<7}"
        print(header)
        print("-" * 90)
        for e in summary["phase9_experiments"]:
            row = (
                f"{e['name']:<36} | "
                f"{e['parameters']:<8,d} | "
                f"{e['iterations']:<6d} | "
                f"{e['train_loss']:<10.4f} | "
                f"{e['val_loss']:<8.4f} | "
                f"{e['val_perplexity']:<8.2f} | "
                f"{e['duration_sec']:<8.2f} | "
                f"{e['tokens_per_sec']:<7.1f}"
            )
            print(row)
        print("=" * 90)
        print(f"Summary JSON saved to: {summary_path}\n")

    return {
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
        "prompts": prompt_res,
        "summary": summary,
    }


def main() -> None:
    """CLI for running experiments."""
    parser = argparse.ArgumentParser(description="MiniGPT Experiments and Improvements CLI")
    parser.add_argument("--all", action="store_true", help="Run all 4 Phase 8 experiments")
    parser.add_argument("--phase9", action="store_true", help="Run all Phase 9 programming domain experiments")
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        choices=["baseline", "longer", "size", "sampling", "phase9_baseline", "phase9_longer", "phase9_hyperparam", "phase9_prompts", "phase9"],
        help="Experiment to run",
    )
    parser.add_argument("--results_dir", type=str, default="experiments", help="Output directory for results JSON")
    args = parser.parse_args()

    if args.phase9 or args.name == "phase9":
        run_phase9_experiments(results_dir=args.results_dir)
    elif args.all:
        run_all_experiments(results_dir=args.results_dir)
    elif args.name == "baseline":
        ds = get_dataset()
        cfg = MiniGPTConfig(vocab_size=ds.vocab_size, block_size=64, batch_size=16, n_embd=64, n_head=4, n_layer=2, learning_rate=1e-3)
        run_experiment("exp1_baseline", cfg=cfg, max_iters=150, results_dir=args.results_dir)
    elif args.name == "longer":
        ds = get_dataset()
        cfg = MiniGPTConfig(vocab_size=ds.vocab_size, block_size=64, batch_size=16, n_embd=64, n_head=4, n_layer=2, learning_rate=1e-3)
        run_experiment("exp2_longer_training", cfg=cfg, max_iters=500, results_dir=args.results_dir)
    elif args.name == "size":
        ds = get_dataset()
        cfg = MiniGPTConfig(vocab_size=ds.vocab_size, block_size=64, batch_size=16, n_embd=128, n_head=4, n_layer=4, learning_rate=1e-3)
        run_experiment("exp3_model_size", cfg=cfg, max_iters=150, results_dir=args.results_dir)
    elif args.name == "sampling":
        run_sampling_experiment("checkpoints/best.pt", results_dir=args.results_dir)
    elif args.name == "phase9_baseline":
        ds = get_programming_dataset()
        cfg = MiniGPTConfig(vocab_size=ds.vocab_size, block_size=64, batch_size=16, n_embd=64, n_head=4, n_layer=2, learning_rate=1e-3)
        run_experiment("phase9_baseline", cfg=cfg, max_iters=150, dataset=ds, results_dir=args.results_dir)
    elif args.name == "phase9_longer":
        ds = get_programming_dataset()
        cfg = MiniGPTConfig(vocab_size=ds.vocab_size, block_size=64, batch_size=16, n_embd=64, n_head=4, n_layer=2, learning_rate=1e-3)
        run_experiment("phase9_longer", cfg=cfg, max_iters=450, dataset=ds, results_dir=args.results_dir)
    elif args.name == "phase9_hyperparam":
        ds = get_programming_dataset()
        cfg = MiniGPTConfig(vocab_size=ds.vocab_size, block_size=64, batch_size=16, n_embd=128, n_head=4, n_layer=4, learning_rate=1e-3)
        run_experiment("phase9_hyperparam", cfg=cfg, max_iters=150, dataset=ds, results_dir=args.results_dir)
    elif args.name == "phase9_prompts":
        run_phase9_prompts_evaluation("checkpoints/phase9_longer/best.pt", results_dir=args.results_dir)


if __name__ == "__main__":
    main()
