"""
End-to-end execution script for Phase 10:
1. Setup and parameter efficiency verification
2. Pretrained Base Model evaluation on benchmark prompts
3. Controlled LoRA fine-tuning experiment
4. Post-adaptation evaluation on identical benchmark prompts
5. Side-by-side comparison table generation
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from phase10_lora.src.config import Phase10Config
from phase10_lora.src.train import train_lora, setup_model_and_tokenizer, count_parameters
from phase10_lora.src.evaluate import run_benchmark, BENCHMARK_PROMPTS
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    print("=" * 70)
    print("PHASE 10: PRETRAINED LLM FINE-TUNING WITH LoRA")
    print("=" * 70)

    config = Phase10Config(
        model_name="distilbert/distilgpt2",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        batch_size=4,
        learning_rate=5e-4,
        num_train_epochs=8,
        seed=42
    )

    print(f"\n[1/5] Inspecting Base Model and LoRA Configuration...")
    print(f"Base Model: {config.model_name}")
    print(f"LoRA Rank (r): {config.r}")
    print(f"LoRA Alpha: {config.lora_alpha}")
    print(f"LoRA Dropout: {config.lora_dropout}")
    print(f"Target Modules: {config.target_modules}")

    # Inspect parameter counts
    peft_model, tokenizer, param_stats = setup_model_and_tokenizer(config)
    print("\nParameter Efficiency Verification:")
    print(f"  Total parameters:     {param_stats['total_parameters']:,}")
    print(f"  Trainable parameters: {param_stats['trainable_parameters']:,}")
    print(f"  Frozen parameters:    {param_stats['frozen_parameters']:,}")
    print(f"  Trainable percentage: {param_stats['trainable_percentage']:.4f}%")

    print("\n[2/5] Evaluating Base Pretrained Model (BEFORE fine-tuning)...")
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)
    base_results = run_benchmark(base_model, tokenizer, prompts=BENCHMARK_PROMPTS, temperature=0.0, max_new_tokens=100)
    for i, res in enumerate(base_results, 1):
        print(f"\nPrompt {i}: {res['prompt']}")
        print(f"Base Model Output: {res['response'][:140]}...")

    print("\n[3/5] Starting LoRA Fine-Tuning...")
    train_results = train_lora(config)
    print(f"Training completed in {train_results['training_time_seconds']}s")
    print(f"Initial Val Loss: {train_results['initial_val_loss']:.4f}")
    print(f"Final Train Loss: {train_results['final_train_loss']:.4f}")
    print(f"Final Val Loss:   {train_results['final_val_loss']:.4f}")
    print(f"Checkpoint saved: {train_results['checkpoint_path']}")

    print("\n[4/5] Evaluating LoRA Fine-Tuned Model (AFTER fine-tuning)...")
    lora_model = PeftModel.from_pretrained(base_model, train_results['checkpoint_path']).to(config.device)
    lora_results = run_benchmark(lora_model, tokenizer, prompts=BENCHMARK_PROMPTS, temperature=0.0, max_new_tokens=100)
    for i, res in enumerate(lora_results, 1):
        print(f"\nPrompt {i}: {res['prompt']}")
        print(f"LoRA Model Output: {res['response'][:140]}...")

    print("\n[5/5] Compiling Comparison and Saving Experiment Artifacts...")
    comparisons = []
    for b, l in zip(base_results, lora_results):
        comparisons.append({
            "prompt": b["prompt"],
            "base_response": b["response"],
            "lora_response": l["response"]
        })

    summary_file = os.path.join(config.experiment_dir, "phase10_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": config.model_name,
            "parameter_stats": param_stats,
            "training_results": train_results,
            "eval_comparisons": comparisons
        }, f, indent=2)

    print(f"Experiment summary written to {summary_file}")
    print("\nPhase 10 pipeline completed successfully!")


if __name__ == "__main__":
    main()
