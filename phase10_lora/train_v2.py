"""
Training Script for Phase 10 v2 LoRA Model.

Fine-tunes DistilGPT-2 using LoRA (r=8, alpha=16) on the expanded, diverse
programming instruction dataset (94 examples).
Saves the resulting adapter checkpoint to:
phase10_lora/checkpoints/distilgpt2_lora_programming_v2/
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase10_lora.src.config import Phase10Config
from phase10_lora.src.train import train_lora


def main():
    print("=" * 70)
    print("      MiniGPT Phase 10 — LoRA Fine-Tuning v2 (Expanded Dataset)       ")
    print("=" * 70)

    config = Phase10Config.get_v2_config()
    print(f"Base Model:             {config.model_name}")
    print(f"LoRA Rank (r):          {config.r}")
    print(f"LoRA Alpha:             {config.lora_alpha}")
    print(f"Target Modules:         {config.target_modules}")
    print(f"Dataset Path:           {config.dataset_path}")
    print(f"Output Checkpoint:      {os.path.join(config.checkpoint_dir, config.adapter_checkpoint_name)}")
    print(f"Epochs:                 {config.num_train_epochs}")
    print(f"Batch Size:             {config.batch_size}")
    print(f"Learning Rate:          {config.learning_rate}")
    print(f"Device:                 {config.device.upper()}")
    print("-" * 70)

    start = time.time()
    results = train_lora(config)
    elapsed = round(time.time() - start, 2)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Time Taken:             {elapsed}s")
    print(f"Total Examples:         {results['dataset_stats']['total_examples']}")
    print(f"Train Examples:         {results['dataset_stats']['train_examples']}")
    print(f"Val Examples:           {results['dataset_stats']['val_examples']}")
    print(f"Initial Val Loss:       {results['initial_val_loss']}")
    print(f"Final Train Loss:       {results['final_train_loss']}")
    print(f"Final Val Loss:         {results['final_val_loss']}")
    print(f"Checkpoint Saved To:    {results['checkpoint_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
