"""
LoRA Training Pipeline for Phase 10 Pretrained LLM Adaptation.

Implements:
- Model and Tokenizer loading
- LoRA adapter integration using Hugging Face PEFT
- Parameter counting (total, trainable, frozen, percentage)
- Controlled SFT training loop with evaluation tracking
- Checkpoint persistence and reload verification
"""

import os
import time
import json
import torch
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from phase10_lora.src.config import Phase10Config
from phase10_lora.src.dataset import create_dataloaders


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculate and return exact parameter counts for total, trainable, and frozen."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": round(trainable_pct, 4)
    }


def setup_model_and_tokenizer(
    config: Phase10Config
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase, Dict[str, Any]]:
    """
    Load base causal LM, configure LoRA adapters, and verify trainable parameters.
    """
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load pretrained causal LM
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    
    # 3. Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        fan_in_fan_out=config.fan_in_fan_out,
        bias=config.bias
    )
    
    # 4 & 5. Attach LoRA adapters (freezes base model, enables LoRA parameters)
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(config.device)
    
    # 6. Verify trainable parameters
    param_stats = count_parameters(peft_model)
    return peft_model, tokenizer, param_stats


def evaluate_loss(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: str) -> float:
    """Evaluate mean cross-entropy loss over validation batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                count += 1
    model.train()
    return round(total_loss / count, 4) if count > 0 else float("nan")


def train_lora(
    config: Phase10Config
) -> Dict[str, Any]:
    """
    Execute controlled Phase 10 LoRA training experiment.
    
    Returns complete dictionary of training metrics, parameter counts, and paths.
    """
    # Set seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.experiment_dir, exist_ok=True)

    # 1. Setup model & tokenizer
    model, tokenizer, param_stats = setup_model_and_tokenizer(config)
    
    # 2. Setup DataLoaders
    train_loader, val_loader, train_data, val_data = create_dataloaders(
        dataset_path=config.dataset_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        train_ratio=config.train_val_split,
        seed=config.seed,
        max_seq_len=config.max_seq_len
    )

    # 3. Optimizer on trainable LoRA parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Initial evaluation before training
    initial_val_loss = evaluate_loss(model, val_loader, config.device)

    epoch_records = []
    start_time = time.time()
    
    model.train()
    for epoch in range(1, config.num_train_epochs + 1):
        running_train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
            
        epoch_train_loss = running_train_loss / train_batches if train_batches > 0 else 0.0
        epoch_val_loss = evaluate_loss(model, val_loader, config.device)
        
        epoch_records.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(epoch_val_loss, 4)
        })

    training_time_seconds = round(time.time() - start_time, 2)
    final_train_loss = epoch_records[-1]["train_loss"] if epoch_records else None
    final_val_loss = epoch_records[-1]["val_loss"] if epoch_records else None

    # Save LoRA adapter checkpoint
    save_path = os.path.join(config.checkpoint_dir, config.adapter_checkpoint_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save training metadata
    results = {
        "config": config.to_dict(),
        "parameter_stats": param_stats,
        "dataset_stats": {
            "total_examples": len(train_data) + len(val_data),
            "train_examples": len(train_data),
            "val_examples": len(val_data)
        },
        "initial_val_loss": initial_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "training_time_seconds": training_time_seconds,
        "epoch_records": epoch_records,
        "checkpoint_path": save_path
    }

    result_filename = (
        "phase10_lora_results.json"
        if config.adapter_checkpoint_name == "distilgpt2_lora_programming"
        else f"phase10_lora_results_{config.adapter_checkpoint_name}.json"
    )
    result_file = os.path.join(config.experiment_dir, result_filename)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def load_fine_tuned_model(
    checkpoint_dir: str,
    base_model_name: str = "distilbert/distilgpt2",
    device: str = "cpu"
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """Load base model and attach fine-tuned LoRA adapters from checkpoint directory."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    peft_model.to(device)
    peft_model.eval()
    return peft_model, tokenizer
