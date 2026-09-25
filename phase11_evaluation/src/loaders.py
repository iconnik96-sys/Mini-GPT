"""
Model Loaders and Metadata Extractors for Phase 11 Evaluation.

Provides unified interfaces to load and inspect:
1. MiniGPT Baseline (Phase 8 from-scratch Transformer)
2. MiniGPT Programming (Phase 9 domain from-scratch Transformer)
3. DistilGPT-2 Base (Pretrained causal LM)
4. DistilGPT-2 + LoRA (Phase 10 parameter-efficient fine-tuned model)
"""

import os
import platform
import torch
from typing import Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from phase11_evaluation.config import ModelTargetConfig
from src.generate import load_from_checkpoint as load_minigpt_checkpoint
from phase10_lora.src.train import load_fine_tuned_model as load_distilgpt2_lora


def get_checkpoint_size_mb(path: str) -> float:
    """Calculate file or directory size on disk in megabytes."""
    if not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_bytes += os.path.getsize(fp)
    return round(total_bytes / (1024 * 1024), 2)


def load_system(
    model_cfg: ModelTargetConfig,
    device: str = "cpu"
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load model and tokenizer for a specified target configuration.
    
    Returns:
        (model, tokenizer, metadata_dict)
    """
    system_id = model_cfg.system_id
    
    if system_id in ["minigpt_baseline", "minigpt_programming"]:
        if not os.path.exists(model_cfg.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {model_cfg.checkpoint_path}")
        model, tokenizer = load_minigpt_checkpoint(model_cfg.checkpoint_path, device=device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = total_params  # Full from-scratch training
        frozen_params = 0
        ckpt_size = get_checkpoint_size_mb(model_cfg.checkpoint_path)
        
        if system_id == "minigpt_baseline":
            training_method = "From-scratch pretraining (Full parameter updates)"
            dataset_name = "Shakespeare Coriolanus (data/coriolanus.txt)"
            steps = 150
            training_time = "8.37s"
        else:
            training_method = "From-scratch domain pretraining (Full parameter updates)"
            dataset_name = "Java & Spring Boot & SQL (data/programming.txt)"
            steps = 450
            training_time = "24.06s"
            
        metadata = {
            "system_id": system_id,
            "display_name": model_cfg.display_name,
            "category": model_cfg.category,
            "architecture": "Custom Decoder-only Transformer (MiniGPT)",
            "tokenizer_type": "Character-level CharTokenizer",
            "vocabulary_size": tokenizer.vocab_size,
            "context_length": model.config.block_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_percentage": 100.0,
            "checkpoint_size_mb": ckpt_size,
            "training_method": training_method,
            "training_dataset": dataset_name,
            "training_steps": steps,
            "training_time": training_time,
            "hardware": "CPU (Intel 12-core)"
        }
        return model, tokenizer, metadata

    elif system_id == "distilgpt2_base":
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_cfg.checkpoint_path).to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        metadata = {
            "system_id": system_id,
            "display_name": model_cfg.display_name,
            "category": model_cfg.category,
            "architecture": "DistilGPT2 LM Head Model (6 layers, 12 heads, 768 dim)",
            "tokenizer_type": "Byte-level BPE (GPT2TokenizerFast)",
            "vocabulary_size": len(tokenizer),
            "context_length": 1024,
            "n_layer": 6,
            "n_head": 12,
            "n_embd": 768,
            "total_parameters": total_params,
            "trainable_parameters": 0,  # Not trained during evaluation/Phase 10
            "frozen_parameters": total_params,
            "trainable_percentage": 0.0,
            "checkpoint_size_mb": 334.35,  # Cached base weights size
            "training_method": "Pretrained on WebText by Hugging Face (Zero fine-tuning)",
            "training_dataset": "WebText / OpenWebText",
            "training_steps": "Pretrained",
            "training_time": "N/A (Pretrained weights)",
            "hardware": "CPU (Intel 12-core)"
        }
        return model, tokenizer, metadata

    elif system_id == "distilgpt2_lora":
        if not os.path.exists(model_cfg.checkpoint_path):
            raise FileNotFoundError(f"LoRA adapter checkpoint not found at: {model_cfg.checkpoint_path}")
        model, tokenizer = load_distilgpt2_lora(
            checkpoint_dir=model_cfg.checkpoint_path,
            base_model_name=model_cfg.base_model_name,
            device=device
        )
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        # In PEFT, adapter parameters have requires_grad or are in adapter modules
        adapter_params = 147456  # 6 layers * (8*768 + 2304*8) = 147,456
        frozen_params = total_params - adapter_params
        trainable_pct = round(adapter_params / total_params * 100.0, 4)
        adapter_size = get_checkpoint_size_mb(model_cfg.checkpoint_path)
        
        metadata = {
            "system_id": system_id,
            "display_name": model_cfg.display_name,
            "category": model_cfg.category,
            "architecture": "DistilGPT2 + Low-Rank Adaptation (LoRA r=8, alpha=16 on c_attn)",
            "tokenizer_type": "Byte-level BPE (GPT2TokenizerFast)",
            "vocabulary_size": len(tokenizer),
            "context_length": 1024,
            "n_layer": 6,
            "n_head": 12,
            "n_embd": 768,
            "total_parameters": total_params,
            "trainable_parameters": adapter_params,
            "frozen_parameters": frozen_params,
            "trainable_percentage": trainable_pct,
            "checkpoint_size_mb": adapter_size,
            "training_method": "Parameter-Efficient Fine-Tuning (LoRA)",
            "training_dataset": "Programming Instructions (phase10_lora/data/programming_instructions.json)",
            "training_steps": 48,  # 8 epochs * 6 batches
            "training_time": "178.44s",
            "hardware": "CPU (Intel 12-core)"
        }
        return model, tokenizer, metadata

    else:
        raise ValueError(f"Unknown system_id: {system_id}")
