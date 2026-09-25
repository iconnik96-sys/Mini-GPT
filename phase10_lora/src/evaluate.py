"""
Evaluation and Comparison for Phase 10: Base Model vs LoRA Fine-Tuned Model.

Evaluates the exact 7 benchmark prompts:
1. What is dependency injection in Spring Boot?
2. Explain the difference between an interface and an abstract class in Java.
3. How does @RestController work?
4. What is Spring Data JPA?
5. Write a simple REST endpoint in Spring Boot.
6. Explain SQL JOIN.
7. What is @Transactional?
"""

import os
import json
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from phase10_lora.src.config import Phase10Config
from phase10_lora.src.generate import generate_response


BENCHMARK_PROMPTS = [
    "What is dependency injection in Spring Boot?",
    "Explain the difference between an interface and an abstract class in Java.",
    "How does @RestController work?",
    "What is Spring Data JPA?",
    "Write a simple REST endpoint in Spring Boot.",
    "Explain SQL JOIN.",
    "What is @Transactional?"
]


def run_benchmark(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str] = BENCHMARK_PROMPTS,
    temperature: float = 0.0,
    max_new_tokens: int = 120,
    device: str = "cpu"
) -> List[Dict[str, str]]:
    """Generate responses for benchmark prompts using deterministic greedy decoding."""
    results = []
    for prompt in prompts:
        resp = generate_response(
            model=model,
            tokenizer=tokenizer,
            instruction=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device
        )
        results.append({
            "prompt": prompt,
            "response": resp
        })
    return results


def run_evaluation_comparison(
    config: Phase10Config
) -> Dict[str, Any]:
    """
    Run complete side-by-side comparison between Base Pretrained Model and LoRA fine-tuned model.
    """
    print(f"Loading base model {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)
    
    print("Evaluating Base Model before fine-tuning...")
    base_results = run_benchmark(base_model, tokenizer, device=config.device)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, config.adapter_checkpoint_name)
    print(f"Loading LoRA adapted model from {checkpoint_path}...")
    lora_model = PeftModel.from_pretrained(base_model, checkpoint_path).to(config.device)
    
    print("Evaluating LoRA Model after fine-tuning...")
    lora_results = run_benchmark(lora_model, tokenizer, device=config.device)
    
    comparison = []
    for base_res, lora_res in zip(base_results, lora_results):
        comparison.append({
            "prompt": base_res["prompt"],
            "base_response": base_res["response"],
            "lora_response": lora_res["response"]
        })
        
    os.makedirs(config.experiment_dir, exist_ok=True)
    comp_file = os.path.join(config.experiment_dir, "evaluation_comparison.json")
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump({
            "base_model": config.model_name,
            "checkpoint_path": checkpoint_path,
            "comparisons": comparison
        }, f, indent=2)
        
    return {
        "base_model": config.model_name,
        "checkpoint_path": checkpoint_path,
        "comparisons": comparison
    }
