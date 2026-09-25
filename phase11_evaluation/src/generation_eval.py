"""
Unified Generation Evaluator for Phase 11.

Executes prompts across:
- MiniGPT Character-level Transformer (Phase 8 Baseline & Phase 9 Programming)
- DistilGPT-2 BPE Causal LM (Base & LoRA)

Captures:
- Output text
- Execution status (success, OOV rejection, error)
- Latency (seconds)
- Token throughput (tokens/second)
- Heuristic quality & diversity metrics
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
import torch

from phase11_evaluation.config import ModelTargetConfig, Phase11Config
from phase11_evaluation.src.loaders import load_system
from phase11_evaluation.src.metrics import analyze_text
from src.generate import generate as minigpt_generate
from phase10_lora.src.generate import generate_response as distilgpt2_generate


def check_char_tokenizer_oov(tokenizer: Any, text: str) -> List[str]:
    """Check if text contains characters not present in the character tokenizer vocabulary."""
    stoi = getattr(tokenizer, "stoi", {})
    return [ch for ch in text if ch not in stoi]


def generate_single_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    model_cfg: ModelTargetConfig,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run generation for a single prompt under specified parameters.
    """
    start_time = time.time()
    
    if model_cfg.tokenizer_type == "character":
        # Character-level MiniGPT
        missing_chars = check_char_tokenizer_oov(tokenizer, prompt)
        if missing_chars:
            unique_missing = sorted(list(set(missing_chars)))
            return {
                "status": "oov_rejected",
                "output": "",
                "error": f"Character vocabulary mismatch: prompt contains characters not in vocabulary {unique_missing}",
                "generated_tokens": 0,
                "generation_time_sec": 0.0,
                "tokens_per_sec": 0.0,
                "metrics": {
                    "char_length": 0,
                    "word_count": 0,
                    "distinct_1": 0.0,
                    "distinct_2": 0.0,
                    "repetition_rate": 0.0,
                    "domain_keywords_found": 0,
                    "domain_keywords_list": [],
                    "syntax_score": 0
                }
            }
            
        greedy = (temperature <= 0.0)
        try:
            full_output = minigpt_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if not greedy else 1.0,
                top_k=top_k,
                greedy=greedy,
                device=device,
                seed=seed
            )
            elapsed = time.time() - start_time
            # Strip initial prompt
            gen_text = full_output[len(prompt):] if full_output.startswith(prompt) else full_output
            tokens_generated = len(gen_text)  # Char tokens
            tok_per_sec = round(tokens_generated / elapsed, 2) if elapsed > 0 else 0.0
            
            return {
                "status": "success",
                "output": gen_text.strip(),
                "full_output": full_output,
                "generated_tokens": tokens_generated,
                "generation_time_sec": round(elapsed, 4),
                "tokens_per_sec": tok_per_sec,
                "metrics": analyze_text(gen_text)
            }
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "error": f"{type(e).__name__}: {str(e)}",
                "generated_tokens": 0,
                "generation_time_sec": round(time.time() - start_time, 4),
                "tokens_per_sec": 0.0,
                "metrics": analyze_text("")
            }
            
    else:
        # BPE DistilGPT-2 (Base or LoRA)
        try:
            gen_text = distilgpt2_generate(
                model=model,
                tokenizer=tokenizer,
                instruction=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k is not None else 50,
                device=device
            )
            elapsed = time.time() - start_time
            # Estimate BPE tokens generated
            tok_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            tokens_generated = len(tok_ids)
            tok_per_sec = round(tokens_generated / elapsed, 2) if elapsed > 0 else 0.0
            
            return {
                "status": "success",
                "output": gen_text.strip(),
                "full_output": gen_text.strip(),
                "generated_tokens": tokens_generated,
                "generation_time_sec": round(elapsed, 4),
                "tokens_per_sec": tok_per_sec,
                "metrics": analyze_text(gen_text)
            }
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "error": f"{type(e).__name__}: {str(e)}",
                "generated_tokens": 0,
                "generation_time_sec": round(time.time() - start_time, 4),
                "tokens_per_sec": 0.0,
                "metrics": analyze_text("")
            }


def evaluate_system_prompts(
    model_cfg: ModelTargetConfig,
    prompts: List[Dict[str, str]],
    mode: str = "deterministic",
    config: Optional[Phase11Config] = None
) -> Dict[str, Any]:
    """
    Run evaluation suite for a single model across all prompts.
    """
    cfg = config or Phase11Config()
    device = cfg.device
    
    if mode == "deterministic":
        max_tokens = cfg.deterministic_max_new_tokens
        temp = cfg.deterministic_temperature
        top_k = None
    else:
        max_tokens = cfg.sampling_max_new_tokens
        temp = cfg.sampling_temperature
        top_k = cfg.sampling_top_k

    print(f"Loading system: {model_cfg.display_name}...")
    model, tokenizer, metadata = load_system(model_cfg, device=device)
    
    prompt_results = []
    for p in prompts:
        res = generate_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=p["prompt"],
            model_cfg=model_cfg,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            device=device,
            seed=cfg.seed
        )
        res["prompt_id"] = p["id"]
        res["category"] = p["category"]
        res["prompt"] = p["prompt"]
        prompt_results.append(res)

    # Compute aggregate statistics
    successful = [r for r in prompt_results if r["status"] == "success"]
    success_count = len(successful)
    
    if success_count > 0:
        avg_time = round(sum(r["generation_time_sec"] for r in successful) / success_count, 4)
        avg_tok_s = round(sum(r["tokens_per_sec"] for r in successful) / success_count, 2)
        avg_d1 = round(sum(r["metrics"]["distinct_1"] for r in successful) / success_count, 4)
        avg_d2 = round(sum(r["metrics"]["distinct_2"] for r in successful) / success_count, 4)
        avg_rep = round(sum(r["metrics"]["repetition_rate"] for r in successful) / success_count, 4)
        avg_kw = round(sum(r["metrics"]["domain_keywords_found"] for r in successful) / success_count, 2)
    else:
        avg_time, avg_tok_s, avg_d1, avg_d2, avg_rep, avg_kw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    aggregates = {
        "total_prompts": len(prompts),
        "successful_prompts": success_count,
        "failed_or_oov_prompts": len(prompts) - success_count,
        "average_generation_time_sec": avg_time,
        "average_tokens_per_sec": avg_tok_s,
        "average_distinct_1": avg_d1,
        "average_distinct_2": avg_d2,
        "average_repetition_rate": avg_rep,
        "average_domain_keywords_found": avg_kw
    }
    
    result_package = {
        "metadata": metadata,
        "mode": mode,
        "generation_config": {
            "max_new_tokens": max_tokens,
            "temperature": temp,
            "top_k": top_k,
            "device": device
        },
        "aggregates": aggregates,
        "prompt_results": prompt_results
    }
    
    # Save raw outputs
    os.makedirs(cfg.raw_results_dir, exist_ok=True)
    raw_file = os.path.join(cfg.raw_results_dir, f"{model_cfg.system_id}_{mode}.json")
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(result_package, f, indent=2)
        
    return result_package
