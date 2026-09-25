"""
Comparative evaluation script for Phase 10:
BEFORE (distilgpt2-lora v1) vs AFTER (distilgpt2-lora-v2).

Tests the 8 standard prompts specified in the user request:
1. What is Java?
2. Is Java a programming language?
3. What is a class in Java?
4. What is inheritance in Java?
5. Write a Java function to add two numbers.
6. Explain what a REST API is.
7. What is Spring Boot?
8. Hello
"""

import sys
import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase10_lora.src.dataset import format_instruction

TEST_PROMPTS = [
    "What is Java?",
    "Is Java a programming language?",
    "What is a class in Java?",
    "What is inheritance in Java?",
    "Write a Java function to add two numbers.",
    "Explain what a REST API is.",
    "What is Spring Boot?",
    "Hello"
]


def load_model(checkpoint_dir: str):
    base_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForCausalLM.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, use_v2_settings=True, max_new_tokens=60, seed=42):
    torch.manual_seed(seed)
    formatted = format_instruction(prompt, None)
    inputs = tokenizer(formatted, return_tensors="pt")
    
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if use_v2_settings:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
        })
    else:
        # V1 baseline settings: no repetition penalty, no ngram restriction, temperature 0.8
        gen_kwargs.update({
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
        })

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    elapsed = time.time() - t0
    
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:\n" in full:
        resp = full.split("### Response:\n")[-1].strip()
    else:
        resp = full[len(formatted):].strip()
        
    num_tokens = len(tokenizer.encode(resp, add_special_tokens=False))
    tok_s = round(num_tokens / elapsed, 1) if elapsed > 0 else 0.0
    return resp, num_tokens, round(elapsed, 3), tok_s


def main():
    v1_dir = os.path.join("phase10_lora", "checkpoints", "distilgpt2_lora_programming")
    v2_dir = os.path.join("phase10_lora", "checkpoints", "distilgpt2_lora_programming_v2")

    print("Loading V1 model...")
    v1_model, v1_tok = load_model(v1_dir)
    print("Loading V2 model...")
    v2_model, v2_tok = load_model(v2_dir)

    comparison_results = []

    print("\n" + "=" * 80)
    print("      HEAD-TO-HEAD EVALUATION: distilgpt2-lora (v1) vs distilgpt2-lora-v2       ")
    print("=" * 80)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")
        print("-" * 80)
        
        # V1 (BEFORE)
        v1_text, v1_toks, v1_time, v1_spd = generate(v1_model, v1_tok, prompt, use_v2_settings=False)
        print(f"--- BEFORE (distilgpt2-lora v1) [{v1_toks} bpe tokens, {v1_time}s, {v1_spd} tok/s] ---")
        print(v1_text)
        
        # V2 (AFTER)
        v2_text, v2_toks, v2_time, v2_spd = generate(v2_model, v2_tok, prompt, use_v2_settings=True)
        print(f"\n--- AFTER (distilgpt2-lora-v2) [{v2_toks} bpe tokens, {v2_time}s, {v2_spd} tok/s] ---")
        print(v2_text)
        print("-" * 80)

        comparison_results.append({
            "prompt": prompt,
            "v1": {
                "response": v1_text,
                "tokens": v1_toks,
                "latency_sec": v1_time,
                "throughput_tok_s": v1_spd
            },
            "v2": {
                "response": v2_text,
                "tokens": v2_toks,
                "latency_sec": v2_time,
                "throughput_tok_s": v2_spd
            }
        })

    out_json = os.path.join("phase10_lora", "experiments", "v1_vs_v2_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nSaved comparison records to: {out_json}")


if __name__ == "__main__":
    main()
