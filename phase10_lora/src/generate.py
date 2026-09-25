"""
Text Generation utilities for Phase 10 Base and LoRA Fine-Tuned Models.

Supports:
- Standard prompt formatting with instruction template
- Greedy and temperature/top-k sampled decoding
- Response extraction
- Generation with either base model or LoRA adapted model
"""

from typing import Optional
import torch
from transformers import PreTrainedTokenizerBase
from phase10_lora.src.dataset import format_instruction


def generate_response(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    device: str = "cpu"
) -> str:
    """
    Generate a response to an instruction using either base or LoRA model.
    """
    model.eval()
    prompt = format_instruction(instruction, None)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    do_sample = temperature > 0.0
    
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask", None),
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
            
    if repetition_penalty is not None and repetition_penalty >= 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        
    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)
        
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the response text after the marker
    marker = "### Response:\n"
    if marker in full_output:
        return full_output.split(marker, 1)[1].strip()
    return full_output.strip()
