"""
Unified Text Generator with Streaming Support for Phase 12 CLI.

Supports:
- Token-by-token streaming to terminal (via callback or stdout)
- Non-streaming one-shot generation
- Character-level MiniGPT and BPE-level DistilGPT-2 architectures
- Accurate latency, token count, and throughput calculation
- Clear Out-of-Vocabulary error diagnostics
"""

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, List
import torch
import torch.nn.functional as F
from transformers import TextStreamer

from phase12_cli.config import GenerationSettings
from phase12_cli.src.model_registry import ModelMetadata
from phase10_lora.src.dataset import format_instruction


@dataclass
class GenerationResult:
    output: str
    tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float
    token_unit: str  # "char" or "bpe"


def check_character_oov(tokenizer: Any, text: str) -> List[str]:
    """Return list of characters in text missing from the character tokenizer."""
    stoi = getattr(tokenizer, "stoi", {})
    return sorted(list(set(ch for ch in text if ch not in stoi)))


class StreamCapture:
    """Captures streamed tokens while printing them directly to terminal."""
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self.callback = callback
        self.buffer = []

    def on_token(self, token_str: str) -> None:
        self.buffer.append(token_str)
        if self.callback:
            self.callback(token_str)
        else:
            sys.stdout.write(token_str)
            sys.stdout.flush()

    def get_text(self) -> str:
        return "".join(self.buffer)


def generate_minigpt_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    settings: GenerationSettings,
    device: str = "cpu",
    stream_callback: Optional[Callable[[str], None]] = None
) -> Tuple[str, int]:
    """
    Autoregressively generate character tokens one by one with live streaming.
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_chars = []
    greedy = (settings.temperature <= 0.0)

    with torch.no_grad():
        for _ in range(settings.max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled_logits = logits / settings.temperature
                if settings.top_k is not None:
                    v, _ = torch.topk(scaled_logits, min(settings.top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(scaled_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            next_token_id = idx_next.item()
            char = tokenizer.decode([next_token_id])
            generated_chars.append(char)
            
            if stream_callback:
                stream_callback(char)
            elif settings.stream:
                sys.stdout.write(char)
                sys.stdout.flush()
                
            idx = torch.cat((idx, idx_next), dim=1)

    return "".join(generated_chars), len(generated_chars)


def generate_text(
    model: Any,
    tokenizer: Any,
    metadata: ModelMetadata,
    prompt: str,
    settings: GenerationSettings,
    device: str = "cpu",
    stream_callback: Optional[Callable[[str], None]] = None
) -> GenerationResult:
    """
    Master generation interface for all models in the CLI.
    """
    settings.validate()
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if settings.seed is not None:
        torch.manual_seed(settings.seed)

    start_time = time.time()

    if metadata.tokenizer_type == "character":
        # Check for OOV characters
        missing = check_character_oov(tokenizer, prompt)
        if missing:
            raise ValueError(
                f"Prompt contains characters not supported by character model '{metadata.id}': {missing}\n"
                f"Character models can only process characters present in their training corpus.\n"
                f"Tip: Try using 'distilgpt2-lora' which supports full universal BPE subwords."
            )

        output_text, token_count = generate_minigpt_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            settings=settings,
            device=device,
            stream_callback=stream_callback
        )
        token_unit = "char"

    else:
        # BPE DistilGPT-2 (Base or LoRA)
        if metadata.model_type == "peft_lora":
            formatted_prompt = format_instruction(prompt, None)
        else:
            formatted_prompt = prompt

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        do_sample = settings.temperature > 0.0
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": inputs.get("attention_mask", None),
            "max_new_tokens": settings.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        
        if do_sample:
            gen_kwargs["temperature"] = settings.temperature
            if settings.top_k is not None:
                gen_kwargs["top_k"] = settings.top_k
            if settings.top_p is not None:
                gen_kwargs["top_p"] = settings.top_p

        if settings.repetition_penalty is not None and settings.repetition_penalty >= 1.0:
            gen_kwargs["repetition_penalty"] = settings.repetition_penalty
        if settings.no_repeat_ngram_size is not None and settings.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = settings.no_repeat_ngram_size

        if settings.stream and stream_callback is None:
            # Use HuggingFace TextStreamer to stream directly to terminal
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)
        else:
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)

        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response portion
        marker = "### Response:\n"
        if marker in full_output:
            output_text = full_output.split(marker, 1)[1].strip()
        elif full_output.startswith(formatted_prompt):
            output_text = full_output[len(formatted_prompt):].strip()
        else:
            output_text = full_output.strip()

        token_count = len(tokenizer.encode(output_text, add_special_tokens=False))
        token_unit = "bpe"

    elapsed = time.time() - start_time
    tok_s = round(token_count / elapsed, 2) if elapsed > 0 else 0.0

    return GenerationResult(
        output=output_text,
        tokens_generated=token_count,
        elapsed_seconds=round(elapsed, 4),
        tokens_per_second=tok_s,
        token_unit=token_unit
    )
