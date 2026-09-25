"""
Model Manager for Phase 13 API.

Coordinates lazy loading, single active model memory management on CPU,
synchronous text generation, and Server-Sent Event streaming generators.
Reuses validated model loading & generation logic from Phase 12.
"""

import json
import threading
import time
from typing import Generator, List, Optional
import torch
from transformers import TextIteratorStreamer

from phase12_cli.config import GenerationSettings
from phase12_cli.src.model_registry import (
    REGISTRY,
    get_model_info as registry_get_model_info,
    is_valid_model,
    ModelMetadata,
)
from phase12_cli.src.model_loader import ModelSession
from phase12_cli.src.generator import (
    generate_text,
    generate_minigpt_stream,
    check_character_oov,
    GenerationResult,
)
from phase10_lora.src.dataset import format_instruction
from phase13_api_web.backend.schemas import (
    ModelMetadataResponse,
    GenerationRequest,
    GenerationResponse,
)


class ModelManager:
    """
    Central manager for model lifecycle and inference execution.
    Maintains a single active model session in CPU memory to avoid RAM exhaustion.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.session = ModelSession()
        self.lock = threading.Lock()

    def list_models(self) -> List[ModelMetadataResponse]:
        """Return metadata for all registered models without loading weights."""
        results = []
        for meta in REGISTRY.values():
            results.append(
                ModelMetadataResponse(
                    id=meta.id,
                    name=meta.display_name,
                    type=meta.model_type,
                    parameters=meta.parameter_count,
                    trainable_parameters=meta.trainable_parameters,
                    frozen_parameters=meta.frozen_parameters,
                    trainable_percentage=meta.trainable_percentage,
                    tokenizer=meta.tokenizer_type,
                    context_length=meta.context_length,
                    checkpoint=meta.checkpoint_path,
                    description=meta.description,
                    base_model_name=meta.base_model_name,
                )
            )
        return results

    def get_model_info(self, model_id: str) -> ModelMetadataResponse:
        """Return detailed metadata for a specific model ID."""
        if not is_valid_model(model_id):
            raise KeyError(f"Model '{model_id}' is not registered.")
        meta = registry_get_model_info(model_id)
        return ModelMetadataResponse(
            id=meta.id,
            name=meta.display_name,
            type=meta.model_type,
            parameters=meta.parameter_count,
            trainable_parameters=meta.trainable_parameters,
            frozen_parameters=meta.frozen_parameters,
            trainable_percentage=meta.trainable_percentage,
            tokenizer=meta.tokenizer_type,
            context_length=meta.context_length,
            checkpoint=meta.checkpoint_path,
            description=meta.description,
            base_model_name=meta.base_model_name,
        )

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        """
        Execute synchronous one-shot generation.
        Thread-safe execution using self.lock.
        """
        if not is_valid_model(req.model):
            raise ValueError(f"Unknown model '{req.model}'.")

        settings = GenerationSettings(
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            seed=req.seed,
            stream=False,
        )

        with self.lock:
            model, tokenizer, metadata = self.session.get_or_load(req.model, device=self.device)

            result: GenerationResult = generate_text(
                model=model,
                tokenizer=tokenizer,
                metadata=metadata,
                prompt=req.prompt,
                settings=settings,
                device=self.device,
            )

        return GenerationResponse(
            model=req.model,
            prompt=req.prompt,
            text=result.output,
            generated_tokens=result.tokens_generated,
            generation_time_seconds=result.elapsed_seconds,
            tokens_per_second=result.tokens_per_second,
            token_unit=result.token_unit,
        )

    def stream_generate(self, req: GenerationRequest) -> Generator[str, None, None]:
        """
        Stream generated tokens incrementally formatted as Server-Sent Events (SSE).
        Yields:
            data: {"token": "..."}\n\n
            ...
            data: {"done": true, ...}\n\n
        """
        if not is_valid_model(req.model):
            error_data = json.dumps({"error": f"Unknown model '{req.model}'."})
            yield f"data: {error_data}\n\n"
            return

        settings = GenerationSettings(
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            seed=req.seed,
            stream=True,
        )
        settings.validate()

        with self.lock:
            model, tokenizer, metadata = self.session.get_or_load(req.model, device=self.device)

            if req.seed is not None:
                torch.manual_seed(req.seed)

            start_time = time.time()
            tokens_count = 0
            token_unit = "char" if metadata.tokenizer_type == "character" else "bpe"

            if metadata.tokenizer_type == "character":
                # Check for OOV
                missing = check_character_oov(tokenizer, req.prompt)
                if missing:
                    err_msg = (
                        f"Prompt contains characters not supported by character model '{metadata.id}': {missing}"
                    )
                    error_payload = json.dumps({"error": err_msg})
                    yield f"data: {error_payload}\n\n"
                    return

                # Character stream generator
                tokens_yielded = []

                def on_char(char: str):
                    nonlocal tokens_count
                    tokens_count += 1
                    tokens_yielded.append(char)

                # Generate and yield tokens
                model.eval()
                tokens = tokenizer.encode(req.prompt)
                idx = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                greedy = (settings.temperature <= 0.0)

                with torch.no_grad():
                    for _ in range(settings.max_new_tokens):
                        idx_cond = (
                            idx
                            if idx.size(1) <= model.config.block_size
                            else idx[:, -model.config.block_size:]
                        )
                        logits = model(idx_cond)
                        logits = logits[:, -1, :]

                        if greedy:
                            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                        else:
                            scaled_logits = logits / settings.temperature
                            if settings.top_k is not None:
                                v, _ = torch.topk(
                                    scaled_logits,
                                    min(settings.top_k, scaled_logits.size(-1)),
                                )
                                scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                            idx_next = torch.multinomial(probs, num_samples=1)

                        next_token_id = idx_next.item()
                        char = tokenizer.decode([next_token_id])
                        tokens_count += 1
                        idx = torch.cat((idx, idx_next), dim=1)

                        token_payload = json.dumps({"token": char})
                        yield f"data: {token_payload}\n\n"

            else:
                # BPE models (distilgpt2, distilgpt2-lora)
                if metadata.model_type == "peft_lora":
                    formatted_prompt = format_instruction(req.prompt, None)
                else:
                    formatted_prompt = req.prompt

                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                do_sample = settings.temperature > 0.0
                gen_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask", None),
                    "max_new_tokens": settings.max_new_tokens,
                    "pad_token_id": (
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    "eos_token_id": tokenizer.eos_token_id,
                    "do_sample": do_sample,
                    "streamer": streamer,
                }
                if do_sample:
                    gen_kwargs["temperature"] = settings.temperature
                    if settings.top_k is not None:
                        gen_kwargs["top_k"] = settings.top_k
                    if settings.top_p is not None:
                        gen_kwargs["top_p"] = settings.top_p

                if settings.repetition_penalty is not None:
                    gen_kwargs["repetition_penalty"] = settings.repetition_penalty
                if settings.no_repeat_ngram_size is not None:
                    gen_kwargs["no_repeat_ngram_size"] = settings.no_repeat_ngram_size

                # Run generation in background thread
                thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()

                # Stream tokens as they emerge from iterator
                for new_text in streamer:
                    if new_text:
                        tokens_count += len(tokenizer.encode(new_text, add_special_tokens=False))
                        token_payload = json.dumps({"token": new_text})
                        yield f"data: {token_payload}\n\n"

                thread.join()

            elapsed = time.time() - start_time
            throughput = round(tokens_count / elapsed, 2) if elapsed > 0 else 0.0

            # Completion event
            completion_payload = json.dumps({
                "done": True,
                "model": req.model,
                "tokens_generated": tokens_count,
                "generation_time_seconds": round(elapsed, 4),
                "tokens_per_second": throughput,
                "token_unit": token_unit,
            })
            yield f"data: {completion_payload}\n\n"


# Global singleton instance for the FastAPI dependency
_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Dependency provider returning singleton ModelManager."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager(device="cpu")
    return _model_manager_instance
