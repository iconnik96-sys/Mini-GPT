"""
Pydantic schemas for request and response validation in Phase 13 API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "MiniGPT API"
    version: str = "1.0.0"


class ModelMetadataResponse(BaseModel):
    id: str
    name: str
    type: str  # "from_scratch", "causal_lm", "peft_lora"
    parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percentage: float
    tokenizer: str  # "character", "bpe"
    context_length: int
    checkpoint: str
    description: str
    base_model_name: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[ModelMetadataResponse]


class GenerationRequest(BaseModel):
    model: str = Field(..., description="ID of the model to generate from")
    prompt: str = Field(..., min_length=1, description="Input prompt for generation")
    max_new_tokens: int = Field(60, ge=1, le=512, description="Maximum new tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=5.0, description="Sampling temperature (0.0 = greedy)")
    top_k: Optional[int] = Field(20, ge=1, le=1000, description="Top-k filtering threshold")
    seed: Optional[int] = Field(42, description="Random seed for deterministic generation")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only.")
        return v.strip()


class GenerationResponse(BaseModel):
    model: str
    prompt: str
    text: str
    generated_tokens: int
    generation_time_seconds: float
    tokens_per_second: float
    token_unit: str  # "char" or "bpe"


class StreamTokenEvent(BaseModel):
    token: str


class StreamCompletionEvent(BaseModel):
    done: bool = True
    model: str
    tokens_generated: int
    generation_time_seconds: float
    tokens_per_second: float
    token_unit: str


class ErrorResponse(BaseModel):
    detail: str
