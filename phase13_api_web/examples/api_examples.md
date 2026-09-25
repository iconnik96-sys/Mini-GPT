# MiniGPT Studio — API Examples

This document provides curl, PowerShell, and JSON examples for all endpoints in the Phase 13 FastAPI backend.

Base URL: `http://127.0.0.1:8000`

---

## 1. Health Check

### Request
```bash
curl -X GET "http://127.0.0.1:8000/api/health"
```

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -Method Get
```

### Response (`200 OK`)
```json
{
  "status": "ok",
  "service": "MiniGPT Studio API",
  "version": "1.0.0"
}
```

---

## 2. Model Catalog (List Models)

### Request
```bash
curl -X GET "http://127.0.0.1:8000/api/models"
```

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/models" -Method Get
```

### Response (`200 OK`)
```json
{
  "models": [
    {
      "id": "distilgpt2-lora",
      "name": "DistilGPT-2 + LoRA (Phase 10)",
      "type": "peft_lora",
      "parameters": 82060032,
      "trainable_parameters": 147456,
      "frozen_parameters": 81912576,
      "trainable_percentage": 0.1797,
      "tokenizer": "bpe",
      "context_length": 1024,
      "checkpoint": "phase10_lora/checkpoints/distilgpt2_lora_programming",
      "description": "Pretrained DistilGPT-2 fine-tuned with LoRA (r=8, alpha=16) on Java & Spring Boot instructions.",
      "base_model_name": "distilbert/distilgpt2"
    },
    {
      "id": "minigpt-programming",
      "name": "MiniGPT Programming (Phase 9)",
      "type": "from_scratch",
      "parameters": 114944,
      "trainable_parameters": 114944,
      "frozen_parameters": 0,
      "trainable_percentage": 100.0,
      "tokenizer": "character",
      "context_length": 64,
      "checkpoint": "checkpoints/phase9_longer/best.pt",
      "description": "From-scratch decoder-only Transformer (2L, 4H, 64D) trained on Java, Spring Boot, and SQL.",
      "base_model_name": null
    },
    {
      "id": "distilgpt2",
      "name": "DistilGPT-2 Base (Pretrained)",
      "type": "causal_lm",
      "parameters": 81912576,
      "trainable_parameters": 0,
      "frozen_parameters": 81912576,
      "trainable_percentage": 0.0,
      "tokenizer": "bpe",
      "context_length": 1024,
      "checkpoint": "distilbert/distilgpt2",
      "description": "Pretrained 82M causal language model by Hugging Face on WebText without task adaptation.",
      "base_model_name": null
    },
    {
      "id": "minigpt-baseline",
      "name": "MiniGPT Baseline (Phase 8)",
      "type": "from_scratch",
      "parameters": 110336,
      "trainable_parameters": 110336,
      "frozen_parameters": 0,
      "trainable_percentage": 100.0,
      "tokenizer": "character",
      "context_length": 64,
      "checkpoint": "checkpoints/exp1_baseline/best.pt",
      "description": "From-scratch decoder-only Transformer (2L, 4H, 64D) trained on Shakespeare (Coriolanus).",
      "base_model_name": null
    }
  ]
}
```

---

## 3. Detailed Model Information

### Request
```bash
curl -X GET "http://127.0.0.1:8000/api/models/distilgpt2-lora"
```

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/models/distilgpt2-lora" -Method Get
```

### Response (`200 OK`)
```json
{
  "id": "distilgpt2-lora",
  "name": "DistilGPT-2 + LoRA (Phase 10)",
  "type": "peft_lora",
  "parameters": 82060032,
  "trainable_parameters": 147456,
  "frozen_parameters": 81912576,
  "trainable_percentage": 0.1797,
  "tokenizer": "bpe",
  "context_length": 1024,
  "checkpoint": "phase10_lora/checkpoints/distilgpt2_lora_programming",
  "description": "Pretrained DistilGPT-2 fine-tuned with LoRA (r=8, alpha=16) on Java & Spring Boot instructions.",
  "base_model_name": "distilbert/distilgpt2"
}
```

---

## 4. Synchronous Text Generation

### Request
```bash
curl -X POST "http://127.0.0.1:8000/api/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "minigpt-programming",
       "prompt": "Explain inheritance in Java",
       "max_new_tokens": 60,
       "temperature": 0.8,
       "top_k": 20,
       "seed": 42
     }'
```

```powershell
$body = @{
    model = "minigpt-programming"
    prompt = "Explain inheritance in Java"
    max_new_tokens = 60
    temperature = 0.8
    top_k = 20
    seed = 42
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/generate" -Method Post -ContentType "application/json" -Body $body
```

### Response (`200 OK`)
```json
{
  "model": "minigpt-programming",
  "prompt": "Explain inheritance in Java",
  "text": "p Dtere in e ier r = = coderrngetuse = wiler Serng Stringeat",
  "generated_tokens": 60,
  "generation_time_seconds": 0.0854,
  "tokens_per_second": 702.8,
  "token_unit": "char"
}
```

---

## 5. Server-Sent Events (SSE) Streaming

### Request
```bash
curl -N -X POST "http://127.0.0.1:8000/api/generate/stream" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "minigpt-programming",
       "prompt": "Explain inheritance in Java",
       "max_new_tokens": 10,
       "temperature": 0.8,
       "top_k": 20,
       "seed": 42
     }'
```

### Stream Event Output
```text
data: {"token": "p"}

data: {"token": " "}

data: {"token": "D"}

data: {"token": "t"}

data: {"token": "e"}

data: {"token": "r"}

data: {"token": "e"}

data: {"token": " "}

data: {"token": "i"}

data: {"token": "n"}

data: {"done": true, "model": "minigpt-programming", "tokens_generated": 10, "generation_time_seconds": 0.0381, "tokens_per_second": 262.5, "token_unit": "char"}
```

---

## 6. Error Responses

### Unknown Model ID (`400 Bad Request`)
```json
{
  "detail": "Unknown model 'gpt-4o'."
}
```

### Unknown Model Lookup (`404 Not Found`)
```json
{
  "detail": "Model 'nonexistent-model' was not found in the model registry."
}
```

### Out-of-Vocabulary Characters for Character Model (`400 Bad Request`)
```json
{
  "detail": "Prompt contains characters not supported by character model 'minigpt-baseline': ['0', '8']\nCharacter models can only process characters present in their training corpus.\nTip: Try using 'distilgpt2-lora' which supports full universal BPE subwords."
}
```

### Pydantic Validation Error (`422 Unprocessable Entity`)
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "temperature"],
      "msg": "Input should be greater than or equal to 0",
      "input": -0.5
    }
  ]
}
```
