# Phase 13 — API + Web Interface: MiniGPT Studio

Welcome to **MiniGPT Studio**, a local developer workspace and web interface for exploring, generating from, and analyzing the MiniGPT family of language models.

---

## 1. Purpose

MiniGPT Studio bridges the gap between terminal-based experimentation and interactive visual exploration:
- Provides a clean, modern REST and Server-Sent Events (SSE) API built with **FastAPI** and **Pydantic**.
- Features a responsive, dark-mode developer UI built with **React** and **Vite**.
- Exposes all four models developed across Phases 8–10 through a single unified interface.
- Implements real-time token streaming, active model memory management, and parameter validation on commodity CPU hardware.
- Fully free, self-contained, and operates with zero cloud dependencies or paid APIs.

---

## 2. Architecture

```text
                    ┌────────────────────────────────────────┐
                    │      React Frontend (Vite)             │
                    │         MiniGPT Studio                 │
                    │   (Model Selector, Settings, Chat)     │
                    └───────────────────┬────────────────────┘
                                        │ HTTP / SSE Stream
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       FastAPI Application              │
                    │    (Routes: health, models, generate)  │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │          ModelManager                  │
                    │  (Lazy loading & active model cache)   │
                    └───────────────────┬────────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
     From-Scratch MiniGPT        DistilGPT-2 Base        DistilGPT-2 + LoRA
     (Char-level Transformer)       (Pretrained)         (PEFT r=8 Adapters)
```

---

## 3. Directory Layout

```text
phase13_api_web/
├── README.md                 # Complete documentation and setup guide
├── test_live_server.py       # End-to-end ASGI integration verification script
├── backend/
│   ├── app.py                # FastAPI entry point & CORS configuration
│   ├── config.py             # Server settings & path resolution
│   ├── schemas.py            # Pydantic request/response validation models
│   ├── model_manager.py      # Lazy loader, active model cache, and SSE streamer
│   ├── dependencies.py       # Dependency injection providers
│   ├── routes/
│   │   ├── __init__.py       # Route exports
│   │   ├── health.py         # GET /api/health
│   │   ├── models.py         # GET /api/models & GET /api/models/{id}
│   │   └── generation.py     # POST /api/generate & POST /api/generate/stream
│   └── tests/
│       ├── __init__.py
│       ├── test_health.py    # Health and root endpoint unit tests
│       ├── test_models.py    # Catalog and metadata lookup tests
│       └── test_generation.py# Parameter validation, OOV check, and generation tests
│
├── frontend/
│   ├── package.json          # React 18, Vite dependencies & build scripts
│   ├── vite.config.js        # Vite config with API proxy
│   ├── index.html            # Studio HTML shell & web fonts
│   ├── public/
│   │   └── favicon.svg       # Studio brand icon
│   └── src/
│       ├── main.jsx          # React DOM entry point
│       ├── App.jsx           # Root layout and application state
│       ├── api.js            # Centralized API fetch & SSE client
│       ├── styles.css        # Professional dark theme design system
│       └── components/
│           ├── Header.jsx             # Top bar with API status badge
│           ├── ModelSelector.jsx      # Interactive model selection cards
│           ├── ModelInfo.jsx          # Model parameter and architecture card
│           ├── GenerationSettings.jsx # Sliders for temperature, top-k, tokens
│           ├── ChatPanel.jsx          # Prompt input, sample chips & response box
│           ├── Message.jsx            # Message bubbles
│           └── StatusBar.jsx          # Latency, throughput & token disclaimer
│
└── examples/
    └── api_examples.md       # Curl, PowerShell, and JSON examples
```

---

## 4. Supported Models

| Model ID | Display Name | Architecture | Tokenizer | Total Parameters | Trainable Params | Checkpoint Path |
|---|---|---|---|---:|---:|---|
| `minigpt-baseline` | MiniGPT Baseline (Phase 8) | Custom Transformer (2L, 4H, 64D) | Character (52 chars) | 110,336 | 110,336 (100%) | `checkpoints/exp1_baseline/best.pt` |
| `minigpt-programming` | MiniGPT Programming (Phase 9) | Custom Transformer (2L, 4H, 64D) | Character (88 chars) | 114,944 | 114,944 (100%) | `checkpoints/phase9_longer/best.pt` |
| `distilgpt2` | DistilGPT-2 Base (Pretrained) | DistilGPT-2 (6L, 12H, 768D) | Byte-level BPE (50,257) | 81,912,576 | 0 (0%) | Base Hugging Face model |
| `distilgpt2-lora` | DistilGPT-2 + LoRA (Phase 10) | DistilGPT-2 + LoRA ($r=8$) | Byte-level BPE (50,257) | 82,060,032 | 147,456 (0.18%) | `phase10_lora/checkpoints/distilgpt2_lora_programming/` |

---

## 5. Getting Started & Running

### Prerequisites
- Python 3.12 with PyTorch (`torch`), `transformers`, `peft`, `fastapi`, and `uvicorn`.
- Node.js (v18+) and npm / pnpm.

### Step 1: Launch Backend Server
```powershell
# From project root
python -m uvicorn phase13_api_web.backend.app:app --host 127.0.0.1 --port 8000 --reload
```
The FastAPI documentation (Swagger UI) is available at:
`http://127.0.0.1:8000/docs`

### Step 2: Launch Frontend Development Server
```powershell
# In a separate terminal, navigate to the frontend directory
cd phase13_api_web/frontend

# Install dependencies (first time only)
npm install
# or with pnpm:
# pnpm install

# Start the Vite development server
npm run dev
# or:
# pnpm run dev
```
Open your browser to:
`http://localhost:5173/`

### Step 3: Production Frontend Build
```powershell
cd phase13_api_web/frontend
npm run build
```
Creates an optimized static bundle in `phase13_api_web/frontend/dist/`.

---

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status and links to documentation. |
| `GET` | `/api/health` | Health check (`{"status": "ok", "service": "MiniGPT Studio API"}`). |
| `GET` | `/api/models` | List metadata for all registered models without loading weights into RAM. |
| `GET` | `/api/models/{model_id}` | Detailed metadata for a specific model (or 404 if not found). |
| `POST` | `/api/generate` | Synchronous one-shot text generation with timing and token metrics. |
| `POST` | `/api/generate/stream` | Real-time Server-Sent Events (SSE) token streaming. |

For detailed request payloads and JSON response examples, see [`phase13_api_web/examples/api_examples.md`](file:///c:/Users/nikhi/OneDrive/Desktop/Project/Mini-Gpt/phase13_api_web/examples/api_examples.md).

---

## 7. Streaming Mechanics (SSE)

The `/api/generate/stream` endpoint uses standard Server-Sent Events:
1. **From-Scratch MiniGPT Models:** An incremental generator computes logits, samples the next character token, and immediately yields:
   ```text
   data: {"token": "p"}
   ```
2. **DistilGPT-2 / LoRA Models:** Generates tokens in a background worker thread using Hugging Face's `TextIteratorStreamer`, yielding BPE subwords in real-time as they emerge from the Transformer.
3. **Completion Payload:** Upon generation finish, an event with comprehensive statistics is sent:
   ```text
   data: {"done": true, "model": "minigpt-programming", "tokens_generated": 60, "generation_time_seconds": 0.0854, "tokens_per_second": 702.8, "token_unit": "char"}
   ```

---

## 8. Verification & Testing

### Run Backend Test Suite
```powershell
# Run with pytest (13 tests)
python -m pytest phase13_api_web/backend/tests -v

# Or run with standard unittest
python -m unittest discover -s phase13_api_web/backend/tests -p "test_*.py" -v
```

### Run Live End-to-End Verification
```powershell
python -m phase13_api_web.test_live_server
```

---

## 9. Performance & CPU Feasibility

Measured on a standard laptop CPU (Intel 12-core, Windows 11):

| Model | Token Type | Latency (60 tokens) | Throughput | Memory Footprint |
|---|---|---|---|---|
| `minigpt-programming` | Character | 0.085s | **702.8 tok/s** | ~2 MB |
| `minigpt-baseline` | Character | 0.092s | **652.1 tok/s** | ~2 MB |
| `distilgpt2-lora` | BPE subword | ~1.35s | **~28.0 tok/s** | ~340 MB |

---

## 10. Honest Limitations

1. **Completion vs. Chatbot:** These models are causal next-token completion models trained on code/Shakespeare text or instruction templates. They are not conversational agents with infinite conversational memory.
2. **Character Vocabulary Bounds:** From-scratch character models are strictly bounded by their training corpus alphabet. Prompts containing characters outside their vocabulary are rejected with clear error details.
3. **CPU Throughput:** DistilGPT-2 (82M params) runs at ~28 tokens/second on CPU, which is fast enough for interactive use but distinct from GPU inference clusters.

---

## 11. Screenshots

*(Reserved for UI captures in `phase13_api_web/screenshots/`)*
