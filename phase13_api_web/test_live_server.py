"""
End-to-End Live Verification Script for Phase 13 FastAPI & Streaming API.
Uses TestClient to test the complete ASGI application stack, routing, validation,
Pydantic serialization, ModelManager, real model generation, and SSE streaming.
"""

import json
from fastapi.testclient import TestClient
from phase13_api_web.backend.app import app

client = TestClient(app)

print("==================================================")
print(" 1. Verifying Live Health Endpoint")
print("==================================================")
r = client.get("/api/health")
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}\n")
assert r.status_code == 200
assert r.json()["status"] == "ok"

print("==================================================")
print(" 2. Verifying Live Models Catalog")
print("==================================================")
r = client.get("/api/models")
print(f"Status: {r.status_code}")
models = r.json().get("models", [])
for m in models:
    print(f" • {m['id']}: {m['name']} ({m['parameters']:,} params, {m['tokenizer']} tokenizer)")
assert len(models) >= 4

print("\n==================================================")
print(" 3. Verifying Live Generation: minigpt-programming")
print("==================================================")
payload = {
    "model": "minigpt-programming",
    "prompt": "Explain inheritance in Java",
    "max_new_tokens": 60,
    "temperature": 0.8,
    "top_k": 20,
    "seed": 42
}
r = client.post("/api/generate", json=payload)
print(f"Status: {r.status_code}")
gen = r.json()
print(f"Prompt: {gen['prompt']}")
print(f"Response: {gen['text']}")
print(f"Tokens Generated: {gen['generated_tokens']} ({gen['token_unit']})")
print(f"Latency: {gen['generation_time_seconds']:.4f}s")
print(f"Throughput: {gen['tokens_per_second']:.1f} tok/s\n")
assert gen["generated_tokens"] == 60
assert r.status_code == 200

print("==================================================")
print(" 4. Verifying Live Server-Sent Events (SSE) Stream")
print("==================================================")
r = client.post("/api/generate/stream", json=payload)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type')}")
print("Streaming tokens in real-time:")
tokens = []
final_payload = None

lines = [line.strip() for line in r.text.split("\n") if line.strip().startswith("data:")]
for line in lines:
    data_json = json.loads(line[5:].strip())
    if "token" in data_json:
        tokens.append(data_json["token"])
        print(data_json["token"], end="", flush=True)
    if data_json.get("done"):
        final_payload = data_json

print("\n")
print(f"Completed streaming {len(tokens)} tokens.")
print(f"Final Event: {final_payload}")
assert len(tokens) == 60
assert final_payload is not None
assert final_payload["done"] is True

print("\n==================================================")
print(" ALL LIVE END-TO-END VERIFICATIONS PASSED!")
print("==================================================")
