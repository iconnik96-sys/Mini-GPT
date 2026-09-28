/**
 * Centralized API client for MiniGPT Studio.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/**
 * Check backend health status.
 */
export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

/**
 * Fetch catalog of all supported models.
 */
export async function getModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) {
    throw new Error(`Failed to load models list (${res.status})`);
  }
  return res.json();
}

/**
 * Fetch detailed metadata for a specific model ID.
 */
export async function getModelInfo(modelId) {
  const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Model not found (${res.status})`);
  }
  return res.json();
}

/**
 * Execute synchronous one-shot generation.
 */
export async function generate(payload) {
  const res = await fetch(`${API_BASE}/api/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Generation request failed (${res.status})`);
  }

  return res.json();
}

/**
 * Execute streaming generation using Server-Sent Events (SSE).
 * Reads the response stream chunks and parses data lines.
 */
export async function streamGenerate(payload, { onToken, onComplete, onError, signal }) {
  try {
    const res = await fetch(`${API_BASE}/api/generate/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Stream connection failed (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data:')) continue;

        const dataStr = trimmed.slice(5).trim();
        if (!dataStr) continue;

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.error) {
            onError(new Error(parsed.error));
            return;
          }
          if (parsed.token) {
            onToken(parsed.token);
          }
          if (parsed.done) {
            onComplete(parsed);
          }
        } catch (e) {
          console.warn('Failed to parse SSE payload:', dataStr);
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      return;
    }
    onError(err);
  }
}
