import React from 'react';

export default function StatusBar({ metrics, isGenerating }) {
  if (!metrics && !isGenerating) {
    return (
      <div className="status-bar-container">
        <div className="metrics-disclaimer">
          Ready for local CPU inference. Select a model, enter a prompt, and click Generate.
        </div>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
          Local PyTorch Engine • FP32
        </div>
      </div>
    );
  }

  return (
    <div className="status-bar-container">
      <div className="metrics-group">
        <div className="metric-pill">
          <span className="metric-label">Latency</span>
          <span className="metric-number">
            {metrics ? `${metrics.generation_time_seconds.toFixed(3)}s` : '...'}
          </span>
        </div>

        <div className="metric-pill">
          <span className="metric-label">Tokens Generated</span>
          <span className="metric-number">
            {metrics ? `${metrics.tokens_generated} ${metrics.token_unit}` : '...'}
          </span>
        </div>

        <div className="metric-pill">
          <span className="metric-label">Throughput</span>
          <span className="metric-number">
            {metrics ? `${metrics.tokens_per_second.toFixed(1)} tok/s` : '...'}
          </span>
        </div>
      </div>

      <div className="metrics-disclaimer">
        ℹ️ Note: Character tokens (MiniGPT) and BPE subwords (DistilGPT-2) are distinct units.
        Character models require ~4 tokens per English word.
      </div>
    </div>
  );
}
