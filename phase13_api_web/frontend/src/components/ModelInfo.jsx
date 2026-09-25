import React from 'react';

export default function ModelInfo({ modelInfo }) {
  if (!modelInfo) return null;

  return (
    <div className="panel-card">
      <div className="panel-title">Model Specifications</div>
      <div className="info-grid">
        <div className="info-item">
          <div className="info-label">Architecture</div>
          <div className="info-value">
            {modelInfo.type === 'from_scratch'
              ? 'Custom Decoder'
              : modelInfo.type === 'peft_lora'
              ? 'DistilGPT-2 + LoRA'
              : 'DistilGPT-2 Base'}
          </div>
        </div>

        <div className="info-item">
          <div className="info-label">Total Parameters</div>
          <div className="info-value">{modelInfo.parameters.toLocaleString()}</div>
        </div>

        <div className="info-item">
          <div className="info-label">Trainable Params</div>
          <div className="info-value">
            {modelInfo.trainable_parameters.toLocaleString()} (
            {modelInfo.trainable_percentage}%)
          </div>
        </div>

        <div className="info-item">
          <div className="info-label">Tokenizer</div>
          <div className="info-value">
            {modelInfo.tokenizer === 'character' ? 'Character-level' : 'Byte-level BPE'}
          </div>
        </div>

        <div className="info-item">
          <div className="info-label">Context Window</div>
          <div className="info-value">{modelInfo.context_length} tokens</div>
        </div>

        <div className="info-item">
          <div className="info-label">Execution Target</div>
          <div className="info-value">CPU (Local FP32)</div>
        </div>
      </div>

      <div className="info-full-desc">{modelInfo.description}</div>
    </div>
  );
}
