import React from 'react';

export default function ModelSelector({ models, selectedModelId, onSelectModel, disabled }) {
  const getBadgeClass = (type) => {
    if (type === 'from_scratch') return 'badge-scratch';
    if (type === 'peft_lora') return 'badge-lora';
    return 'badge-pretrained';
  };

  const getBadgeLabel = (type) => {
    if (type === 'from_scratch') return 'Scratch';
    if (type === 'peft_lora') return 'LoRA';
    return 'Pretrained';
  };

  return (
    <div className="panel-card">
      <div className="panel-title">
        <span>Available Models</span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
          {models.length} registered
        </span>
      </div>
      <div className="model-list">
        {models.map((m) => {
          const isSelected = m.id === selectedModelId;
          return (
            <button
              key={m.id}
              type="button"
              className={`model-option-btn ${isSelected ? 'active' : ''}`}
              onClick={() => onSelectModel(m.id)}
              disabled={disabled}
            >
              <div className="model-option-header">
                <span className="model-name">{m.name}</span>
                <span className={`model-badge ${getBadgeClass(m.type)}`}>
                  {getBadgeLabel(m.type)}
                </span>
              </div>
              <div className="model-desc-short">
                {m.parameters.toLocaleString()} params • {m.tokenizer} tokenizer
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
