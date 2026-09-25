import React from 'react';

export default function GenerationSettings({ settings, onChange, onReset, disabled }) {
  return (
    <div className="panel-card">
      <div className="panel-title">
        <span>Inference Parameters</span>
        <button
          type="button"
          className="reset-btn"
          onClick={onReset}
          disabled={disabled}
        >
          Reset Defaults
        </button>
      </div>

      <div className="settings-group">
        <div className="setting-row">
          <div className="setting-header">
            <span>Max New Tokens</span>
            <span className="setting-val">{settings.max_new_tokens}</span>
          </div>
          <input
            type="range"
            min="10"
            max="256"
            step="5"
            value={settings.max_new_tokens}
            onChange={(e) => onChange('max_new_tokens', parseInt(e.target.value, 10))}
            disabled={disabled}
            className="setting-slider"
          />
        </div>

        <div className="setting-row">
          <div className="setting-header">
            <span>Temperature</span>
            <span className="setting-val">{settings.temperature.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.0"
            max="2.0"
            step="0.05"
            value={settings.temperature}
            onChange={(e) => onChange('temperature', parseFloat(e.target.value))}
            disabled={disabled}
            className="setting-slider"
          />
        </div>

        <div className="setting-row">
          <div className="setting-header">
            <span>Top-K Sampling</span>
            <span className="setting-val">{settings.top_k}</span>
          </div>
          <input
            type="range"
            min="1"
            max="100"
            step="1"
            value={settings.top_k}
            onChange={(e) => onChange('top_k', parseInt(e.target.value, 10))}
            disabled={disabled}
            className="setting-slider"
          />
        </div>

        <div className="setting-row">
          <div className="setting-header">
            <span>Random Seed</span>
          </div>
          <input
            type="number"
            value={settings.seed ?? ''}
            onChange={(e) =>
              onChange('seed', e.target.value === '' ? null : parseInt(e.target.value, 10))
            }
            placeholder="e.g. 42 (blank for random)"
            disabled={disabled}
            className="setting-input"
          />
        </div>

        <div className="toggle-row">
          <div className="setting-header" style={{ marginBottom: 0 }}>
            <span>Stream Tokens (SSE)</span>
          </div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={settings.stream}
              onChange={(e) => onChange('stream', e.target.checked)}
              disabled={disabled}
            />
            <span className="slider-round" />
          </label>
        </div>
      </div>
    </div>
  );
}
