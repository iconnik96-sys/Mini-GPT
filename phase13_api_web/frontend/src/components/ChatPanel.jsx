import React, { useState } from 'react';

const SAMPLE_PROMPTS = [
  'Explain inheritance in Java',
  'Explain dependency injection in Spring Boot',
  'public class ProductController',
  'What is SQL JOIN?',
  'Explain Java interfaces',
];

export default function ChatPanel({
  prompt,
  onChangePrompt,
  onGenerate,
  onClear,
  isGenerating,
  response,
  error,
  selectedModelName,
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!response) return;
    navigator.clipboard.writeText(response);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!isGenerating && prompt.trim()) {
        onGenerate();
      }
    }
  };

  return (
    <div className="studio-workspace">
      {/* Sample Prompts */}
      <div className="sample-prompts-bar">
        <span className="sample-prompts-label">Try:</span>
        {SAMPLE_PROMPTS.map((p, idx) => (
          <button
            key={idx}
            type="button"
            className="prompt-chip"
            onClick={() => onChangePrompt(p)}
            disabled={isGenerating}
          >
            {p}
          </button>
        ))}
      </div>

      {/* Prompt Input Box */}
      <div className="prompt-container">
        <div className="prompt-header">
          <span>Prompt Input</span>
          <span className="prompt-meta">{prompt.length} chars • Ctrl+Enter to generate</span>
        </div>

        <textarea
          className="prompt-textarea"
          rows={4}
          value={prompt}
          onChange={(e) => onChangePrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter a programming question or code snippet for completion..."
          disabled={isGenerating}
        />

        <div className="prompt-actions">
          <button
            type="button"
            className="btn-secondary"
            onClick={onClear}
            disabled={isGenerating || (!prompt && !response)}
          >
            Clear
          </button>

          <button
            type="button"
            className="btn-primary"
            onClick={onGenerate}
            disabled={isGenerating || !prompt.trim()}
          >
            {isGenerating && <span className="spinner" />}
            <span>{isGenerating ? 'Generating...' : 'Generate Text'}</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <span>⚠️</span>
          <div>
            <strong>Generation Error:</strong> {error}
          </div>
        </div>
      )}

      {/* Response Box */}
      <div className="response-container">
        <div className="response-header">
          <h3>
            <span>Model Output</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--cyan-accent)', fontWeight: 500 }}>
              {selectedModelName ? `(${selectedModelName})` : ''}
            </span>
          </h3>

          {response && (
            <button type="button" className="copy-btn" onClick={handleCopy}>
              {copied ? 'Copied ✓' : 'Copy Output'}
            </button>
          )}
        </div>

        <div className="response-box">
          {response ? (
            <>
              {response}
              {isGenerating && <span className="cursor-blink" />}
            </>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">⚡</div>
              <div>No generated text yet.</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                Select a model from the left panel and click &quot;Generate Text&quot; to begin.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
