import React from 'react';

export default function Message({ role, content, modelName, timestamp }) {
  const isUser = role === 'user';

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.35rem',
        padding: '0.85rem 1rem',
        borderRadius: 'var(--radius-sm)',
        background: isUser ? 'rgba(56, 189, 248, 0.06)' : 'var(--bg-secondary)',
        border: `1px solid ${isUser ? 'rgba(56, 189, 248, 0.25)' : 'var(--border-subtle)'}`,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.75rem',
          color: isUser ? 'var(--cyan-accent)' : 'var(--text-dim)',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
        }}
      >
        <span>{isUser ? 'Prompt' : modelName || 'Model Response'}</span>
        {timestamp && <span style={{ fontFamily: 'var(--font-mono)' }}>{timestamp}</span>}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.9rem',
          lineHeight: 1.6,
          color: 'var(--text-main)',
          whiteSpace: 'pre-wrap',
        }}
      >
        {content}
      </div>
    </div>
  );
}
