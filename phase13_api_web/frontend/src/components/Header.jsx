import React from 'react';

export default function Header({ apiConnected }) {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-logo">μ</div>
        <div className="brand-info">
          <h1>MiniGPT Studio</h1>
          <p>Local Educational LLM Workspace • Scratch Transformers &amp; LoRA</p>
        </div>
      </div>
      <div className="header-status">
        <div className="badge-pill">
          <span className={`status-dot ${apiConnected ? 'online' : 'offline'}`} />
          <span>{apiConnected ? 'API Connected (127.0.0.1:8000)' : 'API Disconnected'}</span>
        </div>
      </div>
    </header>
  );
}
