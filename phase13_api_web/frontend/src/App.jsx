import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import ModelSelector from './components/ModelSelector';
import ModelInfo from './components/ModelInfo';
import GenerationSettings from './components/GenerationSettings';
import ChatPanel from './components/ChatPanel';
import StatusBar from './components/StatusBar';
import { getHealth, getModels, getModelInfo, generate, streamGenerate } from './api';

const DEFAULT_SETTINGS = {
  max_new_tokens: 60,
  temperature: 0.8,
  top_k: 20,
  seed: 42,
  stream: true,
};

export default function App() {
  const [apiConnected, setApiConnected] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState('minigpt-programming');
  const [modelInfo, setModelInfo] = useState(null);

  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [prompt, setPrompt] = useState('Explain inheritance in Java');
  const [response, setResponse] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  const abortControllerRef = useRef(null);

  // Poll backend health and initialize models list
  useEffect(() => {
    let isMounted = true;

    async function checkHealthAndLoad() {
      try {
        const health = await getHealth();
        if (isMounted) setApiConnected(health.status === 'ok');

        const modelsData = await getModels();
        if (isMounted && modelsData.models && modelsData.models.length > 0) {
          setModels(modelsData.models);
          // Set default if not set
          if (!selectedModelId) {
            setSelectedModelId(modelsData.models[0].id);
          }
        }
      } catch (err) {
        console.warn('Backend unavailable:', err.message);
        if (isMounted) setApiConnected(false);
      }
    }

    checkHealthAndLoad();
    const interval = setInterval(checkHealthAndLoad, 10000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Update model details when selection changes
  useEffect(() => {
    if (!selectedModelId) return;
    let isMounted = true;

    async function loadInfo() {
      try {
        const info = await getModelInfo(selectedModelId);
        if (isMounted) setModelInfo(info);
      } catch (err) {
        console.warn('Failed to load model specs:', err.message);
      }
    }

    loadInfo();
    return () => {
      isMounted = false;
    };
  }, [selectedModelId]);

  const handleSettingChange = (key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleResetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
  };

  const handleClear = () => {
    setPrompt('');
    setResponse('');
    setMetrics(null);
    setError(null);
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setError(null);
    setIsGenerating(true);
    setResponse('');
    setMetrics(null);

    const payload = {
      model: selectedModelId,
      prompt: prompt.trim(),
      max_new_tokens: settings.max_new_tokens,
      temperature: settings.temperature,
      top_k: settings.top_k,
      seed: settings.seed,
    };

    if (settings.stream) {
      abortControllerRef.current = new AbortController();

      await streamGenerate(payload, {
        signal: abortControllerRef.current.signal,
        onToken: (token) => {
          setResponse((prev) => prev + token);
        },
        onComplete: (completionData) => {
          setMetrics({
            generation_time_seconds: completionData.generation_time_seconds,
            tokens_generated: completionData.tokens_generated,
            tokens_per_second: completionData.tokens_per_second,
            token_unit: completionData.token_unit,
          });
          setIsGenerating(false);
        },
        onError: (err) => {
          setError(err.message || 'Streaming generation failed.');
          setIsGenerating(false);
        },
      });
    } else {
      // Synchronous generation
      try {
        const result = await generate(payload);
        setResponse(result.text);
        setMetrics({
          generation_time_seconds: result.generation_time_seconds,
          tokens_generated: result.generated_tokens,
          tokens_per_second: result.tokens_per_second,
          token_unit: result.token_unit,
        });
      } catch (err) {
        setError(err.message || 'Generation request failed.');
      } finally {
        setIsGenerating(false);
      }
    }
  };

  const selectedModel = models.find((m) => m.id === selectedModelId);

  return (
    <div className="app-container">
      <Header apiConnected={apiConnected} />

      <main className="studio-layout">
        <aside className="studio-sidebar">
          <ModelSelector
            models={models}
            selectedModelId={selectedModelId}
            onSelectModel={setSelectedModelId}
            disabled={isGenerating}
          />

          <ModelInfo modelInfo={modelInfo} />

          <GenerationSettings
            settings={settings}
            onChange={handleSettingChange}
            onReset={handleResetSettings}
            disabled={isGenerating}
          />
        </aside>

        <section className="studio-workspace">
          <ChatPanel
            prompt={prompt}
            onChangePrompt={setPrompt}
            onGenerate={handleGenerate}
            onClear={handleClear}
            isGenerating={isGenerating}
            response={response}
            error={error}
            selectedModelName={selectedModel?.name}
          />

          <StatusBar metrics={metrics} isGenerating={isGenerating} />
        </section>
      </main>

      <footer className="app-footer">
        MiniGPT Project — Phase 13 • From-Scratch Transformers &amp; PEFT LoRA • 100% Free &amp; CPU Compatible
      </footer>
    </div>
  );
}
