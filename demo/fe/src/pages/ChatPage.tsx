import { useState, useEffect } from 'react';
import { GitBranch, Plus, LayoutList, MessageCircle } from 'lucide-react';
import { api } from '@/api/client';
import { useDataset } from '@/contexts/DatasetContext';
import { ResultsTab } from '@/components/ResultsTab';
import { LiveChatTab } from '@/components/LiveChatTab';
import type { RunInfo, QuestionSummary, DatasetInfo } from '@/types';

type Tab = 'results' | 'chat';

const DATASET_COLORS = {
  webqa: {
    active: 'bg-[#ede9fe] text-[#7c3aed]',
    inactive: 'text-[#6a6a6a] hover:bg-[#f7f7f7]',
  },
  mmqa: {
    active: 'bg-[#dbeafe] text-[#1d4ed8]',
    inactive: 'text-[#6a6a6a] hover:bg-[#f7f7f7]',
  },
};

export function ChatPage() {
  const { dataset, setDataset } = useDataset();
  const [runInfo, setRunInfo] = useState<RunInfo | null>(null);
  const [questions, setQuestions] = useState<QuestionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<Tab>('results');
  const [chatKey, setChatKey] = useState(0);
  const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([]);

  // Load available datasets once on mount
  useEffect(() => {
    api.getDatasets().then((resp) => {
      setAvailableDatasets(resp.datasets);
    }).catch(() => {
      // Fallback: assume both available
      setAvailableDatasets([
        { key: 'webqa', label: 'WebQA', available: true, run_id: null },
        { key: 'mmqa', label: 'MultimodalQA', available: true, run_id: null },
      ]);
    });
  }, []);

  // Reload questions and run info whenever dataset changes
  useEffect(() => {
    setLoading(true);
    setRunInfo(null);
    setQuestions([]);
    async function loadInitialData() {
      try {
        const [info, qs] = await Promise.all([
          api.getRunInfo(dataset),
          api.getQuestions(dataset),
        ]);
        setRunInfo(info);
        setQuestions(qs);
      } catch (err) {
        console.error('Failed to load data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadInitialData();
  }, [dataset]);

  function handleNewChat() {
    setActiveTab('chat');
    setChatKey((k) => k + 1);
  }

  function handleDatasetSwitch(key: typeof dataset) {
    if (key === dataset) return;
    setDataset(key);
    setActiveTab('results');
  }

  const predictedCount = questions.filter((q) => q.predicted_answer).length;

  return (
    <div className="h-screen flex bg-[#f7f7f7]">
      {/* Global sidebar */}
      <div className="w-60 bg-white border-r border-[#e8e8e8] flex flex-col shrink-0">
        {/* Logo/header */}
        <div className="px-4 pt-4 pb-3 border-b border-[#f0f0f0]">
          <div className="flex items-center gap-2.5 mb-3">
            <div className="w-8 h-8 rounded-xl bg-[#f5f0ff] flex items-center justify-center">
              <GitBranch size={15} className="text-[#8b5cf6]" strokeWidth={1.75} />
            </div>
            <div>
              <h1 className="text-[15px] text-[#1a1a1a]" style={{ fontWeight: 500 }}>
                ColGraphRAG
              </h1>
              <p className="text-[11px] text-[#9a9a9a]">Multi-Dataset Demo</p>
            </div>
          </div>

          {/* Dataset switcher */}
          <div className="flex gap-1 bg-[#f4f4f4] rounded-lg p-0.5 mb-3">
            {availableDatasets.map((ds) => (
              <button
                key={ds.key}
                onClick={() => handleDatasetSwitch(ds.key)}
                disabled={!ds.available}
                title={!ds.available ? `${ds.label} results not available` : ds.run_id ?? undefined}
                className={`flex-1 text-[11px] px-2 py-1.5 rounded-md transition-all ${
                  dataset === ds.key
                    ? DATASET_COLORS[ds.key].active
                    : ds.available
                    ? DATASET_COLORS[ds.key].inactive
                    : 'text-[#c0c0c0] cursor-not-allowed'
                }`}
                style={{ fontWeight: dataset === ds.key ? 600 : 400 }}
              >
                {ds.label}
              </button>
            ))}
          </div>

          {runInfo && (
            <div className="flex items-center gap-1.5 bg-[#f0fdf4] border border-[#bbf7d0] rounded-full px-3 py-1.5 w-fit">
              <div className="w-1.5 h-1.5 bg-[#22c55e] rounded-full" />
              <span className="text-[10px] text-[#15803d]" style={{ fontWeight: 500 }}>
                {predictedCount} answered / {questions.length} total
              </span>
            </div>
          )}
          {!runInfo && !loading && (
            <div className="flex items-center gap-1.5 bg-[#fef3c7] border border-[#fcd34d] rounded-full px-3 py-1.5 w-fit">
              <div className="w-1.5 h-1.5 bg-[#f59e0b] rounded-full" />
              <span className="text-[10px] text-[#92400e]" style={{ fontWeight: 500 }}>
                {questions.length} questions loaded
              </span>
            </div>
          )}
        </div>

        {/* New Chat button */}
        <div className="px-3 py-3">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-[#e5e5e5] text-[13px] text-[#1a1a1a] hover:bg-[#f7f7f7] transition-colors"
            style={{ fontWeight: 500 }}
          >
            <Plus size={14} className="text-[#9a9a9a]" />
            New Chat
          </button>
        </div>

        {/* Tab navigation */}
        <nav className="px-3 flex flex-col gap-0.5">
          <button
            onClick={() => setActiveTab('results')}
            className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] transition-all ${
              activeTab === 'results'
                ? 'bg-[#f0f0f0] text-[#1a1a1a]'
                : 'text-[#6a6a6a] hover:bg-[#f7f7f7]'
            }`}
            style={{ fontWeight: activeTab === 'results' ? 500 : 400 }}
          >
            <LayoutList size={15} strokeWidth={1.75} />
            Results
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-[13px] transition-all ${
              activeTab === 'chat'
                ? 'bg-[#f0f0f0] text-[#1a1a1a]'
                : 'text-[#6a6a6a] hover:bg-[#f7f7f7]'
            }`}
            style={{ fontWeight: activeTab === 'chat' ? 500 : 400 }}
          >
            <MessageCircle size={15} strokeWidth={1.75} />
            Live Chat
          </button>
        </nav>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Footer */}
        <div className="px-4 pb-4 text-[10px] text-[#c0c0c0] leading-relaxed">
          Gemma 4 E4B + ColEmbed MaxSim
        </div>
      </div>

      {/* Main content area */}
      {activeTab === 'results' ? (
        <ResultsTab questions={questions} loading={loading} dataset={dataset} />
      ) : (
        <LiveChatTab key={chatKey} dataset={dataset} />
      )}
    </div>
  );
}
