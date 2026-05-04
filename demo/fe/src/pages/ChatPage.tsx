import { useState, useEffect } from 'react';
import { GitBranch, Plus, LayoutList, MessageCircle } from 'lucide-react';
import { api } from '@/api/client';
import { ResultsTab } from '@/components/ResultsTab';
import { LiveChatTab } from '@/components/LiveChatTab';
import type { RunInfo, QuestionSummary } from '@/types';

type Tab = 'results' | 'chat';

export function ChatPage() {
  const [runInfo, setRunInfo] = useState<RunInfo | null>(null);
  const [questions, setQuestions] = useState<QuestionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<Tab>('results');
  const [chatKey, setChatKey] = useState(0);

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [info, qs] = await Promise.all([
          api.getRunInfo(),
          api.getQuestions(),
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
  }, []);

  function handleNewChat() {
    setActiveTab('chat');
    setChatKey((k) => k + 1);
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
              <p className="text-[11px] text-[#9a9a9a]">WebQA Demo</p>
            </div>
          </div>
          {runInfo && (
            <div className="flex items-center gap-1.5 bg-[#f0fdf4] border border-[#bbf7d0] rounded-full px-3 py-1.5 w-fit">
              <div className="w-1.5 h-1.5 bg-[#22c55e] rounded-full" />
              <span className="text-[10px] text-[#15803d]" style={{ fontWeight: 500 }}>
                {predictedCount} answered / {questions.length} total
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
        <ResultsTab questions={questions} loading={loading} />
      ) : (
        <LiveChatTab key={chatKey} />
      )}
    </div>
  );
}
