import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, ChevronDown } from 'lucide-react';
import { api } from '@/api/client';
import { RetrievalList } from '@/components/RetrievalList';
import { ImageViewer } from '@/components/ImageViewer';
import { GraphViewer } from '@/components/GraphViewer';
import type { ChatMessage, ChatResponse, DatasetKey, ModelKey } from '@/types';
import { MODEL_OPTIONS } from '@/types';

/** Display server-measured latency (input → output), e.g. ``50s`` or ``3.2s``. */
function formatLatencySeconds(ms: number): string {
  const s = ms / 1000;
  if (!Number.isFinite(s) || s < 0) return '';
  if (s < 60) {
    return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
  }
  const m = Math.floor(s / 60);
  const r = Math.round(s - m * 60);
  return `${m}m ${r}s`;
}

interface LiveChatTabProps {
  dataset: DatasetKey;
}

const SUGGESTIONS: Record<DatasetKey, string[]> = {
  webqa: [
    'Are the rear wheels on the wheelchairs at the 2000 Sydney Paralympic Games straight or angled?',
    'Does the Cincinnati Music Hall have columns inside and outside?',
    'Are there any buildings shorter than the flag pole in 481 8th Ave, New York?',
  ],
  mmqa: [
    'For which film did Ben Piazza play the role of Mr. Simms?',
    'What is the nationality of the director of the film Mask?',
    'Which country won the most medals at the 2000 Summer Olympics?',
  ],
};

let msgCounter = 0;
function nextId() {
  return `msg-${Date.now()}-${++msgCounter}`;
}

export function LiveChatTab({ dataset }: LiveChatTabProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [model, setModel] = useState<ModelKey>('hf_gemma4_e4b');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  function clearConversation() {
    setMessages([]);
    setInput('');
    inputRef.current?.focus();
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || sending) return;

    const userMsg: ChatMessage = {
      id: nextId(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };

    const loadingMsg: ChatMessage = {
      id: nextId(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      loading: true,
    };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setInput('');
    setSending(true);

    const reqStarted = performance.now();
    try {
      const resp: ChatResponse = await api.chat(question, dataset, model);
      const elapsed =
        resp.elapsed_ms != null && Number.isFinite(resp.elapsed_ms)
          ? resp.elapsed_ms
          : performance.now() - reqStarted;
      const assistantMsg: ChatMessage = {
        id: loadingMsg.id,
        role: 'assistant',
        content: resp.answer,
        timestamp: new Date(),
        sources: resp.sources,
        graphData: resp.graph ?? undefined,
        goldFacts: resp.gold_facts,
        elapsedMs: elapsed,
      };
      setMessages((prev) => prev.map((m) => (m.id === loadingMsg.id ? assistantMsg : m)));
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: loadingMsg.id,
        role: 'assistant',
        content: `Error: ${err instanceof Error ? err.message : 'Request failed'}. The backend might not support live queries yet.`,
        timestamp: new Date(),
        elapsedMs: performance.now() - reqStarted,
      };
      setMessages((prev) => prev.map((m) => (m.id === loadingMsg.id ? errorMsg : m)));
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const datasetLabel = dataset === 'mmqa' ? 'MultimodalQA' : 'WebQA';
  const suggestions = SUGGESTIONS[dataset];

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-md">
              <div className="w-14 h-14 rounded-2xl bg-[#eef2ff] flex items-center justify-center mx-auto mb-4">
                <Bot size={24} className="text-[#6366f1]" strokeWidth={1.5} />
              </div>
              <h2 className="text-[16px] text-[#1a1a1a] mb-2" style={{ fontWeight: 500 }}>
                Ask a question
              </h2>
              <p className="text-[13px] text-[#9a9a9a] leading-relaxed mb-5">
                Ask any question about the {datasetLabel} dataset. The pipeline will retrieve
                relevant sources and generate an answer using ColGraphRAG.
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="text-[11px] text-[#5a5a5a] bg-[#f4f4f4] hover:bg-[#eaeaea] border border-[#e5e5e5] rounded-full px-3 py-1.5 transition-colors"
                    style={{ fontWeight: 500 }}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-6 py-6 space-y-5">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                {msg.role === 'user' ? (
                  <div className="w-7 h-7 rounded-full bg-[#f0f0f0] flex items-center justify-center shrink-0 mt-0.5">
                    <User size={13} color="#4a4a4a" />
                  </div>
                ) : (
                  <div className="w-7 h-7 rounded-full bg-[#1a1a1a] flex items-center justify-center shrink-0 mt-0.5">
                    <Bot size={13} color="white" />
                  </div>
                )}

                <div
                  className={`max-w-[82%] flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                >
                  {msg.role === 'user' ? (
                    <div className="rounded-2xl rounded-tr-sm bg-[#1a1a1a] text-white px-4 py-3 text-[14px] leading-relaxed">
                      {msg.content}
                    </div>
                  ) : msg.loading ? (
                    <div className="rounded-2xl rounded-tl-sm bg-[#f9f9f9] border border-[#eeeeee] px-4 py-3">
                      <div className="flex gap-1.5 items-center h-5">
                        <Loader2 size={14} className="text-[#9a9a9a] animate-spin" />
                        <span className="text-[12px] text-[#9a9a9a]">Thinking...</span>
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-2xl rounded-tl-sm bg-[#f9f9f9] text-[#1a1a1a] border border-[#eeeeee] px-4 py-3 text-[14px] leading-relaxed w-full">
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                      {msg.role === 'assistant' && msg.elapsedMs != null && (
                        <p
                          className="mt-2 text-[10px] text-[#b8b8b8] tracking-wide"
                          title="Round-trip time (request to response)"
                        >
                          {formatLatencySeconds(msg.elapsedMs)}
                        </p>
                      )}
                      {msg.sources && msg.sources.length > 0 && (
                        <RetrievalList items={msg.sources} />
                      )}
                      {msg.graphData && (
                        <GraphViewer data={msg.graphData} loading={false} />
                      )}
                      {msg.goldFacts && msg.goldFacts.length > 0 && (
                        <ImageViewer facts={msg.goldFacts} dataset={dataset} />
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="px-6 pb-4 pt-2 bg-[#f7f7f7] shrink-0">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 bg-white rounded-xl border border-[#e5e5e5] px-4 py-3 shadow-[0_1px_4px_rgba(0,0,0,0.04)] focus-within:border-[#b0b0b0] transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask a question about the ${datasetLabel} dataset...`}
              rows={1}
              className="flex-1 text-[14px] text-[#1a1a1a] placeholder:text-[#b0b0b0] outline-none resize-none bg-transparent leading-relaxed max-h-32"
              disabled={sending}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || sending}
              className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
                input.trim() && !sending
                  ? 'bg-[#1a1a1a] hover:bg-[#333] cursor-pointer'
                  : 'bg-[#e0e0e0] cursor-not-allowed'
              }`}
            >
              <Send size={13} color={input.trim() && !sending ? 'white' : '#9a9a9a'} />
            </button>
          </div>
          <div className="flex justify-between items-center mt-2">
            <p className="text-[10px] text-[#c0c0c0]">
              Shift+Enter for new line
            </p>
            <div className="flex items-center gap-3">
              {/* Model selector */}
              <div className="relative flex items-center">
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value as ModelKey)}
                  disabled={sending}
                  className="appearance-none text-[11px] text-[#5a5a5a] bg-transparent border border-[#e0e0e0] rounded-md pl-2.5 pr-6 py-1 outline-none hover:border-[#b0b0b0] focus:border-[#b0b0b0] transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <option key={opt.key} value={opt.key}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <ChevronDown
                  size={10}
                  className="absolute right-1.5 text-[#9a9a9a] pointer-events-none"
                />
              </div>
              {messages.length > 0 && (
                <button
                  onClick={clearConversation}
                  className="text-[11px] text-[#9a9a9a] hover:text-[#5a5a5a] transition-colors"
                >
                  Clear conversation
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
