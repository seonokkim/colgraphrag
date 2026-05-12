import { useState, useEffect, useRef } from 'react';
import { Bot, User, GitBranch, BookOpen } from 'lucide-react';
import { api } from '@/api/client';
import { AnswerComparison } from '@/components/AnswerComparison';
import { RetrievalList } from '@/components/RetrievalList';
import { ImageViewer } from '@/components/ImageViewer';
import { GraphViewer } from '@/components/GraphViewer';
import type { QuestionSummary, QuestionDetail, GraphData, DatasetKey } from '@/types';

interface ResultsTabProps {
  questions: QuestionSummary[];
  loading: boolean;
  dataset: DatasetKey;
}

const QCATE_COLORS: Record<string, string> = {
  YesNo: 'bg-blue-50 text-blue-600',
  choose: 'bg-green-50 text-green-600',
  color: 'bg-amber-50 text-amber-600',
  shape: 'bg-purple-50 text-purple-600',
  text: 'bg-red-50 text-red-600',
  table: 'bg-orange-50 text-orange-600',
  image: 'bg-pink-50 text-pink-600',
  'table+image': 'bg-rose-50 text-rose-600',
  'table+text': 'bg-yellow-50 text-yellow-600',
  'text+image': 'bg-cyan-50 text-cyan-600',
  Others: 'bg-gray-50 text-gray-500',
};

export function ResultsTab({ questions, loading, dataset }: ResultsTabProps) {
  const [selectedQid, setSelectedQid] = useState<string | null>(null);
  const [questionDetail, setQuestionDetail] = useState<QuestionDetail | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [search, setSearch] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [questionDetail, detailLoading]);

  // Reset selection when dataset changes
  useEffect(() => {
    setSelectedQid(null);
    setQuestionDetail(null);
    setGraphData(null);
    setSearch('');
  }, [dataset]);

  useEffect(() => {
    if (!selectedQid) {
      setQuestionDetail(null);
      setGraphData(null);
      return;
    }
    const qid = selectedQid;
    async function loadDetail() {
      setDetailLoading(true);
      setGraphLoading(true);
      try {
        const detail = await api.getQuestion(qid, dataset);
        setQuestionDetail(detail);
        if (detail.graph_available) {
          try {
            setGraphData(await api.getGraph(qid, dataset));
          } catch {
            setGraphData(null);
          }
        } else {
          setGraphData(null);
        }
      } catch {
        setQuestionDetail(null);
        setGraphData(null);
      } finally {
        setDetailLoading(false);
        setGraphLoading(false);
      }
    }
    loadDetail();
  }, [selectedQid, dataset]);

  const filtered = search.trim()
    ? questions.filter(
        (q) =>
          q.question.toLowerCase().includes(search.toLowerCase()) ||
          q.qcate.toLowerCase().includes(search.toLowerCase()),
      )
    : questions;

  const predictedQuestions = questions.filter((q) => q.predicted_answer);

  return (
    <div className="flex flex-1 min-h-0">
      {/* Left sidebar: question list */}
      <div className="w-72 border-r border-[#e8e8e8] bg-white flex flex-col shrink-0">
        <div className="px-3 py-2 border-b border-[#f0f0f0]">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search questions..."
            className="w-full bg-[#f7f7f7] border border-[#e8e8e8] rounded-lg px-3 py-2 text-[12px] text-[#1a1a1a] placeholder:text-[#b0b0b0] outline-none focus:border-[#b0b0b0] transition-colors"
          />
        </div>
        <div className="flex-1 overflow-y-auto px-2 pb-2 scrollbar-thin">
          {loading ? (
            <div className="flex items-center justify-center h-20">
              <div className="flex gap-1.5">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="w-1.5 h-1.5 bg-[#b0b0b0] rounded-full animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          ) : (
            filtered.map((q) => (
              <button
                key={q.qid}
                onClick={() => setSelectedQid(q.qid)}
                className={`w-full text-left px-3 py-2.5 rounded-lg transition-all text-[13px] mb-0.5 ${
                  selectedQid === q.qid
                    ? 'bg-[#f0f0f0] text-[#1a1a1a]'
                    : 'text-[#5a5a5a] hover:bg-[#f7f7f7]'
                }`}
                style={{ fontWeight: selectedQid === q.qid ? 500 : 400 }}
              >
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded ${QCATE_COLORS[q.qcate] || QCATE_COLORS.Others}`}
                    style={{ fontWeight: 500 }}
                  >
                    {q.qcate}
                  </span>
                  {q.has_graph && <GitBranch size={10} className="text-[#b0b0b0]" />}
                </div>
                <p className="line-clamp-2 leading-snug">{q.question}</p>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Right: conversation view */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Top bar */}
        <div className="h-12 border-b border-[#f0f0f0] bg-white flex items-center px-6 shrink-0">
          {selectedQid && questionDetail ? (
            <div className="flex items-center gap-3">
              <span
                className={`text-[11px] px-2 py-0.5 rounded ${QCATE_COLORS[questionDetail.qcate] || QCATE_COLORS.Others}`}
                style={{ fontWeight: 500 }}
              >
                {questionDetail.qcate}
              </span>
              <span className="text-[13px] text-[#5a5a5a] truncate max-w-[500px]">
                {questionDetail.question}
              </span>
            </div>
          ) : (
            <span className="text-[13px] text-[#9a9a9a]">Select a question</span>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto scrollbar-thin">
          {!selectedQid ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-14 h-14 rounded-2xl bg-[#f5f0ff] flex items-center justify-center mx-auto mb-4">
                  <GitBranch size={24} className="text-[#8b5cf6]" strokeWidth={1.5} />
                </div>
                <h2 className="text-[16px] text-[#1a1a1a] mb-2" style={{ fontWeight: 500 }}>
                  Pipeline Results
                </h2>
                <p className="text-[13px] text-[#9a9a9a] leading-relaxed mb-5">
                  Browse pre-computed answers, retrieval sources, and knowledge graphs.
                </p>
                {predictedQuestions.length > 0 && (
                  <div className="flex flex-wrap gap-2 justify-center">
                    <span className="text-[11px] text-[#9a9a9a] self-center mr-1">Try:</span>
                    {predictedQuestions.slice(0, 3).map((q) => (
                      <button
                        key={q.qid}
                        onClick={() => setSelectedQid(q.qid)}
                        className="text-[11px] text-[#5a5a5a] bg-[#f4f4f4] hover:bg-[#eaeaea] border border-[#e5e5e5] rounded-full px-3 py-1.5 transition-colors max-w-[260px] truncate"
                        style={{ fontWeight: 500 }}
                      >
                        {q.question}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-6 py-6 space-y-5">
              {detailLoading ? (
                <div className="flex gap-3">
                  <div className="w-7 h-7 rounded-full bg-[#1a1a1a] flex items-center justify-center shrink-0">
                    <Bot size={13} color="white" />
                  </div>
                  <div className="bg-[#f9f9f9] border border-[#eeeeee] rounded-2xl rounded-tl-sm px-4 py-3">
                    <div className="flex gap-1.5 items-center h-5">
                      {[0, 1, 2].map((i) => (
                        <div
                          key={i}
                          className="w-1.5 h-1.5 bg-[#b0b0b0] rounded-full animate-bounce"
                          style={{ animationDelay: `${i * 0.15}s` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              ) : questionDetail ? (
                <>
                  <div className="flex gap-3 flex-row-reverse">
                    <div className="w-7 h-7 rounded-full bg-[#f0f0f0] flex items-center justify-center shrink-0 mt-0.5">
                      <User size={13} color="#4a4a4a" />
                    </div>
                    <div className="max-w-[82%] flex flex-col items-end">
                      <div className="rounded-2xl rounded-tr-sm bg-[#1a1a1a] text-white px-4 py-3 text-[14px] leading-relaxed">
                        {questionDetail.question}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="w-7 h-7 rounded-full bg-[#1a1a1a] flex items-center justify-center shrink-0 mt-0.5">
                      <Bot size={13} color="white" />
                    </div>
                    <div className="max-w-[82%] flex flex-col items-start">
                      <div className="rounded-2xl rounded-tl-sm bg-[#f9f9f9] text-[#1a1a1a] border border-[#eeeeee] px-4 py-3 text-[14px] leading-relaxed w-full">
                        <AnswerComparison
                          goldAnswers={questionDetail.gold_answers}
                          predictedAnswer={questionDetail.predicted_answer}
                          keywordsAnswer={questionDetail.keywords_answer}
                        />
                        <RetrievalList items={questionDetail.retrieval} />
                        <GraphViewer data={graphData} loading={graphLoading} />
                        {questionDetail.graph_available &&
                          !graphLoading &&
                          graphData === null && (
                            <p className="text-[11px] text-[#b45309] mt-2 px-0.5 leading-relaxed">
                              Knowledge graph failed to load (check Network tab for{' '}
                              <code className="font-mono bg-[#fef3c7] px-1 rounded">/api/graphs/</code>
                              ). Restart the demo backend if MMQA paths were updated.
                            </p>
                          )}
                        <ImageViewer facts={questionDetail.gold_facts} dataset={dataset} />
                      </div>
                      {questionDetail.retrieval.length > 0 && (
                        <div className="mt-2 flex items-center gap-1.5 text-[12px] text-[#6a6a6a]">
                          <BookOpen size={12} className="text-[#9a9a9a]" strokeWidth={1.75} />
                          <span style={{ fontWeight: 500 }}>Sources: {questionDetail.retrieval.length}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : null}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
