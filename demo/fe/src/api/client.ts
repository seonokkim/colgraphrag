import type {
  HealthResponse,
  RunInfo,
  LeaderboardScores,
  QuestionSummary,
  QuestionDetail,
  GraphData,
  ChatResponse,
} from '@/types';

const BASE_URL = '';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${url}`, init);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const api = {
  health: () => fetchJson<HealthResponse>('/health'),

  getRunInfo: () => fetchJson<RunInfo>('/api/run/info'),

  getRunScores: () => fetchJson<LeaderboardScores>('/api/run/scores'),

  getQuestions: () => fetchJson<QuestionSummary[]>('/api/questions'),

  getQuestion: (qid: string) => fetchJson<QuestionDetail>(`/api/questions/${qid}`),

  getGraph: (qid: string) => fetchJson<GraphData>(`/api/graphs/${qid}`),

  getImageUrl: (imageId: string) => `/api/images/${imageId}`,

  chat: (question: string) =>
    fetchJson<ChatResponse>('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    }),
};
