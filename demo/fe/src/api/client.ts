import type {
  HealthResponse,
  RunInfo,
  LeaderboardScores,
  QuestionSummary,
  QuestionDetail,
  GraphData,
  ChatResponse,
  DatasetsResponse,
  DatasetKey,
  ModelKey,
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

  getDatasets: () => fetchJson<DatasetsResponse>('/api/datasets'),

  getRunInfo: (dataset: DatasetKey = 'webqa') =>
    fetchJson<RunInfo>(`/api/run/info?dataset=${dataset}`),

  getRunScores: (dataset: DatasetKey = 'webqa') =>
    fetchJson<LeaderboardScores>(`/api/run/scores?dataset=${dataset}`),

  getQuestions: (dataset: DatasetKey = 'webqa') =>
    fetchJson<QuestionSummary[]>(`/api/questions?dataset=${dataset}`),

  getQuestion: (qid: string, dataset: DatasetKey = 'webqa') =>
    fetchJson<QuestionDetail>(`/api/questions/${qid}?dataset=${dataset}`),

  getGraph: (qid: string, dataset: DatasetKey = 'webqa') =>
    fetchJson<GraphData>(`/api/graphs/${qid}?dataset=${dataset}`),

  getImageUrl: (imageId: string, dataset: DatasetKey = 'webqa') =>
    `/api/images/${imageId}?dataset=${dataset}`,

  chat: (question: string, dataset: DatasetKey = 'webqa', model: ModelKey = 'hf_gemma4_e4b') =>
    fetchJson<ChatResponse>('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, dataset, model }),
    }),
};
