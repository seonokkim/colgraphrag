export interface ScoreSet {
  qa_fl: number;
  qa_acc: number;
  qa: number;
}

export interface QcateScores {
  by_qcate_qa_fl: Record<string, number>;
  by_qcate_qa_acc: Record<string, number>;
  by_qcate_qa: Record<string, number>;
}

export interface LeaderboardScores {
  all: ScoreSet;
  unimodal: ScoreSet;
  multimodal: ScoreSet;
  qcate: QcateScores | null;
  source: string | null;
}

export interface RunInfo {
  run_id: string;
  generated_at: string | null;
  total_questions: number;
  scored_questions: number;
  predictions_path: string;
  gold_jsonl_path: string;
  result_run_dir: string;
  git_commit: string | null;
}

export interface GoldFact {
  fact_type: 'image' | 'text';
  id: string;
  content: string;
  title: string | null;
  caption: string | null;
}

export interface RetrievalItem {
  id: string;
  score: number;
  rank: number | null;
  source: string | null;
}

export interface QuestionSummary {
  qid: string;
  question: string;
  qcate: string;
  gold_answer: string;
  predicted_answer: string | null;
  has_graph: boolean;
}

export interface QuestionDetail {
  qid: string;
  question: string;
  qcate: string;
  split: string;
  gold_answers: string[];
  keywords_answer: string | null;
  predicted_answer: string | null;
  retrieval: RetrievalItem[];
  gold_facts: GoldFact[];
  graph_available: boolean;
}

export interface GraphNode {
  id: string;
  entity_name: string | null;
  node_type: string | null;
  description: string | null;
  source_id: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number | null;
  description: string | null;
  source_id: string | null;
}

export interface GraphData {
  qid: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
}

export interface HealthResponse {
  status: string;
  run_id: string | null;
  message: string | null;
}

export type DatasetKey = 'webqa' | 'mmqa';

export type ModelKey =
  | 'hf_gemma4_e4b'
  | 'ollama_gemma4_e2b'
  | 'ollama_gemma4_e4b';

export const MODEL_OPTIONS: { key: ModelKey; label: string }[] = [
  { key: 'hf_gemma4_e4b', label: 'HF Gemma 4 E4B' },
  { key: 'ollama_gemma4_e2b', label: 'Ollama Gemma 4 E2B' },
  { key: 'ollama_gemma4_e4b', label: 'Ollama Gemma 4 E4B' },
];

export interface DatasetInfo {
  key: DatasetKey;
  label: string;
  available: boolean;
  run_id: string | null;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
  default: DatasetKey;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: RetrievalItem[];
  graphData?: GraphData;
  goldFacts?: GoldFact[];
  loading?: boolean;
  /** Round-trip / generation time in ms (from API or client fallback). */
  elapsedMs?: number;
}

export interface ChatResponse {
  answer: string;
  sources: RetrievalItem[];
  graph: GraphData | null;
  gold_facts: GoldFact[];
  matched_qid?: string | null;
  elapsed_ms?: number | null;
}
