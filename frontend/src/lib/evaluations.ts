import type {
  BlockOut,
  ChunkOut,
  DocumentSummary,
  EvidencePack,
  PageInspection,
} from "./types";

// ----------------------------------------------------------------------
// Evaluation Lab contracts (mirror data/evaluation artifacts)

export interface EvalLevelMetrics {
  hit_at_k: Record<string, number>;
  recall_at_k: Record<string, number>;
  precision_at_k: Record<string, number>;
  mrr: number;
  ndcg_at_k: Record<string, number>;
}

export interface EvalModeAggregate {
  mode: string;
  n_questions: number;
  avg_latency_ms: number | null;
  chunk_metrics: EvalLevelMetrics;
  page_metrics: EvalLevelMetrics;
}

export interface GenerationEvalStatus {
  status: string;
  note: string;
}

export interface EvalFailureCase {
  category: string;
  question_id: string;
  question: string;
  detail: {
    ground_truth_chunk_ids?: string[];
    relevant_pages?: number[];
    [key: string]: unknown;
  };
}

export interface EvalRunListItem {
  run_id: string;
  timestamp: string;
  dataset_version: string | null;
  modes: string[];
  run_name: string | null;
  path: string;
}

export interface EvalPerQueryResult {
  question_id: string;
  question: string;
  mode: string;
  retrieved_chunk_ids: string[];
  relevant_chunk_ids: string[];
  first_relevant_rank_chunk: number | null;
  chunk_metrics: EvalLevelMetrics;
  page_metrics: EvalLevelMetrics;
}

export interface EvalRunDetail {
  run_id: string;
  timestamp: string;
  dataset_version: string;
  per_query: Record<string, EvalPerQueryResult[]>;
  failures: EvalFailureCase[];
  evidence_evals?: unknown[];
}

export interface EvalLatestSummary {
  run_id: string;
  timestamp: string;
  dataset_version: string;
  dataset_fingerprint: string;
  total_time_s: number | null;
  avg_query_time_ms: number | null;
  aggregates: EvalModeAggregate[];
  failure_counts: Record<string, number>;
  evidence_summary: Record<string, number>;
  generation_eval: GenerationEvalStatus;
}

// Re-exported for convenience
export type {
  BlockOut,
  ChunkOut,
  DocumentSummary,
  EvidencePack,
  PageInspection,
};
