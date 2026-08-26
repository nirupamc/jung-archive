import type {
  EvalLatestSummary,
  EvalRunDetail,
  EvalRunListItem,
} from "@/lib/evaluations";

function levelMetrics(hit1: number, recall5: number, mrr: number,
                      ndcg5: number, recall10: number) {
  return {
    hit_at_k: { "1": hit1, "3": 0.5, "5": 0.5, "10": recall10 },
    recall_at_k: { "1": hit1, "3": recall5, "5": recall5, "10": recall10 },
    precision_at_k: { "1": hit1, "3": 0.2, "5": 0.15, "10": 0.08 },
    ndcg_at_k: { "1": hit1, "3": ndcg5, "5": ndcg5, "10": ndcg5 },
    mrr,
  };
}

export const LATEST_SUMMARY: EvalLatestSummary = {
  run_id: "aaa111-run-b",
  timestamp: "2026-08-25T12:00:00+00:00",
  dataset_version: "undiscovered-self-benchmark-1",
  dataset_fingerprint: "3410af1e3a4a82cc",
  total_time_s: 158.0,
  avg_query_time_ms: 753.0,
  aggregates: [
    {
      mode: "dense",
      n_questions: 30,
      avg_latency_ms: 120.0,
      chunk_metrics: levelMetrics(0.433, 0.606, 0.602, 0.524, 0.689),
      page_metrics: levelMetrics(0, 0, 0, 0, 0),
    },
    {
      mode: "bm25",
      n_questions: 30,
      avg_latency_ms: 30.0,
      chunk_metrics: levelMetrics(0.567, 0.711, 0.685, 0.648, 0.817),
      page_metrics: levelMetrics(0, 0, 0, 0, 0),
    },
    {
      mode: "hybrid",
      n_questions: 30,
      avg_latency_ms: 200.0,
      chunk_metrics: levelMetrics(0.5, 0.631, 0.654, 0.567, 0.803),
      page_metrics: levelMetrics(0, 0, 0, 0, 0),
    },
    {
      mode: "hybrid_rerank",
      n_questions: 30,
      avg_latency_ms: 2400.0,
      chunk_metrics: levelMetrics(0.767, 0.783, 0.853, 0.761, 0.847),
      page_metrics: levelMetrics(0.8, 0.85, 0.9, 0.8, 0.9),
    },
  ],
  failure_counts: {
    bm25_only_win: 4,
    reranker_improves_rank: 10,
    reranker_hurts_rank: 2,
  },
  evidence_summary: {
    evidence_accuracy_chunk: 0.184,
    evidence_coverage_chunk: 0.719,
  },
  generation_eval: {
    status: "NOT_RUN",
    note: "generation is outside the current M1-M6 system.",
  },
};

export const EVAL_RUNS: EvalRunListItem[] = [
  {
    run_id: "bbb222-run-a",
    timestamp: "2026-08-25T11:00:00+00:00",
    dataset_version: "undiscovered-self-benchmark-1",
    modes: ["hybrid"],
    run_name: "hybrid-only",
    path: "data/evaluation/runs/bbb222-run-a.json",
  },
  {
    run_id: "aaa111-run-b",
    timestamp: "2026-08-25T12:00:00+00:00",
    dataset_version: "undiscovered-self-benchmark-1",
    modes: ["dense", "bm25", "hybrid", "hybrid_rerank"],
    run_name: "m6-baseline",
    path: "data/evaluation/runs/aaa111-run-b.json",
  },
];

const perQuery = (mode: string, qid: string, rank: number | null) => ({
  question_id: qid,
  question: `Question ${qid}`,
  mode,
  retrieved_chunk_ids:
    rank === null ? ["x1"] : ["381d2da4b68e-c00029", "x1"],
  relevant_chunk_ids: ["381d2da4b68e-c00029"],
  first_relevant_rank_chunk: rank,
  chunk_metrics: levelMetrics(rank === 1 ? 1 : 0, 0, rank ? 1 / rank : 0,
                              0, 0),
  page_metrics: levelMetrics(0, 0, 0, 0, 0),
});

export const EVAL_RUN_A: EvalRunDetail & {
  aggregates: Array<Record<string, unknown>>;
} = {
  run_id: "bbb222-run-a",
  timestamp: "2026-08-25T11:00:00+00:00",
  dataset_version: "undiscovered-self-benchmark-1",
  per_query: {},
  failures: [],
  aggregates: [
    {
      mode: "hybrid",
      n_questions: 30,
      avg_latency_ms: 200,
      chunk_metrics: levelMetrics(0.5, 0.631, 0.654, 0.567, 0.803),
      page_metrics: levelMetrics(0, 0, 0, 0, 0),
    },
    {
      mode: "hybrid_rerank",
      n_questions: 30,
      avg_latency_ms: 2500,
      chunk_metrics: levelMetrics(0.45, 0.6, 0.6, 0.55, 0.78),
      page_metrics: levelMetrics(0, 0, 0, 0, 0),
    },
    // RUN A's weaker reranker vs RUN B -> positive deltas + shared modes
  ],
};

export const EVAL_RUN_B: EvalRunDetail & {
  aggregates: Array<Record<string, unknown>>;
} = {
  run_id: "aaa111-run-b",
  timestamp: "2026-08-25T12:00:00+00:00",
  dataset_version: "undiscovered-self-benchmark-1",
  per_query: {
    dense: [perQuery("dense", "qb", null)],
    bm25: [perQuery("bm25", "qa", 2), perQuery("bm25", "qb", null)],
  },
  failures: [
    {
      category: "bm25_only_win",
      question_id: "qa",
      question: "Question qa about mass psychology?",
      detail: { ground_truth_chunk_ids: ["381d2da4b68e-c00029"] },
    },
    {
      category: "all_methods_fail",
      question_id: "qb",
      question: "Question qb nobody can find.",
      detail: { ground_truth_chunk_ids: ["381d2da4b68e-c00029"] },
    },
  ],
  evidence_evals: [],
  aggregates: [
    {
      mode: "hybrid",
      n_questions: 30,
      avg_latency_ms: 210,
      chunk_metrics: levelMetrics(0.5, 0.7, 0.7, 0.6, 0.8),
      page_metrics: levelMetrics(0, 0, 0, 0, 0),
    },
    {
      mode: "hybrid_rerank",
      n_questions: 30,
      avg_latency_ms: 2400,
      chunk_metrics: levelMetrics(0.767, 0.783, 0.853, 0.761, 0.847),
      page_metrics: levelMetrics(0, 0, 0, 0, 0),
    },
  ],
};
