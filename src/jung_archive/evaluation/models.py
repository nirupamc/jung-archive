"""Canonical evaluation models (M6)."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class BenchmarkItem(BaseModel):
    """One manually grounded benchmark question."""
    id: str
    question: str = Field(min_length=1)
    relevant_chunk_ids: List[str] = []
    relevant_page_numbers: List[int] = []
    relevant_document_ids: List[str] = []
    reference_answer: Optional[str] = None
    notes: str = ""
    tags: List[str] = []

    @model_validator(mode="after")
    def check_relevance(self):
        if not self.relevant_chunk_ids and not self.relevant_page_numbers:
            raise ValueError(
                f"benchmark item {self.id!r} has no relevance labels")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError(
                f"benchmark item {self.id!r} has duplicate ground-truth "
                f"chunk ids")
        return self


class DatasetMeta(BaseModel):
    dataset_version: str = "undiscovered-self-benchmark-1"
    chunking_config_version: str = ""
    document_sha256: Dict[str, str] = {}   # document_id -> sha256
    created: str = ""
    ground_truth_method: str = (
        "manual source inspection of The Undiscovered Self PDF and "
        "chunk artifacts; never derived from the retriever under test")


class BenchmarkDataset(BaseModel):
    meta: DatasetMeta
    items: List[BenchmarkItem]

    @model_validator(mode="after")
    def check_unique_ids(self):
        ids = [i.id for i in self.items]
        if any(not i.strip() for i in ids):
            raise ValueError("benchmark item ids must be non-empty")
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        return self

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.meta.dataset_version.encode("utf-8"))
        for item in sorted(self.items, key=lambda x: x.id):
            h.update(item.model_dump_json().encode("utf-8"))
        return h.hexdigest()[:16]


class ExperimentConfig(BaseModel):
    run_name: str = "baseline"
    modes: List[str] = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    k_values: List[int] = [1, 3, 5, 10]
    dense_candidate_k: int = 30
    bm25_candidate_k: int = 30
    rrf_k: int = 60
    fusion_candidate_k: int = 20
    rerank_top_k: int = 10
    # corpus namespace (default production corpus)
    chunks_dir: str = "data/chunks"
    chroma_dir: str = "data/chroma"
    bm25_state_dir: str = "data/bm25"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunking_config_version: str = ""
    dataset_fingerprint: str = ""
    record_latencies: bool = True   # False => fully deterministic output
    notes: str = ""

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class LevelMetrics(BaseModel):
    """Metrics at one relevance level (chunk or page)."""
    hit_at_k: Dict[str, float] = {}
    recall_at_k: Dict[str, float] = {}
    precision_at_k: Dict[str, float] = {}
    mrr: float = 0.0
    ndcg_at_k: Dict[str, float] = {}


class PerQueryResult(BaseModel):
    question_id: str
    question: str
    mode: str
    retrieved_chunk_ids: List[str]
    retrieved_page_numbers: List[int] = []
    relevant_chunk_ids: List[str] = []
    relevant_page_numbers: List[int] = []
    first_relevant_rank_chunk: Optional[int] = None
    first_relevant_rank_page: Optional[int] = None
    chunk_metrics: LevelMetrics
    page_metrics: LevelMetrics
    score_paths: List[Dict[str, Any]] = []   # per-result score path snapshot
    latency_ms: Optional[float] = None


class ModeAggregate(BaseModel):
    mode: str
    n_questions: int
    chunk_metrics: LevelMetrics
    page_metrics: LevelMetrics
    avg_latency_ms: Optional[float] = None


class FailureCase(BaseModel):
    category: str
    question_id: str
    question: str
    detail: Dict[str, Any] = {}


class EvidenceEvalItem(BaseModel):
    question_id: str
    n_items: int
    tokens_used: int
    max_tokens: int
    # accuracy = relevant selected / all selected (chunk-level + page-level)
    evidence_accuracy_chunk: float
    evidence_accuracy_page: float
    # coverage = ground-truth represented / total ground truth
    evidence_coverage_chunk: float
    evidence_coverage_page: float


class GenerationEvaluationRecord(BaseModel):
    """Optional contract for future answer-generation evaluation.

    M6 does NOT include a generation system; this schema exists so later
    milestones can fill it without changing the run format.
    """
    status: str = "NOT_RUN"
    note: str = "generation is outside the current M1-M6 system."
    question_id: str = ""
    reference_answer: Optional[str] = None
    generated_answer: Optional[str] = None
    citations: List[str] = []
    faithfulness_score: Optional[float] = None
    answer_correctness_score: Optional[float] = None


class RunRecord(BaseModel):
    run_id: str
    timestamp: str
    dataset_path: str
    dataset_version: str
    dataset_fingerprint: str
    config: ExperimentConfig
    aggregates: List[ModeAggregate] = []
    per_query: Dict[str, List[PerQueryResult]] = {}   # mode -> results
    failures: List[FailureCase] = []
    evidence_evals: List[EvidenceEvalItem] = []
    generation_eval: GenerationEvaluationRecord = Field(
        default_factory=GenerationEvaluationRecord)
    total_time_s: Optional[float] = None
    avg_query_time_ms: Optional[float] = None
