"""Reranking retrieval pipeline (M4).

QUERY -> dense + BM25 -> RRF -> candidate pool (fusion_candidate_k)
      -> cross-encoder reranking -> top rerank_top_k evidence candidates.

The existing plain modes (dense | bm25 | hybrid) are untouched; this
pipeline adds a separate "hybrid_rerank" path so M6 evaluation can
compare all four orderings. Reranker failure is explicit: no silent
fallback to unreranked ordering unless allow_reranker_fallback is set.
"""
import time
from typing import List, Optional

from jung_archive.indexing.vector_index import VectorIndex
from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
from jung_archive.retrieval.lexical import BM25Retriever
from jung_archive.retrieval.results import RetrievalResult, RetrievalResponse

VALID_RERANK_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class RerankingPipelineConfig:
    def __init__(
        self,
        dense_candidate_k: int = 30,
        bm25_candidate_k: int = 30,
        rrf_k: int = 60,
        fusion_candidate_k: int = 20,
        rerank_top_k: int = 8,
        mode: str = "hybrid_rerank",
        allow_reranker_fallback: bool = False,
    ):
        if mode not in VALID_RERANK_MODES:
            raise ValueError(
                f"invalid mode {mode!r}; expected one of {VALID_RERANK_MODES}")
        for name, v in (("dense_candidate_k", dense_candidate_k),
                        ("bm25_candidate_k", bm25_candidate_k),
                        ("fusion_candidate_k", fusion_candidate_k),
                        ("rerank_top_k", rerank_top_k)):
            if v < 1:
                raise ValueError(f"{name} must be >= 1")
        if fusion_candidate_k < rerank_top_k:
            raise ValueError(
                "fusion_candidate_k must be >= rerank_top_k")
        self.dense_candidate_k = dense_candidate_k
        self.bm25_candidate_k = bm25_candidate_k
        self.rrf_k = rrf_k
        self.fusion_candidate_k = fusion_candidate_k
        self.rerank_top_k = rerank_top_k
        self.mode = mode
        # Explicit opt-in only; any fallback is reported in warnings.
        self.allow_reranker_fallback = allow_reranker_fallback


class RerankingPipeline:
    """Owns a HybridRetriever configured at pool depth and a Reranker."""

    def __init__(
        self,
        vector_index: VectorIndex,
        bm25: BM25Retriever,
        reranker,
        config: Optional[RerankingPipelineConfig] = None,
    ):
        from jung_archive.reranking.base import Reranker

        if not isinstance(reranker, Reranker):
            raise TypeError(
                "reranker must implement jung_archive.reranking.Reranker")
        self.config = config or RerankingPipelineConfig()
        cfg = self.config
        self.retriever = HybridRetriever(
            vector_index,
            bm25,
            HybridRetrieverConfig(
                dense_candidate_k=cfg.dense_candidate_k,
                bm25_candidate_k=cfg.bm25_candidate_k,
                rrf_k=cfg.rrf_k,
                final_top_k=cfg.fusion_candidate_k,
                mode="hybrid",
            ),
        )
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> RetrievalResponse:
        started = time.perf_counter()
        cfg = self.config
        k = cfg.rerank_top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")

        warnings: List[str] = []
        base_mode = cfg.mode.replace("hybrid_rerank", "hybrid")
        base = self.retriever.search(
            query, top_k=cfg.fusion_candidate_k, filters=filters,
            mode=base_mode,
        )
        warnings.extend(base.warnings)

        candidates = base.results
        results: List[RetrievalResult] = []
        pairs_truncated: Optional[int] = None

        if not candidates:
            warnings.append("empty candidate pool; nothing to rerank")
        else:
            try:
                ranked, report = self.reranker.rank_results(query, candidates)
                pairs_truncated = report.truncated_documents + \
                    report.truncated_queries
                if report.truncated_documents or report.truncated_queries:
                    warnings.append(
                        f"reranker sequence-limit truncation applied "
                        f"({report.summary()})")
                results = ranked[:k]
            except Exception as e:
                if not cfg.allow_reranker_fallback:
                    raise
                warnings.append(
                    f"reranker failed ({e}); fell back to unreranked "
                    f"fusion ordering")
                results = candidates[:k]

        latency_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalResponse(
            query=query,
            mode="hybrid_rerank",
            top_k=k,
            filters=filters or {},
            results=results,
            warnings=warnings,
            latency_ms=round(latency_ms, 2),
            candidates_retrieved=len(candidates),
            candidates_reranked=len(candidates),
            pairs_truncated=pairs_truncated,
        )
