"""Canonical retrieval result models (M3).

Every result preserves full provenance (chunk -> blocks -> pages ->
document) plus its complete scoring path through each retrieval leg.
Raw scores from different systems are NEVER collapsed into one number;
fusion uses ranks only.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from jung_archive.models.document import SourceType


class RetrievalResult(BaseModel):
    """One ranked retrieval hit with full provenance and score path."""
    chunk_id: str
    document_id: str
    text: str
    page_numbers: List[int]
    source_block_ids: List[str]
    heading_path: List[str] = []
    source_type: SourceType

    # Dense leg (None when the chunk was not retrieved densely)
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None   # cosine similarity in [-1, 1]

    # Lexical leg (None when the chunk was not retrieved by BM25)
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None    # raw BM25 score (NOT comparable to cosine)

    # Fusion outcome
    fusion_rank: Optional[int] = None
    fusion_score: Optional[float] = None  # RRF score

    # Reranking stage (M4). None when the chunk was never reranked.
    reranker_rank: Optional[int] = None
    reranker_score: Optional[float] = None

    # Extra context (author/title when available)
    author: Optional[str] = None
    title: Optional[str] = None
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    def preview(self, n: int = 160) -> str:
        return " ".join(self.text.split())[:n]


class RetrievalResponse(BaseModel):
    """Structured response for one hybrid/plain query."""
    query: str
    mode: str                      # "dense" | "bm25" | "hybrid" | "hybrid_rerank"
    top_k: int
    filters: Dict[str, Any] = {}
    results: List[RetrievalResult] = []
    warnings: List[str] = []
    latency_ms: Optional[float] = None
    # M4 reranking telemetry (None for non-reranked modes)
    candidates_retrieved: Optional[int] = None
    candidates_reranked: Optional[int] = None
    pairs_truncated: Optional[int] = None
