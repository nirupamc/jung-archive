"""Canonical evidence models (M4)."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from jung_archive.models.document import SourceType


class ScorePath(BaseModel):
    """Full path of one chunk through retrieval, per stage."""
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    fusion_rank: Optional[int] = None
    fusion_score: Optional[float] = None
    reranker_rank: Optional[int] = None
    reranker_score: Optional[float] = None


class EvidenceItem(BaseModel):
    """One evidence unit for downstream generation.

    `text` is the immutable original chunk text; `clean_text` is a
    derived, source-preserving cleanup. The canonical chunk is never
    mutated.
    """
    evidence_id: str                       # stable display id: S1, S2, ...
    chunk_id: str
    document_id: str
    text: str                              # ORIGINAL chunk text (never mutated)
    clean_text: str                        # derived cleanup output
    page_numbers: List[int]
    source_block_ids: List[str]
    heading_path: List[str] = []
    source_type: SourceType
    author: Optional[str] = None
    title: Optional[str] = None
    section_id: Optional[str] = None

    scores: ScorePath = Field(default_factory=ScorePath)

    token_count: int = 0                   # tokens of clean_text
    was_cleaned: bool = False
    cleanup_operations: List[str] = []
    duplicate_group: Optional[int] = None  # set when dedup merged candidates
    selection_reason: str = "reranked_relevance"

    def pages_display(self) -> str:
        if not self.page_numbers:
            return "?"
        lo, hi = min(self.page_numbers), max(self.page_numbers)
        return str(lo) if lo == hi else f"{lo}-{hi}"

    def preview(self, n: int = 160) -> str:
        return " ".join(self.clean_text.split())[:n]

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceItem":
        return cls(**data)


class SuppressedItem(BaseModel):
    """A candidate removed during assembly, with an explicit reason."""
    chunk_id: str
    reason: str                            # e.g. duplicate_of:S1, diversity_cap, oversized


class EvidencePack(BaseModel):
    """Final deterministic evidence assembly for one question."""
    question: str
    items: List[EvidenceItem] = []
    tokens_used: int = 0
    max_evidence_tokens: int = 0
    max_evidence_items: int = 0
    candidates_considered: int = 0
    suppressed_duplicates: List[SuppressedItem] = []
    suppressed_diversity: List[SuppressedItem] = []
    skipped_oversized: List[SuppressedItem] = []
    warnings: List[str] = []

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "EvidencePack":
        return cls(**data)
