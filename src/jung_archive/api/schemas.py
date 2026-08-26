"""Typed API response contracts (M5).

Backend models are reused wherever they serialize safely
(RetrievalResponse, EvidencePack, Chunk); API-specific views are
defined for document/page/block summaries so no field is invented.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DocumentSummary(BaseModel):
    document_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    source_type: str = "UNKNOWN"
    index_status: str = "REVIEW"
    page_count: int = 0
    chunk_count: int = 0
    source_path: Optional[str] = None
    has_pdf: bool = False
    # Corpus-discovery additions (post-M7). `status` is the honest pipeline
    # state: DISCOVERED/REVIEW/EXCLUDED/PROCESSED/CHUNKED/INDEXED/ERROR.
    status: str = "DISCOVERED"
    section: str = "UNKNOWN"
    registered: bool = False
    registered_reason: Optional[str] = None
    sha256: Optional[str] = None


class CorpusStats(BaseModel):
    discovered_total: int = 0
    pages_total: int = 0
    included: int = 0
    excluded: int = 0
    review: int = 0
    error: int = 0
    by_section: Dict[str, int] = {}
    by_status: Dict[str, int] = {}


class BlockOut(BaseModel):
    block_id: str
    block_type: str
    text: str
    bbox: Dict[str, float]
    reading_order: int
    extraction_method: str
    confidence: Optional[float] = None
    heuristic_quality_score: Optional[float] = None
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    page_number: int


class PageInspection(BaseModel):
    document_id: str
    page_number: int
    width: float
    height: float
    classification: str
    classification_confidence: Optional[float] = None
    classification_reason: Optional[str] = None
    layout: str
    layout_confidence: Optional[float] = None
    layout_reason: Optional[str] = None
    ocr_confidence: Optional[float] = None   # measured OCR only; else null
    warnings: List[str] = []
    blocks: List[BlockOut] = []


class StructureItem(BlockOut):
    pass


class ChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    heading_path: List[str] = []
    page_numbers: List[int]
    token_count: int
    source_type: str
    source_block_ids: List[str]
    strategy: Optional[str] = None
    section_id: Optional[str] = None
    chunk_index: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    char_count: Optional[int] = None
    text: str


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"          # dense | bm25 | hybrid | hybrid_rerank
    top_k: int = 5
    filters: Dict[str, Any] = {}
    # rerank pool tuning (ignored for non-rerank modes)
    fusion_candidate_k: int = 20


class EvidenceRequest(BaseModel):
    question: str
    top_k: int = 8
    max_tokens: int = 2500
    max_items: int = 8
    filters: Dict[str, Any] = {}


class ApiError(BaseModel):
    error: str
    detail: Optional[str] = None
