from typing import Any, Dict, List, Optional
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from jung_archive.models.document import SourceType


class Chunk(BaseModel):
    """Retrieval chunk with full source provenance.

    A chunk is always traceable to its document and to the exact source
    block IDs and pages that produced it.
    """
    chunk_id: str
    document_id: str
    text: str = Field(min_length=1)  # no empty chunks
    source_block_ids: List[str]
    page_numbers: List[int]
    heading_path: List[str] = []
    token_count: int
    source_type: SourceType
    section_id: Optional[str] = None
    chunk_index: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    char_count: Optional[int] = None
    strategy: Optional[str] = None
    created_from_blocks: Optional[List[str]] = None
    metadata: Dict[str, Any] = {}

    @model_validator(mode="after")
    def check_provenance_basics(self):
        if not self.source_block_ids:
            raise ValueError(f"chunk {self.chunk_id} has no source blocks")
        if not self.page_numbers:
            raise ValueError(f"chunk {self.chunk_id} has no source pages")
        return self

    def to_dict(self) -> dict:
        d = self.model_dump(mode="json")
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


class ChunkingConfig(BaseModel):
    """Structure-aware chunker configuration.

    Defaults chosen for all-MiniLM-L6-v2 (context ~256 wordpieces):
    target well below truncation while keeping chunks meaningful.
    """
    target_tokens: int = 220
    max_tokens: int = 300
    min_tokens: int = 50
    overlap_tokens: int = 30
    strategy_name: str = "structure_aware_v1"

    # Bump when chunking behavior changes in a way that should trigger
    # re-chunking of indexed documents.
    CONFIG_VERSION: ClassVar[str] = "chunking-config-1"

    @model_validator(mode="after")
    def check_sanity(self):
        if not (0 <= self.overlap_tokens < self.min_tokens <= self.target_tokens):
            raise ValueError("require 0 <= overlap < min <= target tokens")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        return self


class IndexSchemaMeta(BaseModel):
    """Versioned metadata describing index/chunk compatibility."""
    index_schema_version: str = "index-schema-1"
    chunking_config_version: str = ChunkingConfig.CONFIG_VERSION
    embedding_model: str
    embedding_dimension: int
    normalized: bool = True
