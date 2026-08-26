from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
import hashlib
import json


class SourceType(str, Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    UNKNOWN = "UNKNOWN"


class IndexStatus(str, Enum):
    """Explicit curation decision for whether a document enters the index."""
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    REVIEW = "REVIEW"


class PageClassification(str, Enum):
    NATIVE = "NATIVE"
    OCR_REQUIRED = "OCR_REQUIRED"
    HYBRID = "HYBRID"
    EMPTY = "EMPTY"
    SUSPICIOUS = "SUSPICIOUS"
    FAILED = "FAILED"


class LayoutType(str, Enum):
    SINGLE_COLUMN = "SINGLE_COLUMN"
    TWO_COLUMN = "TWO_COLUMN"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class BlockType(str, Enum):
    TITLE = "TITLE"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST = "LIST"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    CAPTION = "CAPTION"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    PAGE_NUMBER = "PAGE_NUMBER"
    UNKNOWN = "UNKNOWN"


class ExtractionMethod(str, Enum):
    NATIVE = "NATIVE"
    OCR = "OCR"
    HYBRID = "HYBRID"
    NONE = "NONE"
    FAILED = "FAILED"


class BoundingBox(BaseModel):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(gt=0)
    y1: float = Field(gt=0)

    @model_validator(mode="after")
    def check_orientation(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"inverted/degenerate bbox: ({self.x0}, {self.y0}, {self.x1}, {self.y1})"
            )
        return self

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0

    def area(self) -> float:
        return self.width() * self.height()

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class Block(BaseModel):
    """Block-level record with full provenance.

    Confidence semantics (M2):
      - ``confidence``: MEASURED extraction certainty only (e.g. mean OCR
        word confidence). Native PDF extraction yields no measured value,
        so it stays None rather than a fabricated prior.
      - ``heuristic_quality_score``: explicitly heuristic rating of how
        strong the evidence was for the block-type assignment. Never a
        claim about extraction fidelity.
    """
    block_id: str
    block_type: BlockType
    text: str
    bbox: BoundingBox
    reading_order: int
    extraction_method: ExtractionMethod
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    heuristic_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    flags: Optional[int] = None
    number: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "reading_order": self.reading_order,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
        }
        # Preserve native font provenance when available
        if self.font_name is not None:
            d["font_name"] = self.font_name
        if self.font_size is not None:
            d["font_size"] = self.font_size
        if self.number is not None:
            d["number"] = self.number
        if self.heuristic_quality_score is not None:
            d["heuristic_quality_score"] = self.heuristic_quality_score
        return d


class Page(BaseModel):
    page_number: int
    width: float
    height: float
    classification: PageClassification
    classification_confidence: float = Field(ge=0.0, le=1.0)
    layout: LayoutType
    layout_confidence: float = Field(ge=0.0, le=1.0)
    blocks: List[Block] = []
    warnings: List[str] = []
    reason: Optional[str] = None
    layout_reason: Optional[str] = None
    # Measured mean OCR word confidence; None when OCR was not used.
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def to_dict(self) -> dict:
        d = {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "reason": self.reason,
            "layout": self.layout,
            "layout_confidence": self.layout_confidence,
            "layout_reason": self.layout_reason,
            "blocks": [b.to_dict() for b in self.blocks],
            "warnings": self.warnings,
        }
        if self.ocr_confidence is not None:
            d["ocr_confidence"] = self.ocr_confidence
        return d


class Document(BaseModel):
    document_id: str
    title: str
    # Author is NEVER defaulted: identity must come from verified metadata
    # or the explicit registry, not assumption.
    author: Optional[str] = None
    source_type: SourceType
    source_path: str
    page_count: int
    pages: List[Page] = []
    index_status: IndexStatus = IndexStatus.REVIEW
    source_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "document_id": self.document_id,
            "title": self.title,
            "author": self.author,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "page_count": self.page_count,
            "pages": [p.to_dict() for p in self.pages],
        }
        if self.source_sha256 is not None:
            d["source_sha256"] = self.source_sha256
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class Diagnostics(BaseModel):
    document_id: str
    page_count: int
    classification_counts: dict
    layout_counts: dict
    block_counts: dict
    # Mean of MEASURED extraction confidences; None when none were measured.
    average_extraction_confidence: Optional[float] = None
    measured_confidence_block_count: int = 0
    warnings: List[str]

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "page_count": self.page_count,
            "classification_counts": self.classification_counts,
            "layout_counts": self.layout_counts,
            "block_counts": self.block_counts,
            "average_extraction_confidence": self.average_extraction_confidence,
            "measured_confidence_block_count": self.measured_confidence_block_count,
            "warnings": self.warnings,
        }