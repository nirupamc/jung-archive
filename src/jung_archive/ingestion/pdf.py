import fitz  # PyMuPDF
import hashlib
import os
from typing import List, Optional, Tuple
import logging
from jung_archive.models.document import (
    Document, Page, Block, BoundingBox,
    PageClassification, LayoutType, BlockType, ExtractionMethod,
    SourceType
)
from jung_archive.analysis.signals import PageSignalExtractor
from jung_archive.analysis.classifier import PageClassifier
from jung_archive.extraction.native import NativeExtractor
from jung_archive.extraction.ocr import OCRExtractor
from jung_archive.layout.analyzer import LayoutAnalyzer
from jung_archive.structure.analyzer import StructureAnalyzer
from jung_archive.metadata import DEFAULT_REGISTRY_PATH, MetadataRegistry, sha256_of_file

logger = logging.getLogger(__name__)

class PDFIngestor:
    """
    Ingests a PDF and produces a structured document representation.
    """

    def __init__(self, enable_ocr: bool = True, registry: Optional[MetadataRegistry] = None):
        self.enable_ocr = enable_ocr
        self.page_signals = PageSignalExtractor()
        self.page_classifier = PageClassifier()
        self.native_extractor = NativeExtractor()
        self.ocr_extractor = OCRExtractor() if enable_ocr else None
        self.layout_analyzer = LayoutAnalyzer()
        self.structure_analyzer = StructureAnalyzer()
        self.registry = registry if registry is not None else MetadataRegistry.load(
            DEFAULT_REGISTRY_PATH
        )

    def ingest(self, pdf_path: str, document_id: Optional[str] = None) -> Document:
        """
        Main ingestion method.
        """
        logger.info(f"Starting ingestion of {pdf_path}")
        doc = fitz.open(pdf_path)

        # Generate document_id if not provided
        if document_id is None:
            document_id = self._generate_document_id(pdf_path)

        # Tentative source type from folder; the registry has final say.
        source_type = self._determine_source_type(pdf_path)

        # Process each page
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                processed_page = self._process_page(page, page_num + 1)
                pages.append(processed_page)
            except Exception as e:
                logger.error(f"Failed to process page {page_num + 1}: {e}")
                # Create a failed page record
                failed_page = Page(
                    page_number=page_num + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    classification=PageClassification.FAILED,
                    classification_confidence=0.0,
                    layout=LayoutType.UNKNOWN,
                    layout_confidence=0.0,
                    warnings=[f"Processing failed: {str(e)}"]
                )
                pages.append(failed_page)

        doc.close()

        # Create document object; author/title start unasserted and are then
        # filled from embedded PDF metadata (weak evidence) and finally the
        # explicit registry (strongest).
        pdf_meta_title, pdf_meta_author = self._embedded_metadata(pdf_path)
        document = Document(
            document_id=document_id,
            title=pdf_meta_title or self._extract_title(pdf_path),
            author=pdf_meta_author,
            source_type=source_type,
            source_path=pdf_path,
            page_count=len(pages),
            pages=pages,
            index_status=self.registry_index_status_default(),
        )
        document.source_sha256 = sha256_of_file(pdf_path)
        self.registry.apply(document)

        logger.info(f"Ingestion complete. Processed {len(pages)} pages.")
        return document

    def _process_page(self, page: fitz.Page, page_number: int) -> Page:
        """
        Process a single page.
        """
        # Get page dimensions
        width = page.rect.width
        height = page.rect.height

        # Extract signals
        signals = self.page_signals.extract(page)

        # Classify page
        classification, confidence, reason = self.page_classifier.classify(signals)

        # Extract blocks based on classification
        blocks = []
        warnings = []
        extraction_note = None
        ocr_confidence = None
        if classification == PageClassification.NATIVE:
            blocks.extend(self.native_extractor.extract(page))
        elif classification == PageClassification.HYBRID:
            blocks.extend(self.native_extractor.extract(page))
            # Conservative M1 behavior: raster regions are NOT silently
            # substituted; note that image content was left unprocessed.
            extraction_note = (
                "hybrid page: native text preserved; raster/image content "
                "not extracted (M1 conservative policy)"
            )
        elif classification == PageClassification.OCR_REQUIRED:
            if self.ocr_extractor is not None and self.ocr_extractor.available:
                ocr_blocks = self.ocr_extractor.extract(page)
                blocks.extend(ocr_blocks)
                if self.ocr_extractor.last_error:
                    warnings.append(self.ocr_extractor.last_error)
                ocr_confidence = self.ocr_extractor.last_mean_confidence
            elif self.ocr_extractor is not None:
                warnings.append(
                    f"page requires OCR but backend unavailable: "
                    f"{self.ocr_extractor.unavailable_reason}"
                )
            else:
                warnings.append("page requires OCR but OCR is disabled (--no-ocr)")
        elif classification == PageClassification.SUSPICIOUS:
            # Safe attempt at native extraction, retain warning
            blocks.extend(self.native_extractor.extract(page))
        # EMPTY and FAILED pages get no artificial text

        if extraction_note:
            warnings.append(extraction_note)

        # Detect layout (before reading order so order can respect columns)
        layout, layout_confidence, layout_reason = self.layout_analyzer.detect(
            blocks, page.rect, page=page
        )

        # Assign reading order (layout-aware: columns read sequentially)
        blocks = self._assign_reading_order(blocks, width, layout)

        # Classify blocks (structure)
        blocks = self.structure_analyzer.analyze(blocks, page)

        # Create page object
        processed_page = Page(
            page_number=page_number,
            width=width,
            height=height,
            classification=classification,
            classification_confidence=confidence,
            layout=layout,
            layout_confidence=layout_confidence,
            blocks=blocks,
            warnings=warnings,
            reason=reason,
            layout_reason=layout_reason,
            ocr_confidence=ocr_confidence,
        )

        return processed_page

    @staticmethod
    def _assign_reading_order(
        blocks: List[Block], page_width: float, layout: LayoutType
    ) -> List[Block]:
        """
        Assign deterministic reading-order indexes (1-based).

        SINGLE_COLUMN / MIXED / UNKNOWN: top-to-bottom, left-to-right.
        TWO_COLUMN: left column top-to-bottom first, then right column,
        split at the page's horizontal midline (conservative M1 policy).
        """
        def yx(block: Block) -> Tuple[float, float]:
            return (round(block.bbox.y0, 1), round(block.bbox.x0, 1))

        if layout == LayoutType.TWO_COLUMN and page_width > 0:
            mid = page_width / 2.0
            left = sorted((b for b in blocks if b.bbox.x1 <= mid or
                           (b.bbox.x0 + b.bbox.x1) / 2 < mid), key=yx)
            right = sorted((b for b in blocks if b not in left), key=yx)
            ordered = left + right
        else:
            ordered = sorted(blocks, key=yx)

        for i, block in enumerate(ordered, start=1):
            block.reading_order = i
        return ordered

    @staticmethod
    def _generate_document_id(pdf_path: str) -> str:
        """
        Deterministic document ID: SHA-256 of absolute file path (12 hex chars).
        """
        return hashlib.sha256(os.path.abspath(pdf_path).encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _determine_source_type(pdf_path: str) -> SourceType:
        """
        Tentative source type from corpus directory layout only.
        The explicit metadata registry always overrides this.
        """
        if "primary" in pdf_path.lower():
            return SourceType.PRIMARY
        return SourceType.SECONDARY

    @staticmethod
    def _embedded_metadata(pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Weak evidence: embedded PDF title/author, when present."""
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata or {}
            doc.close()
        except Exception:
            return None, None
        title = (meta.get("title") or "").strip() or None
        author = (meta.get("author") or "").strip() or None
        return title, author

    @staticmethod
    def _extract_title(pdf_path: str) -> str:
        """Fallback title: filename stem."""
        return os.path.splitext(os.path.basename(pdf_path))[0]

    @staticmethod
    def registry_index_status_default():
        from jung_archive.models.document import IndexStatus
        return IndexStatus.REVIEW