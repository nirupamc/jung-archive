import fitz
import pytest

from jung_archive.extraction.native import NativeExtractor
from jung_archive.extraction.ocr import OCRExtractor
from jung_archive.ingestion.pdf import PDFIngestor


class TestNativeExtraction:
    def test_extracts_blocks_with_bbox(self, text_pdf):
        doc = fitz.open(text_pdf)
        blocks = NativeExtractor().extract(doc[0])
        doc.close()
        assert len(blocks) >= 2  # heading + body textbox at minimum
        for b in blocks:
            assert b.bbox.x1 > b.bbox.x0
            assert b.bbox.y1 > b.bbox.y0
            assert b.text.strip()
            assert b.extraction_method.value == "NATIVE"

    def test_font_metadata_captured(self, text_pdf):
        doc = fitz.open(text_pdf)
        blocks = NativeExtractor().extract(doc[0])
        doc.close()
        sizes = [b.font_size for b in blocks if b.font_size]
        assert sizes, "expected font size metadata from native extraction"
        assert any(s >= 18 for s in sizes), "heading font should be large"

    def test_block_ids_deterministic_and_unique(self, text_pdf):
        doc = fitz.open(text_pdf)
        blocks_a = NativeExtractor().extract(doc[0])
        blocks_b = NativeExtractor().extract(doc[0])
        doc.close()
        ids_a = [b.block_id for b in blocks_a]
        ids_b = [b.block_id for b in blocks_b]
        assert ids_a == ids_b, "block IDs must be deterministic across runs"
        assert len(ids_a) == len(set(ids_a)), "block IDs must be unique per page"
        assert all(i.startswith("p0001-") for i in ids_a)  # 1-based pages


class TestOCRAvailability:
    def test_missing_backend_is_explicit_not_fatal(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError("simulated missing pytesseract")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        extractor = OCRExtractor()
        assert extractor.available is False
        assert extractor.unavailable_reason

    def test_unavailable_extractor_returns_no_fabricated_text(self, scanned_pdf, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError("simulated missing pytesseract")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        extractor = OCRExtractor()
        doc = fitz.open(scanned_pdf)
        blocks = extractor.extract(doc[0])
        reason = extractor.last_error
        doc.close()
        assert blocks == []
        assert reason and "unavailable" in reason.lower()

    def test_ocr_disabled_ingest_records_warning_on_page(self, mixed_pdf):
        ingestor = PDFIngestor(enable_ocr=False)
        document = ingestor.ingest(str(mixed_pdf))
        ocr_pages = [p for p in document.pages
                     if p.classification.value == "OCR_REQUIRED"]
        assert ocr_pages, "fixture should contain a scan-like page"
        for p in ocr_pages:
            assert p.warnings, "OCR-required page must record explicit warning"
            assert not p.blocks or all(
                b.extraction_method.value != "OCR" for b in p.blocks
            )


class TestRealTesseractIfPresent:
    def test_tesseract_presence_reported(self):
        # Informational: must never fail the suite either way.
        ext = OCRExtractor()
        assert isinstance(ext.available, bool)
