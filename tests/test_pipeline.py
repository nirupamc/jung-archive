import json

import pytest

from jung_archive.cli import generate_diagnostics
from jung_archive.ingestion.pdf import PDFIngestor
from jung_archive.models.document import Document


class TestDiagnostics:
    def test_aggregation_matches_document(self, mixed_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(mixed_pdf))
        d = generate_diagnostics(document)

        assert d["document_id"] == document.document_id
        assert d["page_count"] == 3
        assert sum(d["classification_counts"].values()) == 3
        assert sum(d["layout_counts"].values()) == 3
        assert sum(d["block_counts"].values()) == sum(len(p.blocks) for p in document.pages)
        # Native-only fixtures have no MEASURED extraction confidences,
        # so the average must be null rather than a fabricated prior.
        avg = d["average_extraction_confidence"]
        assert avg is None or 0.0 <= avg <= 1.0
        assert d["measured_confidence_block_count"] == (
            sum(1 for b in (blk for p in document.pages for blk in p.blocks)
                if b.confidence is not None)
        )

        # Empty middle page must be counted as EMPTY
        assert d["classification_counts"].get("EMPTY", 0) >= 1

    def test_diagnostics_not_hardcoded(self, text_pdf):
        doc_a = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        d = generate_diagnostics(doc_a)
        # A generated 1-page text doc must not look like the defaults
        assert d["page_count"] == 1
        assert d["classification_counts"].get("NATIVE") == 1

    def test_native_only_document_reports_no_measured_confidence(self, text_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        d = generate_diagnostics(document)
        # Native extraction provides no measured certainty: must be null.
        assert d["average_extraction_confidence"] is None
        assert d["measured_confidence_block_count"] == 0

    def test_measured_confidences_are_averaged_when_present(self, text_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        # Simulate measured OCR confidences on the blocks.
        for i, page in enumerate(document.pages):
            for j, blk in enumerate(page.blocks):
                blk.confidence = 0.80 + 0.01 * ((i + j) % 10)
        d = generate_diagnostics(document)
        confs = [b.confidence for p in document.pages for b in p.blocks
                 if b.confidence is not None]
        expected = round(sum(confs) / len(confs), 4)
        assert d["average_extraction_confidence"] == expected
        assert d["measured_confidence_block_count"] == len(confs)


class TestFailureIsolation:
    def test_page_failure_does_not_destroy_document(self, mixed_pdf, monkeypatch):
        ingestor = PDFIngestor(enable_ocr=False)

        original_extract = ingestor.native_extractor.extract
        calls = {"n": 0}

        def flaky_extract(page):
            calls["n"] += 1
            if page.number == 0:  # first page fails
                raise RuntimeError("simulated extraction crash")
            return original_extract(page)

        monkeypatch.setattr(ingestor.native_extractor, "extract", flaky_extract)
        document = ingestor.ingest(str(mixed_pdf))

        assert document.page_count == 3, "all pages must be present"
        failed = [p for p in document.pages if p.classification.value == "FAILED"]
        assert len(failed) == 1
        assert failed[0].page_number == 1
        assert any("crash" in w.lower() or "failed" in w.lower() for w in failed[0].warnings)
        # Later pages still processed normally
        later = [p for p in document.pages if p.page_number > 1]
        assert all(p.classification.value != "FAILED" for p in later)

    def test_corrupt_file_raises_cleanly(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4 this is not a real pdf")
        with pytest.raises(Exception):
            PDFIngestor(enable_ocr=False).ingest(str(bad))


class TestJSONIntegrity:
    def test_round_trip(self, mixed_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(mixed_pdf))
        parsed = json.loads(document.to_json())
        rebuilt = Document(**parsed)
        assert rebuilt == document

    def test_schema_keys_present(self, text_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        parsed = json.loads(document.to_json())
        for key in ("document_id", "title", "author", "source_type",
                    "source_path", "page_count", "pages"):
            assert key in parsed
        page = parsed["pages"][0]
        for key in ("page_number", "width", "height", "classification",
                    "classification_confidence", "layout", "layout_confidence",
                    "blocks", "warnings"):
            assert key in page
        blk = page["blocks"][0]
        for key in ("block_id", "block_type", "text", "bbox", "reading_order",
                    "extraction_method", "confidence"):
            assert key in blk
        assert set(blk["bbox"].keys()) == {"x0", "y0", "x1", "y1"}
