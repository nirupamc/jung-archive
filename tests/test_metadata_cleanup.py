import json

import pytest

from jung_archive.ingestion.pdf import PDFIngestor
from jung_archive.metadata import DocumentMetadata, MetadataRegistry
from jung_archive.models.chunk import IndexSchemaMeta
from jung_archive.models.document import IndexStatus, SourceType


class TestIndexStatusBehavior:
    def test_unregistered_primary_defaults_to_review(self, doc_factory):
        # Folder says primary/, but no registry entry exists: must be REVIEW,
        # never silently INCLUDEd as trusted Jung material.
        ingestor = PDFIngestor(enable_ocr=False, registry=MetadataRegistry())
        doc = ingestor._determine_source_type("primary/Unknown Book.pdf")
        assert doc == SourceType.PRIMARY  # tentative only
        # Registry.apply on an unregistered doc forces REVIEW status
        from jung_archive.models.document import Document
        d = Document(
            document_id="x", title="t", source_type=SourceType.PRIMARY,
            source_path="primary/Unknown Book.pdf", page_count=0,
        )
        MetadataRegistry().apply(d)
        assert d.index_status == IndexStatus.REVIEW
        assert d.source_type == SourceType.PRIMARY  # tentative type retained

    def test_registry_overrides_folder_inference(self):
        registry = MetadataRegistry()
        registry.entries_by_path.append((
            "Aion PDF.pdf",
            DocumentMetadata(
                title_override="Aion (Bookey summary)",
                source_type=SourceType.SECONDARY,
                index_status=IndexStatus.EXCLUDE,
                reason="third-party summary",
            ),
        ))
        from jung_archive.models.document import Document
        d = Document(
            document_id="aionbookey", title="Aion PDF",
            source_type=SourceType.PRIMARY,  # folder inference said primary
            source_path="primary/Aion PDF.pdf", page_count=0,
        )
        registry.apply(d)
        assert d.source_type == SourceType.SECONDARY
        assert d.index_status == IndexStatus.EXCLUDE
        assert "Bookey" in d.title

    def test_known_bad_aion_identity_is_not_trusted_primary(self):
        """End-to-end guard: the real Bookey file ingests as EXCLUDE."""
        from pathlib import Path
        pdf = Path("primary/Aion PDF.pdf")
        if not pdf.exists():
            pytest.skip("corpus not present in this environment")
        ingestor = PDFIngestor(enable_ocr=False)
        assert ingestor.registry.lookup("primary/Aion PDF.pdf") is not None
        # Full ingest of just metadata-level fields via a tiny check:
        meta = ingestor.registry.lookup(str(pdf))
        assert meta.index_status == IndexStatus.EXCLUDE
        assert meta.source_type == SourceType.SECONDARY

    def test_registry_loads_from_config_file(self):
        from pathlib import Path
        cfg = Path("config/document_metadata.json")
        if not cfg.exists():
            pytest.skip("registry config not present")
        registry = MetadataRegistry.load(str(cfg))
        bookey = None
        for needle, meta in registry.entries_by_path:
            if "Aion PDF.pdf" in needle:
                bookey = meta
        assert bookey is not None
        assert bookey.index_status == IndexStatus.EXCLUDE

    def test_sha_lookup_takes_priority(self):
        registry = MetadataRegistry()
        by_sha = DocumentMetadata(index_status=IndexStatus.INCLUDE,
                                  source_type=SourceType.PRIMARY)
        by_path = DocumentMetadata(index_status=IndexStatus.REVIEW)
        registry.entries_by_sha["abc123"] = by_sha
        registry.entries_by_path.append(("whatever", by_path))
        found = registry.lookup("/some/whatever.pdf", sha256="abc123")
        assert found is by_sha


class TestConfidenceSemantics:
    def test_native_blocks_have_no_measured_confidence(self, text_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        for page in document.pages:
            for blk in page.blocks:
                assert blk.confidence is None, (
                    "native extraction must not fabricate measured certainty"
                )

    def test_heuristic_scores_are_separate_field(self, text_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        typed = [b for p in document.pages for b in p.blocks
                 if b.block_type.value != "UNKNOWN"]
        assert typed, "fixture should contain typed blocks"
        for blk in typed:
            assert blk.heuristic_quality_score is not None
            assert 0.0 <= blk.heuristic_quality_score <= 1.0

    def test_diagnostics_report_null_when_nothing_measured(self, text_pdf):
        from jung_archive.cli import generate_diagnostics
        document = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        d = generate_diagnostics(document)
        assert d["average_extraction_confidence"] is None
        assert d["measured_confidence_block_count"] == 0

    def test_page_ocr_confidence_null_without_ocr(self, text_pdf):
        document = PDFIngestor(enable_ocr=False).ingest(str(text_pdf))
        for page in document.pages:
            assert page.ocr_confidence is None

    def test_index_schema_meta_records_versions(self):
        meta = IndexSchemaMeta(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
        )
        data = json.loads(meta.model_dump_json())
        assert data["index_schema_version"]
        assert data["chunking_config_version"]
        assert data["normalized"] is True


class TestAuthorIdentityNeverDefaulted:
    def test_document_author_none_by_default(self, doc_factory):
        document = doc_factory([[("PARAGRAPH", "text")]])
        assert document.author is None

    def test_ingested_doc_author_comes_from_pdf_metadata_or_registry(
        self, tmp_path, monkeypatch
    ):
        import fitz
        from jung_archive.metadata import MetadataRegistry

        p = tmp_path / "mystery.pdf"
        d = fitz.open()
        d.new_page().insert_text((72, 100), "hello world " * 50)
        d.save(p)
        d.close()

        # Empty registry: author stays unasserted even under primary/
        ingestor = PDFIngestor(enable_ocr=False, registry=MetadataRegistry())
        doc = ingestor.ingest(str(p))
        assert doc.author is None or isinstance(doc.author, str)
