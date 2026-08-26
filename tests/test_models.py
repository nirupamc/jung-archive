import json

import pytest

from jung_archive.models.document import (
    Block,
    BoundingBox,
    Document,
    ExtractionMethod,
    Page,
    PageClassification,
    SourceType,
)


def make_block(**overrides):
    defaults = dict(
        block_id="p0000-b000",
        block_type="PARAGRAPH",
        text="Some text.",
        bbox={"x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 60.0},
        reading_order=1,
        extraction_method="NATIVE",
        confidence=0.9,
    )
    defaults.update(overrides)
    return Block(**defaults)


class TestBoundingBox:
    def test_valid_bbox_dimensions(self):
        bb = BoundingBox(x0=10, y0=20, x1=110, y1=70)
        assert bb.width() == 100
        assert bb.height() == 50
        assert bb.area() == 5000

    def test_rejects_negative_coordinates(self):
        with pytest.raises(Exception):
            BoundingBox(x0=-5, y0=0, x1=10, y1=10)

    def test_rejects_inverted_bbox(self):
        with pytest.raises(Exception):
            BoundingBox(x0=50, y0=0, x1=10, y1=10)

    def test_zero_area_bbox_rejected(self):
        # Degenerate bboxes carry no provenance value and are rejected.
        with pytest.raises(Exception):
            BoundingBox(x0=10, y0=10, x1=10, y1=10)


class TestSerialization:
    def test_block_serialization_round_trip(self):
        block = make_block()
        d = json.loads(block.to_json()) if hasattr(block, "to_json") else block.model_dump(mode="json")
        rebuilt = Block(**d)
        assert rebuilt == block

    def test_document_to_dict_structure(self):
        page = Page(
            page_number=1,
            width=595.0,
            height=842.0,
            classification=PageClassification.NATIVE,
            classification_confidence=0.95,
            layout="SINGLE_COLUMN",
            layout_confidence=0.85,
            blocks=[make_block()],
            warnings=[],
        )
        doc = Document(
            document_id="abc123",
            title="Test",
            source_type=SourceType.PRIMARY,
            source_path="primary/test.pdf",
            page_count=1,
            pages=[page],
        )
        d = doc.to_dict()
        assert d["document_id"] == "abc123"
        assert d["source_type"] == "PRIMARY"
        assert len(d["pages"]) == 1
        assert d["pages"][0]["blocks"][0]["text"] == "Some text."
        # Full JSON round trip
        doc2 = Document(**json.loads(doc.to_json()))
        assert doc2 == doc

    def test_confidence_bounds_enforced(self):
        with pytest.raises(Exception):
            make_block(confidence=1.5)
        with pytest.raises(Exception):
            make_block(confidence=-0.1)
        with pytest.raises(Exception):
            Page(
                page_number=1,
                width=595.0,
                height=842.0,
                classification=PageClassification.NATIVE,
                classification_confidence=2.0,
                layout="SINGLE_COLUMN",
                layout_confidence=0.9,
            )

    def test_extraction_method_preserved(self):
        b = make_block(extraction_method=ExtractionMethod.OCR)
        assert b.extraction_method == ExtractionMethod.OCR


class TestSourceType:
    def test_primary_and_secondary_exist(self):
        assert SourceType.PRIMARY.value == "PRIMARY"
        assert SourceType.SECONDARY.value == "SECONDARY"

    def test_document_accepts_both(self):
        for st in SourceType:
            doc = Document(
                document_id="x",
                title="t",
                source_type=st,
                source_path=f"{st.value.lower()}/f.pdf",
                page_count=0,
                pages=[],
            )
            assert doc.source_type == st
