import sys
from pathlib import Path

import fitz
import pytest

# Make src/ importable without installing the package
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def build_synthetic_document(
    pages_spec,
    document_id="testdoc000",
    title="Test Document",
    source_type="PRIMARY",
    page_width=595.0,
    page_height=842.0,
):
    """Build an in-memory Document from a spec.

    pages_spec: list of pages; each page is a list of
        (block_type_str, text) tuples.
    Blocks get deterministic IDs p{page:04d}-b{idx:03d} and plausible
    geometry stacked down each page.
    """
    from jung_archive.models.document import (
        Block,
        BoundingBox,
        Document,
        Page,
    )

    pages = []
    for page_no, blocks_spec in enumerate(pages_spec, start=1):
        blocks = []
        y = 40.0
        for idx, (btype, text) in enumerate(blocks_spec):
            n_lines = max(1, text.count("\n") + 1)
            height = 14.0 * n_lines + 6
            block = Block(
                block_id=f"p{page_no:04d}-b{idx:03d}",
                block_type=btype,
                text=text,
                bbox={
                    "x0": 50.0,
                    "y0": y,
                    "x1": 545.0,
                    "y1": min(y + height, page_height - 10),
                },
                reading_order=idx + 1,
                extraction_method="NATIVE",
                confidence=None,
            )
            blocks.append(block)
            y += height + 12
        pages.append(Page(
            page_number=page_no,
            width=page_width,
            height=page_height,
            classification="NATIVE",
            classification_confidence=0.95,
            layout="SINGLE_COLUMN",
            layout_confidence=0.85,
            blocks=blocks,
        ))
    return Document(
        document_id=document_id,
        title=title,
        author=None,
        source_type=source_type,
        source_path=f"primary/{title}.pdf",
        page_count=len(pages),
        pages=pages,
    )


@pytest.fixture
def doc_factory():
    return build_synthetic_document


def make_text_pdf(path: Path, n_pages: int = 1) -> Path:
    """Deterministic single-column text-only PDF."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Page {i + 1} Heading", fontsize=20)
        body = (
            "This is a body paragraph with enough words to wrap across "
            "several rendered lines so that it resembles ordinary prose. "
        ) * 6
        page.insert_textbox(fitz.Rect(72, 140, 520, 700), body, fontsize=12)
    doc.save(path)
    doc.close()
    return path


def make_empty_pdf(path: Path) -> Path:
    """PDF with one truly blank page."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(path)
    doc.close()
    return path


def make_scanned_pdf(path: Path) -> Path:
    """PDF with one image-only page simulating a scan."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 600))
    pix.clear_with(200)  # light gray "paper"
    page.insert_image(fitz.Rect(50, 50, 545, 792), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def make_multipage_mixed_pdf(path: Path) -> Path:
    """Three pages: text / empty / scanned-like."""
    doc = fitz.open()
    p = doc.new_page(width=595, height=842)
    p.insert_text((72, 100), "Title", fontsize=24)
    p.insert_textbox(
        fitz.Rect(72, 140, 520, 700),
        "Body text for the first page. " * 30,
        fontsize=12,
    )
    doc.new_page(width=595, height=842)  # blank
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 400))
    pix.clear_with(180)
    page.insert_image(fitz.Rect(60, 60, 500, 780), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def text_pdf(tmp_path) -> Path:
    return make_text_pdf(tmp_path / "text.pdf")


@pytest.fixture
def empty_pdf(tmp_path) -> Path:
    return make_empty_pdf(tmp_path / "empty.pdf")


@pytest.fixture
def scanned_pdf(tmp_path) -> Path:
    return make_scanned_pdf(tmp_path / "scanned.pdf")


@pytest.fixture
def mixed_pdf(tmp_path) -> Path:
    return make_multipage_mixed_pdf(tmp_path / "mixed.pdf")
