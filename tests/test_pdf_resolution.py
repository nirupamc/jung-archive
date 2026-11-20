"""PDF path resolution and route parity tests.

These tests exercise the centralized resolver used by both
``document_summary()`` (for ``has_pdf``) and the
``/api/documents/<id>/pdf`` route, so they write real chunk artifacts under a
temporary data dir and point the API module at those fixtures.

Note: ``jung_archive.api.app`` is re-exported by ``jung_archive.api`` as the
FastAPI instance, so ``import jung_archive.api.app as pkg`` would bind the app,
not the module. The module object is fetched from ``sys.modules`` instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _api_module():
    import jung_archive.api.app  # noqa: F401  (ensure module is imported)

    return sys.modules["jung_archive.api.app"]


def _artifact(
    source_path: str,
    *,
    title: str = "Doc",
    source_type: str = "PRIMARY",
    chunk_count: int = 1,
) -> dict:
    return {
        "document": {
            "source_path": source_path,
            "title": title,
            "source_type": source_type,
        },
        "chunk_count": chunk_count,
    }


def _mount_fixtures(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Create a PDF root + data dir and point the API module at them."""
    fixture_root = tmp_path / "pdfroot"
    (fixture_root / "primary").mkdir(parents=True)
    (fixture_root / "secondary").mkdir(parents=True)

    data_root = tmp_path / "data"
    (data_root / "chunks").mkdir(parents=True)
    (data_root / "processed").mkdir(parents=True)

    monkeypatch.setenv("JUNG_ARCHIVE_PDF_DIR", str(fixture_root))
    monkeypatch.setenv("JUNG_ARCHIVE_DATA_DIR", str(data_root))

    pkg = _api_module()
    monkeypatch.setattr(pkg, "PDF_DIR", fixture_root, raising=False)
    monkeypatch.setattr(pkg, "CHUNKS_DIR", data_root / "chunks", raising=False)
    monkeypatch.setattr(pkg, "PROCESSED_DIR", data_root / "processed", raising=False)

    return fixture_root, data_root


def _clear_artifact_caches() -> None:
    pkg = _api_module()
    pkg.load_chunk_artifact_by_doc.cache_clear()
    pkg._load_processed.cache_clear()


def _write_chunk_artifact(data_root: Path, document_id: str, payload: dict) -> None:
    path = data_root / "chunks" / f"{document_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _clear_artifact_caches()


# ---------------------------------------------------------------------------
# Resolver behavior


def test_resolver_keeps_relative_primary_and_secondary_paths(tmp_path, monkeypatch):
    from jung_archive.config import resolve_pdf_path

    root, _ = _mount_fixtures(tmp_path, monkeypatch)

    (root / "primary" / "good.pdf").write_bytes(b"%PDF-1.4\n")
    (root / "secondary" / "other.pdf").write_bytes(b"%PDF-1.4\n")

    assert resolve_pdf_path("primary/good.pdf") is not None
    assert resolve_pdf_path("secondary/other.pdf") is not None
    assert resolve_pdf_path("primary/missing.pdf") is None


def test_resolver_cannot_escape_configured_root(tmp_path, monkeypatch):
    from jung_archive.config import resolve_pdf_path

    root, _ = _mount_fixtures(tmp_path, monkeypatch)

    outside_pdf = tmp_path / "outside.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4\n")
    (root / "primary" / "inside.pdf").write_bytes(b"%PDF-1.4\n")

    # A traversal that climbs out of the configured root is rejected.
    assert resolve_pdf_path("primary/../../outside.pdf") is None
    # A file physically outside the root is never served.
    resolved = resolve_pdf_path(str(outside_pdf))
    assert resolved is None


def test_resolver_handles_stored_windows_absolute_paths(tmp_path, monkeypatch):
    """Stored Windows absolute paths map onto the configured PDF root."""
    from jung_archive.config import resolve_pdf_path

    root, _ = _mount_fixtures(tmp_path, monkeypatch)

    expected = root / "primary" / "windows.pdf"
    expected.write_bytes(b"%PDF-1.4\n")

    stored = r"C:\Users\someone\jung-archive\primary\windows.pdf"
    resolved = resolve_pdf_path(stored)

    assert resolved == expected.resolve()
    assert resolved.is_file()


def test_resolver_rejects_unsafe_paths(tmp_path, monkeypatch):
    from jung_archive.config import resolve_pdf_path

    root, _ = _mount_fixtures(tmp_path, monkeypatch)
    (root / "primary" / "ok.pdf").write_bytes(b"%PDF-1.4\n")

    assert resolve_pdf_path(None) is None
    assert resolve_pdf_path("") is None
    assert resolve_pdf_path("notes.txt") is None
    assert resolve_pdf_path("config/document_metadata.json") is None
    assert resolve_pdf_path("primary/../../../etc/passwd") is None


def test_resolver_requires_existing_file(tmp_path, monkeypatch):
    from jung_archive.config import resolve_pdf_path

    root, _ = _mount_fixtures(tmp_path, monkeypatch)
    (root / "primary" / "dir_named.pdf").mkdir()

    assert resolve_pdf_path("primary/dir_named.pdf") is None


# ---------------------------------------------------------------------------
# document_summary / PDF route parity


def test_api_has_pdf_uses_resolver(tmp_path, monkeypatch):
    """document_summary() has_pdf must reflect actual PDF availability."""
    pkg = _api_module()
    root, data_root = _mount_fixtures(tmp_path, monkeypatch)

    (root / "primary" / "good.pdf").write_bytes(b"%PDF-1.4\n")

    _write_chunk_artifact(
        data_root, "good-doc", _artifact("primary/good.pdf", title="Good Pdf Doc")
    )
    good = pkg.document_summary("good-doc")
    assert good.has_pdf is True, good.model_dump()

    _write_chunk_artifact(
        data_root,
        "missing-doc",
        _artifact("primary/missing.pdf", title="Missing Pdf Doc"),
    )
    missing = pkg.document_summary("missing-doc")
    assert missing.has_pdf is False, missing.model_dump()


def test_pdf_route_returns_404_when_missing(tmp_path, monkeypatch):
    """The PDF route must return 404 when the PDF is unavailable."""
    pkg = _api_module()
    _, data_root = _mount_fixtures(tmp_path, monkeypatch)

    _write_chunk_artifact(
        data_root,
        "missing-doc",
        _artifact("primary/missing.pdf", title="Missing Pdf Doc"),
    )

    client = TestClient(pkg.app)
    r = client.get("/api/documents/missing-doc/pdf")
    assert r.status_code == 404, (r.status_code, r.json())
    assert "source PDF unavailable" in r.json()["detail"], r.json()


def test_pdf_route_returns_pdf_when_present(tmp_path, monkeypatch):
    """The PDF route must serve an existing PDF with application/pdf."""
    pkg = _api_module()
    root, data_root = _mount_fixtures(tmp_path, monkeypatch)

    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj trailer<</Root 1 0 R>>%%EOF"
    (root / "primary" / "present.pdf").write_bytes(pdf_bytes)

    _write_chunk_artifact(
        data_root,
        "present-doc",
        _artifact("primary/present.pdf", title="Present Pdf Doc"),
    )

    client = TestClient(pkg.app)
    r = client.get("/api/documents/present-doc/pdf")
    assert r.status_code == 200, (r.status_code, r.json())
    assert r.headers["content-type"].startswith("application/pdf"), r.headers
    assert r.content == pdf_bytes


@pytest.mark.parametrize("source_path", ["primary/both.pdf", "primary\\both.pdf"])
def test_pdf_route_uses_same_resolver_as_summary(
    tmp_path, monkeypatch, source_path
):
    """document_summary and document_pdf must agree via one resolver."""
    pkg = _api_module()
    root, data_root = _mount_fixtures(tmp_path, monkeypatch)

    (root / "primary" / "both.pdf").write_bytes(b"%PDF-1.4\n%%EOF")

    _write_chunk_artifact(
        data_root, "both-doc", _artifact(source_path, title="Both Doc")
    )

    summary = pkg.document_summary("both-doc")
    assert summary.has_pdf is True, summary.model_dump()

    client = TestClient(pkg.app)
    r = client.get("/api/documents/both-doc/pdf")
    assert r.status_code == 200, (r.status_code, r.json())
    assert r.headers["content-type"].startswith("application/pdf"), r.headers
