"""Corpus discovery + batch-ingestion safety tests (post-M7).

Uses an isolated tmp repo with synthetic PDFs and a stub embedding
provider so no real models or production artifacts are touched.
"""
import json
from pathlib import Path

import fitz
import pytest

from jung_archive.corpus import PipelineStatus, corpus_report, \
    derive_status, discover_corpus


def make_pdf(path: Path, title: str = "", author: str = "",
             pages: int = 2) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{title} page {i + 1} lorem ipsum")
    doc.set_metadata({"title": title, "author": author})
    doc.save(str(path))
    doc.close()


def write_registry(root: Path, entries: list) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    with open(cfg / "document_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"documents": entries}, f)


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def corpus_root(tmp_path) -> Path:
    root = tmp_path
    (root / "primary").mkdir()
    (root / "secondary").mkdir()
    make_pdf(root / "primary" / "approved.pdf", title="Approved Work",
             author="C. G. Jung")
    make_pdf(root / "primary" / "bogus-summary.pdf", title="Bogus Summary")
    make_pdf(root / "primary" / "mystery.pdf", title="Mystery Text")
    make_pdf(root / "secondary" / "scholarship.pdf", title="About Jung",
             author="A. Scholar")
    # registry: one INCLUDE keyed by sha, one EXCLUDE, one REVIEW;
    # mystery.pdf stays unregistered on purpose.
    write_registry(root, [
        {
            "path_contains": "approved.pdf",
            "title": "Approved Work",
            "author": "C. G. Jung",
            "source_type": "PRIMARY",
            "index_status": "INCLUDE",
            "reason": "verified authentic",
        },
        {
            "path_contains": "bogus-summary.pdf",
            "title": "Bogus Summary",
            "source_type": "SECONDARY",
            "index_status": "EXCLUDE",
            "reason": "third-party summary must never enter the index",
        },
        {
            "path_contains": "scholarship.pdf",
            "title": "About Jung",
            "source_type": "SECONDARY",
            "index_status": "REVIEW",
            "reason": "awaiting curation decision",
        },
    ])
    return root


class StubProvider:
    """Deterministic tiny embedding provider (no model download)."""

    model_name = "stub-embedder"
    dimension = 8
    normalized = True

    def embed(self, texts):
        import numpy as np

        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.encode("utf-8")):
                out[i, j % self.dimension] += (ch % 17) - 8
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


# ----------------------------------------------------------------------
# Status derivation matrix

def test_derive_status_matrix():
    d = derive_status
    assert d(readable=False, index_status_value="INCLUDE", registered=True,
             has_processed=False, has_chunks=False,
             index_state=None) == "ERROR"
    assert d(readable=True, index_status_value="EXCLUDE", registered=True,
             has_processed=True, has_chunks=False,
             index_state=None) == "EXCLUDED"
    assert d(readable=True, index_status_value="REVIEW", registered=True,
             has_processed=False, has_chunks=False,
             index_state=None) == "REVIEW"
    assert d(readable=True, index_status_value="UNKNOWN", registered=False,
             has_processed=False, has_chunks=False,
             index_state=None) == "REVIEW"
    assert d(readable=True, index_status_value="INCLUDE", registered=True,
             has_processed=False, has_chunks=False,
             index_state=None) == "DISCOVERED"
    assert d(readable=True, index_status_value="INCLUDE", registered=True,
             has_processed=True, has_chunks=False,
             index_state=None) == "PROCESSED"
    assert d(readable=True, index_status_value="INCLUDE", registered=True,
             has_processed=True, has_chunks=True,
             index_state={"chunk_count": 5}) == "INDEXED"
    assert d(readable=True, index_status_value="INCLUDE", registered=True,
             has_processed=True, has_chunks=True,
             index_state=None) == "CHUNKED"


# ----------------------------------------------------------------------
# Discovery over a synthetic corpus

def test_discovery_lists_every_pdf_with_honest_statuses(corpus_root):
    docs = discover_corpus(repo_root=corpus_root)
    by_path = {d.path: d for d in docs}
    assert len(docs) == 4

    approved = by_path["primary/approved.pdf"]
    assert approved.status == PipelineStatus.DISCOVERED.value
    assert approved.index_status == "INCLUDE"
    assert approved.section == "PRIMARY"
    assert approved.registered and approved.title == "Approved Work"

    bogus = by_path["primary/bogus-summary.pdf"]
    assert bogus.status == PipelineStatus.EXCLUDED.value

    scholarship = by_path["secondary/scholarship.pdf"]
    assert scholarship.status == PipelineStatus.REVIEW.value
    assert scholarship.section == "SECONDARY"

    mystery = by_path["primary/mystery.pdf"]
    assert not mystery.registered
    # folder location alone never grants trust: unregistered => REVIEW
    assert mystery.status == PipelineStatus.REVIEW.value

    rep = corpus_report(docs)
    assert rep["discovered_total"] == 4
    assert rep["by_section"]["PRIMARY"] == 3
    assert rep["by_section"]["SECONDARY"] == 1
    assert rep["excluded"] == 1
    assert rep["review"] == 2


def test_discovery_survives_unreadable_pdf(corpus_root):
    (corpus_root / "primary" / "corrupt.pdf").write_bytes(b"not a pdf")
    docs = discover_corpus(repo_root=corpus_root)
    corrupt = [d for d in docs if d.file_name == "corrupt.pdf"][0]
    assert corrupt.status == PipelineStatus.ERROR.value
    assert corrupt.error
    # ...and it is still listed, never hidden
    assert len(docs) == 5


# ----------------------------------------------------------------------
# Batch ingestion safety + idempotency

def _ingest(root: Path, **kw):
    from jung_archive.ingestion.batch import ingest_batch

    return ingest_batch(
        repo_root=root,
        provider=StubProvider(),
        progress=lambda m: None,
        **kw,
    )


def test_batch_ingests_only_approved_documents(corpus_root):
    report = _ingest(corpus_root)

    # only the single INCLUDE document was processed
    assert [p["path"] for p in report["processed_ok"]] == \
        ["primary/approved.pdf"]
    held = {h["path"]: h for h in report["held_back"]}
    assert "primary/bogus-summary.pdf" in held
    assert held["primary/bogus-summary.pdf"]["registry_decision"] == "EXCLUDE"
    assert held["secondary/scholarship.pdf"]["registry_decision"] == "REVIEW"
    # the unregistered file is held back as REVIEW, never silently trusted
    assert held["primary/mystery.pdf"]["registry_decision"] == "UNKNOWN"

    # EXCLUDE document got NO artifacts at all
    chunks_dir = corpus_root / "data" / "chunks"
    from jung_archive.corpus import generate_document_id

    bogus_id = generate_document_id(str(corpus_root / "primary" /
                                        "bogus-summary.pdf"))
    assert not (chunks_dir / f"{bogus_id}.json").exists()

    # post-state discovery reflects pipeline progress
    docs = {d.path: d for d in discover_corpus(repo_root=corpus_root)}
    assert docs["primary/approved.pdf"].status == "INDEXED"
    assert docs["primary/bogus-summary.pdf"].status == "EXCLUDED"
    assert docs["secondary/scholarship.pdf"].status == "REVIEW"
    assert docs["primary/mystery.pdf"].status == "REVIEW"


def test_batch_is_idempotent_and_reuses_artifacts(corpus_root):
    first = _ingest(corpus_root)
    assert len(first["processed_ok"]) == 1
    assert first["freshly_ingested"] == 1

    second = _ingest(corpus_root)
    assert len(second["processed_ok"]) == 1
    # canonical artifact reused; embeddings not regenerated
    assert second["artifacts_reused"] == 1
    assert second["freshly_ingested"] == 0
    assert second["totals"]["vectors_indexed"] == 0
    assert second["totals"]["index_unchanged"] == 1


def test_batch_reingests_when_source_changes(corpus_root):
    _ingest(corpus_root)
    # mutate the approved source -> sha changes
    pdf_path = corpus_root / "primary" / "approved.pdf"
    make_pdf(pdf_path, title="Approved Work", author="C. G. Jung", pages=3)

    report = _ingest(corpus_root)
    entry = report["processed_ok"][0]
    assert entry["vectors_indexed"] > 0  # re-embedded, not skipped


def test_batch_registry_flip_to_exclude_blocks_reprocessing(corpus_root):
    _ingest(corpus_root)
    # curator flips the decision after first ingestion
    write_registry(corpus_root, [
        {
            "path_contains": "approved.pdf",
            "title": "Approved Work",
            "source_type": "PRIMARY",
            "index_status": "EXCLUDE",
            "reason": "decision reversed",
        },
    ])
    report = _ingest(corpus_root)
    assert report["candidates"] == 0
    assert report["processed_ok"] == []
