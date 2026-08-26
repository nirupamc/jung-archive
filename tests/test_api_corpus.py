"""Corpus-facing API surface (post-M7): merged library + discovery report.

Read-only checks against the real repository state; assertions are
robust to ingestion progress (they assert honest statuses, not counts).
"""
import pytest
from fastapi.testclient import TestClient

from jung_archive.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


UNDISCOVERED_ID = "381d2da4b68e"  # The Undiscovered Self


def test_documents_lists_processed_and_unprocessed(client):
    r = client.get("/api/documents")
    assert r.status_code == 200
    docs = {d["document_id"]: d for d in r.json()}
    # processed entry keeps full fidelity
    d = docs[UNDISCOVERED_ID]
    assert d["title"] == "The Undiscovered Self"
    assert d["status"] in ("INDEXED", "CHUNKED", "PROCESSED")
    assert d["chunk_count"] > 0
    # discovered-but-unprocessed entries are present, never hidden,
    # and expose the honest pipeline status vocabulary
    statuses = {d_["status"] for d_ in docs.values()}
    allowed = {"DISCOVERED", "REVIEW", "EXCLUDED", "PROCESSED",
               "CHUNKED", "INDEXED", "ERROR"}
    assert statuses <= allowed
    by_title = {d_["title"]: d_ for d_ in docs.values() if d_.get("title")}
    aion = [t for t in by_title if t.startswith("Aion")]
    if aion:
        assert by_title[aion[0]]["status"] == "EXCLUDED"


def test_corpus_report_endpoint(client):
    r = client.get("/api/corpus")
    assert r.status_code == 200
    body = r.json()
    rep = body["report"]
    assert rep["discovered_total"] >= 16
    assert rep["pages_total"] > 8000
    for status in ("DISCOVERED", "REVIEW", "EXCLUDED", "PROCESSED",
                   "CHUNKED", "INDEXED", "ERROR"):
        assert status in rep["by_status"]
    sections = {d["section"] for d in body["documents"]}
    assert "PRIMARY" in sections
    assert "SECONDARY" in sections


def test_corpus_stats_endpoint(client):
    r = client.get("/api/corpus/stats")
    assert r.status_code == 200
    stats = r.json()
    assert stats["discovered_total"] >= 16
    assert set(stats["by_section"]) == {"PRIMARY", "SECONDARY"}


def test_document_detail_still_served_for_processed_doc(client):
    r = client.get(f"/api/documents/{UNDISCOVERED_ID}")
    assert r.status_code == 200
    assert r.json()["status"] in ("INDEXED", "CHUNKED", "PROCESSED")
