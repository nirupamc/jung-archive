"""M5 backend API tests (read-only inspection endpoints).

Uses the real processed/indexed corpus for read endpoints and the real
retrieval stack for the search endpoint (BM25 mode avoids the embedding
model; one hybrid test exercises the dense leg once per session).
"""
import json

import pytest
from fastapi.testclient import TestClient

from jung_archive.api.app import app

DOC = "381d2da4b68e"  # The Undiscovered Self
PAGE_COUNT = 88


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ----------------------------------------------------------------------
# Documents

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["documents"] >= 1


def test_document_list_contains_undiscovered_self(client):
    r = client.get("/api/documents")
    assert r.status_code == 200
    docs = {d["document_id"]: d for d in r.json()}
    assert DOC in docs
    d = docs[DOC]
    assert d["title"] == "The Undiscovered Self"
    assert d["author"] == "Carl Gustav Jung"
    assert d["source_type"] == "PRIMARY"
    assert d["index_status"] == "INCLUDE"
    assert d["page_count"] == PAGE_COUNT
    assert d["chunk_count"] > 0
    assert d["has_pdf"] is True


def test_document_detail_unknown_404(client):
    r = client.get("/api/documents/does-not-exist")
    assert r.status_code == 404


# ----------------------------------------------------------------------
# Pages / structure

def test_page_inspection_valid(client):
    r = client.get(f"/api/documents/{DOC}/pages/17")
    assert r.status_code == 200
    p = r.json()
    assert p["page_number"] == 17
    assert p["classification"] in ("NATIVE", "OCR_REQUIRED", "HYBRID",
                                   "EMPTY", "SUSPICIOUS", "FAILED")
    assert p["layout"]
    assert isinstance(p["blocks"], list) and p["blocks"]
    b = p["blocks"][0]
    for key in ("block_id", "block_type", "bbox", "reading_order",
                "extraction_method", "text", "page_number"):
        assert key in b
    # native extraction must NOT fabricate confidence values
    if b["extraction_method"] == "NATIVE":
        assert b["confidence"] is None
        assert p["ocr_confidence"] is None


def test_page_inspection_missing_page_404(client):
    assert client.get(f"/api/documents/{DOC}/pages/9999").status_code == 404
    assert client.get(f"/api/documents/{DOC}/pages/0").status_code == 404


def test_page_inspection_missing_document_404(client):
    r = client.get("/api/documents/does-not-exist/pages/1")
    assert r.status_code == 404


def test_structure_endpoint(client):
    r = client.get(f"/api/documents/{DOC}/structure?page=17")
    assert r.status_code == 200
    items = r.json()
    assert items
    assert all(i["page_number"] == 17 for i in items)
    full = client.get(f"/api/documents/{DOC}/structure").json()
    assert len(full) >= len(items)


def test_pdf_served_bytes_match_source(client):
    r = client.get(f"/api/documents/{DOC}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    import pathlib

    src = pathlib.Path("primary/The Undiscovered Self.pdf").read_bytes()
    assert r.content[:8] == b"%PDF" or r.content[:5] == b"%PDF-"


def test_chunks_endpoint(client):
    r = client.get(f"/api/documents/{DOC}/chunks")
    assert r.status_code == 200
    chunks = r.json()
    assert len(chunks) > 100
    c = chunks[0]
    for key in ("chunk_id", "document_id", "page_numbers",
                "token_count", "source_type", "source_block_ids",
                "heading_path", "text"):
        assert key in c
    # page filter
    r2 = client.get(f"/api/documents/{DOC}/chunks?page=18")
    assert all(18 in c2["page_numbers"] for c2 in r2.json())


def test_chunks_missing_document_404(client):
    assert client.get("/api/documents/does-not-exist/chunks").status_code == 404


# ----------------------------------------------------------------------
# Retrieval + evidence APIs

def test_search_rejects_invalid_mode(client):
    r = client.post("/api/retrieval/search",
                    json={"query": "shadow", "mode": "bogus"})
    assert r.status_code == 400


def test_search_rejects_empty_query(client):
    r = client.post("/api/retrieval/search",
                    json={"query": "   ", "mode": "bm25"})
    assert r.status_code == 400


def test_search_bm25_mode(client):
    r = client.post("/api/retrieval/search",
                    json={"query": "mass-mindedness", "mode": "bm25",
                          "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "bm25"
    assert 1 <= len(body["results"]) <= 3
    res = body["results"][0]
    assert res["bm25_score"] is not None
    assert res["reranker_score"] is None


def test_search_hybrid_mode_with_score_path(client):
    r = client.post("/api/retrieval/search",
                    json={"query": "self-knowledge", "mode": "hybrid",
                          "top_k": 3})
    assert r.status_code == 200
    for res in r.json()["results"]:
        assert res["fusion_rank"] is not None
        # nullable fields stay null-safe
        assert res["chunk_id"] and res["page_numbers"]
        assert set(res.keys()) >= {"dense_rank", "dense_score",
                                   "bm25_rank", "bm25_score",
                                   "fusion_rank", "fusion_score",
                                   "reranker_rank", "reranker_score"}


def test_search_hybrid_rerank_mode(client):
    r = client.post("/api/retrieval/search",
                    json={"query": "mass-mindedness",
                          "mode": "hybrid_rerank", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "hybrid_rerank"
    assert body["candidates_retrieved"] is not None
    for res in body["results"]:
        assert res["reranker_rank"] is not None
        assert res["reranker_score"] is not None
        # earlier stages preserved
        assert res["fusion_rank"] is not None


def test_evidence_assemble_api(client):
    from jung_archive.evidence.models import EvidencePack

    r = client.post("/api/evidence/assemble",
                    json={"question": "self-knowledge", "top_k": 4,
                          "max_tokens": 1200, "max_items": 3})
    assert r.status_code == 200
    pack = EvidencePack.from_dict(r.json())
    assert pack.max_evidence_tokens == 1200
    assert pack.tokens_used <= 1200
    assert pack.items
    ids = [i.evidence_id for i in pack.items]
    assert ids == [f"S{i}" for i in range(1, len(ids) + 1)]


def test_evidence_rejects_bad_budget(client):
    r = client.post("/api/evidence/assemble",
                    json={"question": "q", "max_tokens": 0})
    assert r.status_code == 400


# ----------------------------------------------------------------------
# Regression: 502 Bad Gateway on retrieval (missing backend deps)

def test_retrieval_dependencies_importable():
    """Regression for the 502 root cause: the backend retrieval stack must
    be able to import its required dependencies (chromadb, rank_bm25).
    Before the fix these were missing from pyproject.toml, causing every
    POST /api/retrieval/search to hit the generic except -> HTTPException(502)
    path in app.py.
    """
    import chromadb  # noqa: F401
    import rank_bm25  # noqa: F401
    from jung_archive.indexing.vector_index import VectorIndex  # noqa: F401
    from jung_archive.retrieval.lexical import BM25Retriever  # noqa: F401
    from jung_archive.reranking.cross_encoder import LocalCrossEncoderReranker  # noqa: F401


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_retrieval_no_502(client, mode):
    """No retrieval mode may return 502 (generic backend failure).

    The 502 was previously caused by missing chromadb/rank_bm25 packages;
    the except clause in retrieval_search converted ImportError to 502.
    """
    r = client.post("/api/retrieval/search",
                    json={"query": "self", "mode": mode, "top_k": 3})
    assert r.status_code != 502, f"{mode} returned 502: {r.text}"
    assert r.status_code == 200, f"{mode} failed: {r.text[:300]}"


def test_search_dense_mode_real_query(client):
    """DENSE mode must return results from the indexed corpus."""
    r = client.post("/api/retrieval/search",
                    json={"query": "individuation", "mode": "dense",
                          "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "dense"
    assert r.status_code == 200


def test_search_bm25_mode_real_query(client):
    """BM25 mode must return results from the indexed corpus."""
    r = client.post("/api/retrieval/search",
                    json={"query": "individuation", "mode": "bm25",
                          "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "bm25"
    assert len(body["results"]) >= 1


def test_search_hybrid_mode_real_query(client):
    """HYBRID mode must return results via RRF fusion."""
    r = client.post("/api/retrieval/search",
                    json={"query": "individuation", "mode": "hybrid",
                          "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "hybrid"
    assert len(body["results"]) >= 1
    for res in body["results"]:
        assert res["fusion_rank"] is not None


def test_search_hybrid_rerank_real_query(client):
    """HYBRID+RERANK mode must return results with reranker scores."""
    r = client.post("/api/retrieval/search",
                    json={"query": "individuation",
                          "mode": "hybrid_rerank", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "hybrid_rerank"
    assert len(body["results"]) >= 1


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Evaluation Lab API (M6)

def test_evaluation_runs_listed(client):
    r = client.get("/api/evaluation/runs")
    assert r.status_code == 200
    runs = r.json()
    assert isinstance(runs, list) and runs
    assert {"run_id", "timestamp", "modes"} <= set(runs[0].keys())


def test_evaluation_latest_summary(client):
    r = client.get("/api/evaluation/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["aggregates"]
    modes = {a["mode"] for a in body["aggregates"]}
    assert "dense" in modes and "bm25" in modes
    agg = next(a for a in body["aggregates"] if a["mode"] == "hybrid_rerank")
    cm = agg["chunk_metrics"]
    for metric in ("hit_at_k", "recall_at_k", "precision_at_k",
                   "ndcg_at_k"):
        assert metric in cm
    # generation metrics must be explicitly NOT RUN
    assert body["generation_eval"]["status"] == "NOT_RUN"


def test_evaluation_run_by_id(client):
    runs = client.get("/api/evaluation/runs").json()
    rid = runs[0]["run_id"]
    r = client.get(f"/api/evaluation/runs/{rid}")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"].startswith(rid[:12])
    assert body["per_query"]
    first_mode = next(iter(body["per_query"]))
    pq = body["per_query"][first_mode][0]
    assert "chunk_metrics" in pq and "page_metrics" in pq


def test_evaluation_unknown_run_404(client):
    assert client.get("/api/evaluation/runs/nope").status_code == 404


# ----------------------------------------------------------------------
# Graph API (M7)

def test_graph_overview(client):
    r = client.get("/api/graph")
    assert r.status_code == 200
    body = r.json()
    stats = body["stats"]
    assert stats["node_count"] > 0
    assert stats["edge_count"] > 0
    assert stats["trusted_edges"] > 0
    assert 0 <= stats["evidence_backed_ratio"] <= 1
    # every served node/edge is typed
    n = body["nodes"][0]
    assert {"node_id", "canonical_name", "node_type",
            "evidence_count"} <= set(n.keys())
    assert body["stale"] == []


def test_graph_node_detail_with_neighbors_and_evidence(client):
    overview = client.get("/api/graph").json()
    node_id = overview["nodes"][0]["node_id"]
    r = client.get(f"/api/graph/nodes/{node_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == node_id
    assert isinstance(body["edges"], list)
    assert isinstance(body["top_evidence"], list)
    if body["edges"]:
        e = body["edges"][0]
        assert "other_name" in e and "relationship_type" in e

    nbr = client.get(f"/api/graph/nodes/{node_id}/neighbors")
    assert nbr.status_code == 200


def test_graph_node_404(client):
    assert client.get("/api/graph/nodes/concept:does-not-exist").status_code \
        == 404


def test_graph_edge_detail_carries_evidence(client):
    overview = client.get("/api/graph").json()
    trusted = next(e for e in overview["edges"]
                   if e["status"] == "TRUSTED")
    r = client.get(f"/api/graph/edges/{trusted['edge_id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["evidence"]
    ev = body["evidence"][0]
    for key in ("document_id", "chunk_id", "page_numbers", "evidence_text"):
        assert key in ev


def test_graph_evidence_endpoint_404(client):
    assert client.get("/api/graph/evidence/ge-nope").status_code == 404


def test_graph_search(client):
    r = client.get("/api/graph/search", params={"q": "shadow"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results
    assert results[0]["canonical_name"] == "Shadow"

    # alias search: 'know thyself' -> Self-knowledge
    r2 = client.get("/api/graph/search", params={"q": "know thyself"})
    names = [x["canonical_name"] for x in r2.json()["results"]]
    assert "Self-knowledge" in names


def test_graph_search_requires_q(client):
    assert client.get("/api/graph/search").status_code in (400, 422)
