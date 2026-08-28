"""Read-only FastAPI backend for the M5 Document Inspector frontend.

The API is a thin, typed projection over existing M1-M4 artifacts and
services. No retrieval logic is duplicated here; search/evidence
endpoints delegate to the same HybridRetriever / RerankingPipeline /
EvidenceAssembler used by the CLI.

Run:
    python -m uvicorn jung_archive.api.app:app --port 8000
"""
import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from jung_archive.api.ask_schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
)
from jung_archive.api.schemas import (
    BlockOut,
    ChunkOut,
    CorpusStats,
    DocumentSummary,
    EvidenceRequest,
    PageInspection,
    SearchRequest,
)
from jung_archive.config import (
    BM25_DIR,
    CHROMA_DIR,
    CHUNKS_DIR,
    EVAL_DIR,
    GRAPH_DIR,
    PROCESSED_DIR,
)
from jung_archive.models.document import SourceType

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_DIR = REPO_ROOT / "primary"

VALID_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

_lock = threading.Lock()


class ApiStateError(RuntimeError):
    """Raised when required artifacts are missing/unreadable."""


# ----------------------------------------------------------------------
# Artifact loading (cached per process)

@lru_cache(maxsize=8)
def _load_processed(document_id: str) -> Optional[dict]:
    path = PROCESSED_DIR / f"{document_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def load_chunk_artifact_by_doc(document_id: str) -> Optional[dict]:
    path = CHUNKS_DIR / f"{document_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_document_ids() -> List[str]:
    if not CHUNKS_DIR.exists():
        return []
    return sorted(p.stem for p in CHUNKS_DIR.glob("*.json"))


def document_summary(document_id: str) -> DocumentSummary:
    artifact = load_chunk_artifact_by_doc(document_id)
    processed = _load_processed(document_id)
    if artifact is None and processed is None:
        raise HTTPException(status_code=404, detail=f"unknown document {document_id!r}")
    doc_meta = (artifact or {}).get("document") or {}
    page_count = (processed or {}).get("page_count") or doc_meta.get("page_count") or 0
    source_path = doc_meta.get("source_path") or \
        (processed or {}).get("source_path")
    pdf_ok = bool(source_path) and (REPO_ROOT / source_path).exists()
    index_state = load_index_state().get(document_id) or {}
    indexed_ok = bool(index_state.get("chunk_count")) and artifact is not None
    return DocumentSummary(
        document_id=document_id,
        title=doc_meta.get("title") or (processed or {}).get("title"),
        author=doc_meta.get("author") or (processed or {}).get("author"),
        source_type=(doc_meta.get("source_type")
                     or (processed or {}).get("source_type") or "UNKNOWN"),
        index_status=doc_meta.get("index_status", "INCLUDE"),
        page_count=int(page_count),
        chunk_count=int((artifact or {}).get("chunk_count") or 0),
        source_path=source_path,
        has_pdf=pdf_ok,
        status="INDEXED" if indexed_ok else
               ("CHUNKED" if artifact is not None else "PROCESSED"),
        section=str(source_path or "").split("/")[0].split("\\")[0].upper()
        or "UNKNOWN",
        registered=True,
        sha256=(processed or {}).get("source_sha256")
        or doc_meta.get("source_sha256"),
    )


# Corpus discovery (post-M7): every PDF under primary/ and secondary/,
# including files that have never been processed.
_discovery_cache: Dict = {}


def _discovery_fingerprint() -> tuple:
    import hashlib

    h = hashlib.sha256()
    for section in ("primary", "secondary"):
        folder = REPO_ROOT / section
        if not folder.exists():
            continue
        for pdf in sorted(folder.glob("*.pdf")):
            st = pdf.stat()
            h.update(f"{pdf.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    reg = REPO_ROOT / "config" / "document_metadata.json"
    if reg.exists():
        st = reg.stat()
        h.update(f"reg:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def discovered_documents():
    """Discovery with a fingerprint-keyed cache (fresh files invalidate)."""
    from jung_archive.corpus import discover_corpus

    fp = _discovery_fingerprint()
    if _discovery_cache.get("fp") == fp and "docs" in _discovery_cache:
        return _discovery_cache["docs"]
    docs = discover_corpus()
    _discovery_cache.clear()
    _discovery_cache.update({"fp": fp, "docs": docs})
    return docs


def load_index_state() -> Dict:
    from jung_archive.corpus import load_index_state as _lis

    return _lis(CHROMA_DIR)


def discovered_summary(doc) -> DocumentSummary:
    """Project a DiscoveredDocument into the library summary shape."""
    has_chunks = doc.status in ("CHUNKED", "INDEXED")
    return DocumentSummary(
        document_id=doc.document_id or doc.path,
        title=doc.title,
        author=doc.author,
        source_type=doc.source_type,
        index_status=doc.index_status,
        page_count=int(doc.page_count),
        chunk_count=0 if not has_chunks else _chunk_count_for(
            doc.document_id),
        source_path=doc.path,
        has_pdf=True,
        status=doc.status,
        section=doc.section,
        registered=doc.registered,
        registered_reason=doc.reason,
        sha256=doc.sha256,
    )


def _chunk_count_for(document_id):
    artifact = load_chunk_artifact_by_doc(document_id)
    if artifact is None:
        return 0
    return int(artifact.get("chunk_count") or 0)


# ----------------------------------------------------------------------
# Lazy shared services (single instance per process)

_vector_index = None
_bm25 = None
_reranker = None


def get_services():
    """Build (vector_index, bm25, pipeline_factory) lazily."""
    global _vector_index, _bm25, _reranker
    with _lock:
        if _bm25 is None:
            from jung_archive.embedding.provider import LocalSentenceTransformerProvider
            from jung_archive.indexing.vector_index import VectorIndex
            from jung_archive.retrieval.lexical import BM25Retriever

            provider = LocalSentenceTransformerProvider()
            _vector_index = VectorIndex(provider, persist_dir=str(CHROMA_DIR))
            _bm25 = BM25Retriever(chunks_dir=str(CHUNKS_DIR),
                                  state_dir=str(BM25_DIR))
        if _reranker is None:
            from jung_archive.reranking.cross_encoder import LocalCrossEncoderReranker

            _reranker = LocalCrossEncoderReranker()  # lazy model load
    return _vector_index, _bm25, _reranker


def run_search(req: SearchRequest):
    from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
    from jung_archive.retrieval.pipeline import RerankingPipeline, RerankingPipelineConfig

    vi, bm25, reranker = get_services()
    if req.mode == "hybrid_rerank":
        pipe = RerankingPipeline(
            vi, bm25, reranker,
            RerankingPipelineConfig(
                fusion_candidate_k=max(req.fusion_candidate_k, req.top_k),
                rerank_top_k=req.top_k,
            ))
        return pipe.search(req.query, top_k=req.top_k,
                           filters=req.filters or {})
    retriever = HybridRetriever(vi, bm25, HybridRetrieverConfig())
    return retriever.search(req.query, top_k=req.top_k,
                            filters=req.filters or {}, mode=req.mode)


def run_evidence(req: EvidenceRequest):
    from jung_archive.evidence import EvidenceAssembler, EvidenceConfig
    from jung_archive.retrieval.pipeline import RerankingPipeline, RerankingPipelineConfig

    vi, bm25, reranker = get_services()
    pipe = RerankingPipeline(
        vi, bm25, reranker,
        RerankingPipelineConfig(fusion_candidate_k=max(20, req.top_k),
                                rerank_top_k=req.top_k))
    resp = pipe.search(req.question, top_k=req.top_k, filters=req.filters or {})
    assembler = EvidenceAssembler(EvidenceConfig(
        max_evidence_tokens=req.max_tokens, max_evidence_items=req.max_items))
    return assembler.assemble(req.question, resp.results)


def run_ask(req: AskRequest):
    from jung_archive.generation import AskService, OpenAICompatibleProvider

    vi, bm25, reranker = get_services()
    service = AskService(vi, bm25, reranker, OpenAICompatibleProvider())
    return service.ask(req.query, req.filters, req.generation)


# ----------------------------------------------------------------------
# App

def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS") or os.environ.get("FRONTEND_URL") or ""
    extra = [o.strip() for o in raw.split(",") if o.strip()]
    base = ["http://localhost:3000", "http://127.0.0.1:3000"]
    # Also allow NEXT_PUBLIC_API_BASE_URL host for sanity (not needed but harmless)
    return base + extra


app = FastAPI(title="Jung Archive Inspector API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    docs = list_document_ids()
    return {"status": "ok", "documents": len(docs)}


@app.get("/api/documents", response_model=List[DocumentSummary])
def documents():
    """Full library: processed documents PLUS every discovered-but-
    unprocessed PDF. Unprocessed documents are never hidden."""
    out: Dict[str, DocumentSummary] = {}
    try:
        for doc in discovered_documents():
            s = discovered_summary(doc)
            out[s.document_id] = s
    except Exception:
        pass  # discovery is additive; processed list still served
    for doc_id in list_document_ids():
        if doc_id in out:
            continue
        try:
            out[doc_id] = document_summary(doc_id)
        except HTTPException:
            continue
    return list(out.values())


@app.get("/api/corpus")
def corpus_overview():
    """Corpus discovery report: counts by status/section + every file."""
    from jung_archive.corpus import corpus_report

    docs = discovered_documents()
    return {
        "report": corpus_report(docs),
        "documents": [discovered_summary(d).model_dump() for d in docs],
    }


@app.get("/api/corpus/stats", response_model=CorpusStats)
def corpus_stats():
    from jung_archive.corpus import corpus_report

    docs = discovered_documents()
    rep = corpus_report(docs)
    return CorpusStats(
        discovered_total=rep["discovered_total"],
        pages_total=rep["pages_total"],
        included=rep["included"],
        excluded=rep["excluded"],
        review=rep["review"],
        error=rep["error"],
        by_section=rep["by_section"],
        by_status=rep["by_status"],
    )


@app.get("/api/documents/{document_id}", response_model=DocumentSummary)
def document_detail(document_id: str):
    return document_summary(document_id)


@app.get("/api/documents/{document_id}/pages/{page_number}",
         response_model=PageInspection)
def page_inspection(document_id: str, page_number: int):
    processed = _load_processed(document_id)
    if processed is None:
        raise HTTPException(404, f"no canonical JSON for {document_id!r}")
    pages = {p["page_number"]: p for p in processed["pages"]}
    if page_number not in pages:
        raise HTTPException(
            404, f"page {page_number} out of range "
                 f"(1..{processed['page_count']}) for {document_id!r}")
    p = pages[page_number]

    # Measured OCR confidence: mean over blocks that carry a measured value.
    ocr_values = [b["confidence"] for b in p.get("blocks", [])
                  if b.get("confidence") is not None]
    ocr_mean = round(sum(ocr_values) / len(ocr_values), 4) if ocr_values else None

    blocks = [BlockOut(
        block_id=b["block_id"],
        block_type=b["block_type"],
        text=b["text"],
        bbox={k: float(b["bbox"][k]) for k in ("x0", "y0", "x1", "y1")},
        reading_order=b["reading_order"],
        extraction_method=b.get("extraction_method", "NATIVE"),
        confidence=b.get("confidence"),
        heuristic_quality_score=b.get("heuristic_quality_score"),
        font_name=b.get("font_name"),
        font_size=b.get("font_size"),
        page_number=p["page_number"],
    ) for b in p.get("blocks", [])]

    return PageInspection(
        document_id=document_id,
        page_number=page_number,
        width=float(p["width"]),
        height=float(p["height"]),
        classification=p.get("classification", "UNKNOWN"),
        classification_confidence=p.get("classification_confidence"),
        classification_reason=p.get("reason"),
        layout=p.get("layout", "UNKNOWN"),
        layout_confidence=p.get("layout_confidence"),
        layout_reason=p.get("layout_reason"),
        ocr_confidence=ocr_mean,
        warnings=list(p.get("warnings", [])),
        blocks=blocks,
    )


@app.get("/api/documents/{document_id}/structure",
         response_model=List[BlockOut])
def structure(document_id: str,
              page: Optional[int] = Query(default=None)):
    processed = _load_processed(document_id)
    if processed is None:
        raise HTTPException(404, f"no canonical JSON for {document_id!r}")
    items: List[BlockOut] = []
    for p in sorted(processed["pages"], key=lambda x: x["page_number"]):
        if page is not None and p["page_number"] != page:
            continue
        for b in p.get("blocks", []):
            items.append(BlockOut(
                block_id=b["block_id"],
                block_type=b["block_type"],
                text=b["text"],
                bbox={k: float(b["bbox"][k]) for k in ("x0", "y0", "x1", "y1")},
                reading_order=b["reading_order"],
                extraction_method=b.get("extraction_method", "NATIVE"),
                confidence=b.get("confidence"),
                heuristic_quality_score=b.get("heuristic_quality_score"),
                font_name=b.get("font_name"),
                font_size=b.get("font_size"),
                page_number=p["page_number"],
            ))
    return items


@app.get("/api/documents/{document_id}/chunks",
         response_model=List[ChunkOut])
def chunks(document_id: str,
           page: Optional[int] = Query(default=None),
           section: Optional[str] = Query(default=None)):
    artifact = load_chunk_artifact_by_doc(document_id)
    if artifact is None:
        raise HTTPException(404, f"no chunk artifact for {document_id!r}")
    out = []
    for c in artifact.get("chunks", []):
        pages = c.get("page_numbers", [])
        section_id = c.get("section_id")
        if page is not None and page not in pages:
            continue
        if section is not None and section_id != section:
            continue
        out.append(ChunkOut(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            heading_path=c.get("heading_path", []),
            page_numbers=pages,
            token_count=c["token_count"],
            source_type=c.get("source_type",
                              SourceType.UNKNOWN.value),
            source_block_ids=c["source_block_ids"],
            strategy=c.get("strategy"),
            section_id=section_id,
            chunk_index=c.get("chunk_index"),
            start_page=c.get("start_page"),
            end_page=c.get("end_page"),
            char_count=c.get("char_count"),
            text=c["text"],
        ))
    return out


@app.get("/api/documents/{document_id}/pdf")
def document_pdf(document_id: str):
    summary = document_summary(document_id)
    if not summary.source_path or not summary.has_pdf:
        raise HTTPException(404, f"source PDF unavailable for {document_id!r}")
    pdf_path = REPO_ROOT / summary.source_path
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=pdf_path.name)


@app.post("/api/retrieval/search")
def retrieval_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query must be non-empty")
    mode = req.mode.lower().strip()
    if mode not in VALID_MODES:
        raise HTTPException(
            400, f"invalid mode {req.mode!r}; expected one of {VALID_MODES}")
    if req.top_k < 1:
        raise HTTPException(400, "top_k must be >= 1")
    try:
        response = run_search(req)
    except Exception as e:
        raise HTTPException(502, f"retrieval failed: {e}") from e
    return json.loads(response.model_dump_json())


@app.post("/api/evidence/assemble")
def evidence_assemble(req: EvidenceRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must be non-empty")
    if req.max_tokens < 1 or req.max_items < 1 or req.top_k < 1:
        raise HTTPException(400, "max_tokens/max_items/top_k must be >= 1")
    try:
        pack = run_evidence(req)
    except Exception as e:
        raise HTTPException(502, f"evidence assembly failed: {e}") from e
    return json.loads(pack.model_dump_json())


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(400, "query must be non-empty")
    try:
        response = run_ask(req)
    except Exception as e:
        raise HTTPException(502, f"ask failed: {e}") from e
    return json.loads(response.model_dump_json())


# ----------------------------------------------------------------------
# Knowledge graph (M7, read-only)

def get_graph():
    from jung_archive.graph.build import load_graph

    g = load_graph(GRAPH_DIR)
    if g is None:
        raise HTTPException(
            404, "no graph built; run `python -m jung_archive.cli "
                 "graph build` first")
    return g


@app.get("/api/graph")
def graph_overview(include_evidence: bool = False):
    from jung_archive.graph.build import staleness_report

    g = get_graph()
    from collections import Counter

    statuses = Counter(e.status for e in g.edges)
    relations = Counter(e.relationship_type for e in g.edges)
    trusted = [e for e in g.edges if e.status == "TRUSTED"]
    ev_backed = sum(1 for e in g.edges if e.evidence_count > 0)
    return {
        "state": json.loads(g.state.model_dump_json()),
        "stale": staleness_report(g, str(REPO_ROOT / "data" / "chunks")),
        "stats": {
            "node_count": len(g.nodes),
            "edge_count": len(g.edges),
            "evidence_count": len(g.evidence),
            "trusted_edges": statuses.get("TRUSTED", 0),
            "weak_edges": statuses.get("WEAK", 0),
            "unverified_edges": statuses.get("UNVERIFIED", 0),
            "orphan_nodes": sum(
                1 for n in g.nodes
                if not any(e.source_node_id == n.node_id
                           or e.target_node_id == n.node_id
                           for e in g.edges)),
            "avg_evidence_per_trusted_edge":
                round(sum(e.evidence_count for e in trusted) / len(trusted), 1)
                if trusted else 0,
            "evidence_backed_ratio": round(ev_backed / len(g.edges), 3)
            if g.edges else 0,
        },
        "relation_counts": dict(relations),
        "nodes": [json.loads(n.model_dump_json()) for n in g.nodes],
        "edges": [json.loads(e.model_dump_json()) for e in g.edges],
        "evidence": ([json.loads(ev.model_dump_json()) for ev in g.evidence]
                     if include_evidence else None),
    }


@app.get("/api/graph/nodes")
def graph_nodes(node_type: Optional[str] = None, q: Optional[str] = None):
    g = get_graph()
    nodes = g.nodes
    if node_type:
        nodes = [n for n in nodes if n.node_type == node_type.upper()]
    if q:
        ql = q.lower()
        nodes = [n for n in nodes
                 if ql in n.canonical_name.lower()
                 or any(ql in a.lower() for a in n.aliases)]
    return [json.loads(n.model_dump_json()) for n in nodes]


@app.get("/api/graph/nodes/{node_id}")
def graph_node_detail(node_id: str):
    from jung_archive.graph.models import GraphNode

    g = get_graph()
    node = next((n for n in g.nodes if n.node_id == node_id), None)
    if node is None:
        # try search fallback (alias/canonical match)
        matches = graph_search_impl(q=node_id.replace("concept:", ""))
        if matches:
            return graph_node_detail(matches[0]["node_id"])
        raise HTTPException(404, f"unknown node {node_id!r}")
    assert isinstance(node, GraphNode)
    out_edges = []
    for e in g.edges:
        if node_id in (e.source_node_id, e.target_node_id):
            other = e.target_node_id if e.source_node_id == node_id \
                else e.source_node_id
            other_name = next((n.canonical_name for n in g.nodes
                               if n.node_id == other), other)
            out_edges.append({
                **json.loads(e.model_dump_json()),
                "other_node_id": other,
                "other_name": other_name,
                "direction": "outgoing"
                if e.source_node_id == node_id else "incoming",
            })
    # top evidence: strongest edges touching this node
    ev_ids: List[str] = []
    for e in sorted(out_edges, key=lambda x: -x["confidence"]):
        ev_ids.extend(x for x in e["evidence_ids"][:2] if x not in ev_ids)
    evidence = []
    ev_index = {ev.evidence_id: ev for ev in g.evidence}
    for eid in ev_ids[:12]:
        ev = ev_index.get(eid)
        if ev:
            evidence.append(json.loads(ev.model_dump_json()))
    return {
        **json.loads(node.model_dump_json()),
        "edges": out_edges,
        "top_evidence": evidence,
    }


@app.get("/api/graph/nodes/{node_id}/neighbors")
def graph_neighbors(node_id: str, min_confidence: float = 0.0):
    g = get_graph()
    if not any(n.node_id == node_id for n in g.nodes):
        raise HTTPException(404, f"unknown node {node_id!r}")
    neighbors = {}
    for e in g.edges:
        if e.confidence < min_confidence:
            continue
        if node_id in (e.source_node_id, e.target_node_id):
            other = e.target_node_id if e.source_node_id == node_id \
                else e.source_node_id
            name = next((n.canonical_name for n in g.nodes
                         if n.node_id == other), other)
            cur = neighbors.setdefault(other, {
                "node_id": other, "canonical_name": name, "edges": []})
            cur["edges"].append(json.loads(e.model_dump_json()))
    return list(neighbors.values())


@app.get("/api/graph/edges/{edge_id}")
def graph_edge_detail(edge_id: str):
    g = get_graph()
    edge = next((e for e in g.edges if e.edge_id == edge_id), None)
    if edge is None:
        raise HTTPException(404, f"unknown edge {edge_id!r}")
    ev_index = {ev.evidence_id: ev for ev in g.evidence}
    evidence = [json.loads(ev_index[eid].model_dump_json())
                for eid in edge.evidence_ids if eid in ev_index]
    names = {n.node_id: n.canonical_name for n in g.nodes}
    return {
        **json.loads(edge.model_dump_json()),
        "source_name": names.get(edge.source_node_id, ""),
        "target_name": names.get(edge.target_node_id, ""),
        "evidence": evidence,
    }


@app.get("/api/graph/evidence/{evidence_id}")
def graph_evidence_detail(evidence_id: str):
    g = get_graph()
    ev = next((x for x in g.evidence if x.evidence_id == evidence_id), None)
    if ev is None:
        raise HTTPException(404, f"unknown evidence {evidence_id!r}")
    return json.loads(ev.model_dump_json())


@app.get("/api/graph/search")
def graph_search(q: str, limit: int = 10):
    if not q.strip():
        raise HTTPException(400, "q must be non-empty")
    return {"results": graph_search_impl(q, limit)}


def graph_search_impl(q: str, limit: int = 10):
    from jung_archive.graph.vocabulary import Vocabulary, normalize_name

    g = get_graph()
    vocab = Vocabulary()
    canonical = vocab.canonical(q)
    results = []
    for n in g.nodes:
        score = 0.0
        if q.lower() == n.canonical_name.lower():
            score = 3.0
        elif canonical and canonical == n.canonical_name:
            score = 2.5
        elif q.lower() in n.canonical_name.lower():
            score = 1.5
        elif any(q.lower() in a.lower() for a in n.aliases):
            score = 1.0
        else:
            canon_alias = vocab.alias_to_canonical.get(normalize_name(q))
            if canon_alias and canon_alias == n.canonical_name:
                score = 2.5
        if score > 0:
            results.append({**json.loads(n.model_dump_json()),
                            "score": score})
    results.sort(key=lambda r: (-r["score"], r["canonical_name"]))
    return results[:limit]


# ----------------------------------------------------------------------
# Evaluation Lab (M6, read-only)

@app.get("/api/evaluation/runs")
def evaluation_runs():
    from jung_archive.evaluation.runner import list_runs

    return list_runs(EVAL_DIR)


@app.get("/api/evaluation/latest")
def evaluation_latest():
    latest = EVAL_DIR / "latest_summary.json"
    if not latest.exists():
        raise HTTPException(
            404, "no evaluation runs available; run "
                 "`python -m jung_archive.cli eval retrieval` first")
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/evaluation/runs/{run_id}")
def evaluation_run(run_id: str):
    from jung_archive.evaluation.runner import load_run

    matches = sorted((EVAL_DIR / "runs").glob(f"{run_id}*.json")) \
        if (EVAL_DIR / "runs").exists() else []
    if not matches:
        raise HTTPException(404, f"unknown evaluation run {run_id!r}")
    rec = load_run(str(matches[0]))
    return json.loads(rec.model_dump_json())

