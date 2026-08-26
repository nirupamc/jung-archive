"""Graph build pipeline + persistence + staleness detection (M7).

    processed documents -> chunks -> concept extraction -> normalization
    -> relationship extraction -> evidence linking -> validation
    -> persisted graph (data/graph/)

Persistence: plain JSON files (graph.json snapshot + state), no Neo4j.
Staleness: the stored corpus fingerprint / doc checksums / vocab and
extractor versions are compared against the current artifacts; a stale
graph is reported rather than silently served.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from jung_archive.graph.models import (
    EXTRACTOR_VERSION,
    GRAPH_SCHEMA_VERSION,
    GraphEdge,
    GraphEvidence,
    GraphNode,
    GraphSnapshot,
    GraphState,
)
from jung_archive.graph.vocabulary import VOCAB_VERSION, Vocabulary

GRAPH_DIR = Path("data/graph")


def load_chunks(chunks_dir: str = "data/chunks"):
    from jung_archive.retrieval.lexical import BM25Retriever
    from jung_archive.retrieval.lexical import _corpus_fingerprint

    retriever = BM25Retriever(chunks_dir=chunks_dir,
                              state_dir=str(Path(chunks_dir).parent / "bm25_graph_tmp"))
    chunks, doc_meta = retriever._load_all_chunks()
    fingerprint = _corpus_fingerprint(chunks)
    return chunks, doc_meta, fingerprint


def build_graph(chunks_dir: str = "data/chunks",
                vocab: Optional[Vocabulary] = None) -> GraphSnapshot:
    """Deterministically build the evidence-backed graph from chunks."""
    started = time.perf_counter()
    vocab = vocab or Vocabulary()
    chunks, doc_meta, fingerprint = load_chunks(chunks_dir)

    nodes_index, edges, evidence = _extract(chunks, vocab)

    # finalize node stats from edges + evidence
    ev_by_node: Dict[str, int] = {}
    docs_by_node: Dict[str, set] = {}
    ev_index = {e.evidence_id for e in evidence}
    for edge in edges:
        for nid in (edge.source_node_id, edge.target_node_id):
            ev_by_node[nid] = ev_by_node.get(nid, 0) + edge.evidence_count
            docs = docs_by_node.setdefault(nid, set())
            for eid in edge.evidence_ids:
                if eid in ev_index:
                    e = next(x for x in evidence if x.evidence_id == eid)
                    docs.add(e.document_id)

    nodes = [GraphNode(
        node_id=n["node_id"],
        canonical_name=n["canonical_name"],
        node_type=n["node_type"],
        aliases=n.get("aliases", []),
        description=n.get("description", ""),
        source_count=len(docs_by_node.get(n["node_id"], set())),
        evidence_count=ev_by_node.get(n["node_id"], 0),
    ) for n in sorted(nodes_index.values(), key=lambda x: x["node_id"])]

    state = GraphState(
        graph_schema_version=GRAPH_SCHEMA_VERSION,
        vocab_version=vocab.version,
        extractor_version=EXTRACTOR_VERSION,
        corpus_fingerprint=fingerprint,
        document_sha256={
            did: (meta or {}).get("source_sha256") or ""
            for did, meta in doc_meta.items()},
        built_at=_now(),
        build_time_s=round(time.perf_counter() - started, 2),
    )
    return GraphSnapshot(state=state, nodes=nodes, edges=edges,
                         evidence=evidence)


def _extract(chunks, vocab):
    from jung_archive.graph.extract import build_edges_and_evidence

    return build_edges_and_evidence(chunks, vocab)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_graph(graph: GraphSnapshot, graph_dir: Path = GRAPH_DIR) -> Path:
    graph_dir.mkdir(parents=True, exist_ok=True)
    payload = graph.model_dump(mode="json")
    path = graph_dir / "graph.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    (graph_dir / "state.json").write_text(
        json.dumps(graph.state.model_dump(), indent=2), encoding="utf-8")
    return path


def load_graph(graph_dir: Path = GRAPH_DIR) -> Optional[GraphSnapshot]:
    path = Path(graph_dir) / "graph.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return GraphSnapshot(**data)


def staleness_report(graph: GraphSnapshot,
                     chunks_dir: str = "data/chunks") -> List[str]:
    """Return reasons why the persisted graph is stale ([] == fresh)."""
    issues: List[str] = []
    s = graph.state
    if s.graph_schema_version != GRAPH_SCHEMA_VERSION:
        issues.append("graph schema version changed")
    if s.vocab_version != VOCAB_VERSION:
        issues.append("concept vocabulary changed")
    if s.extractor_version != EXTRACTOR_VERSION:
        issues.append("relationship extractor changed")
    try:
        _, _, fingerprint = load_chunks(chunks_dir)
    except Exception as e:
        issues.append(f"chunk artifacts unavailable: {e}")
        return issues
    if s.corpus_fingerprint != fingerprint:
        issues.append("chunk corpus changed since graph build")
    return issues




