"""Canonical knowledge-graph models (M7).

Every trusted relationship carries immutable evidence linking back to
chunk -> block -> page -> document. Confidence is an explicitly
heuristic score (NOT a probability) derived from documented evidence
signals.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

GRAPH_SCHEMA_VERSION = "graph-schema-1"
VOCAB_VERSION = "jung-vocab-1"
EXTRACTOR_VERSION = "relation-extractor-1"

NODE_TYPES = (
    "CONCEPT", "ARCHETYPE", "PERSON", "WORK", "SYMBOL",
    "PLACE", "THEME", "UNKNOWN",
)

RELATIONSHIP_TYPES = (
    "RELATED_TO",        # co-occurrence within a sentence (no explicit verb)
    "PART_OF",
    "CONTRASTS_WITH",
    "ASSOCIATED_WITH",   # same block/chunk, no sentence-level link
    "DEVELOPS",
    "INTEGRATES",
    "SYMBOLIZES",
    "DISCUSSES",
    "INFLUENCES",
    "REFERENCES",
)

EDGE_STATUSES = ("TRUSTED", "WEAK", "UNVERIFIED")


class GraphEvidence(BaseModel):
    """Immutable provenance for one graph assertion."""
    evidence_id: str
    document_id: str
    chunk_id: str
    page_numbers: List[int]
    source_block_ids: List[str] = []
    heading_path: List[str] = []
    evidence_text: str                  # exact span from the chunk
    relation_source: str                # sentence | block | chunk | pattern
    signal: str                         # extraction signal description

    def preview(self, n: int = 160) -> str:
        return " ".join(self.evidence_text.split())[:n]


class GraphNode(BaseModel):
    node_id: str                        # deterministic slug of canonical name
    canonical_name: str
    node_type: str = "CONCEPT"
    aliases: List[str] = []
    description: str = ""
    source_count: int = 0               # distinct documents mentioning it
    evidence_count: int = 0             # distinct evidence spans
    metadata: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    edge_id: str                        # deterministic: src|rel|tgt|evidence-hash
    source_node_id: str
    target_node_id: str
    relationship_type: str
    confidence: float = 0.0             # heuristic score in [0,1]; NOT a probability
    evidence_ids: List[str] = []
    evidence_count: int = 0
    status: str = "UNVERIFIED"          # TRUSTED | WEAK | UNVERIFIED

    @model_validator(mode="after")
    def check(self):
        if self.source_node_id == self.target_node_id:
            raise ValueError(
                f"self-edge not permitted: {self.edge_id}")
        if self.relationship_type not in RELATIONSHIP_TYPES:
            raise ValueError(
                f"unknown relationship type {self.relationship_type!r}")
        if self.status not in EDGE_STATUSES:
            raise ValueError(f"unknown edge status {self.status!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0,1]")
        return self


class GraphState(BaseModel):
    """Build metadata used for staleness detection."""
    graph_schema_version: str = GRAPH_SCHEMA_VERSION
    vocab_version: str = VOCAB_VERSION
    extractor_version: str = EXTRACTOR_VERSION
    corpus_fingerprint: str = ""        # BM25-style chunk corpus fingerprint
    document_sha256: Dict[str, str] = {}
    built_at: str = ""
    build_time_s: Optional[float] = None


class GraphSnapshot(BaseModel):
    state: GraphState
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    evidence: List[GraphEvidence] = []

    def node_by_id(self, node_id: str) -> Optional[GraphNode]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def validate(self) -> List[str]:
        from jung_archive.graph.validation import validate_graph

        return validate_graph(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
