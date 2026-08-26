"""Evidence-backed relationship extraction (M7).

Conservative, deterministic rules only. Signals, strongest first:

  1. sentence co-occurrence + explicit relation phrase -> typed edge
     (e.g. "X is part of Y" -> PART_OF; "X contrasts with / unlike Y"
     -> CONTRASTS_WITH; "symbolizes / symbol of" -> SYMBOLIZES;
     "develops / development of" -> DEVELOPS)
  2. sentence co-occurrence without a phrase -> RELATED_TO
  3. same block, different sentences      -> ASSOCIATED_WITH
  4. same chunk, different blocks         -> ASSOCIATED_WITH

Nothing weaker than same-chunk creates an edge, and every edge keeps
the exact evidence span.

Confidence formula (heuristic score in [0,1], NOT a probability):

    base        = relation weight
                  (typed pattern 0.6, sentence RELATED_TO 0.5,
                   block ASSOCIATED_WITH 0.3, chunk 0.2)
    + 0.05 per additional independent supporting chunk   (cap +0.20)
    + 0.10 if the pair also co-occurs within one sentence somewhere
    + 0.05 if the pair co-occurs inside a heading context
    = min(1.0, total)

Status thresholds: >=0.70 TRUSTED, >=0.40 WEAK, else UNVERIFIED.
"""
import re
from typing import Dict, List, Optional

from jung_archive.graph.models import GraphEdge, GraphEvidence
from jung_archive.graph.vocabulary import Vocabulary, node_id_for

# Explicit linguistic patterns -> relationship types.
PATTERNS = [
    (re.compile(r"\bpart of\b|\bcomponent of\b", re.I), "PART_OF"),
    (re.compile(r"\bcontrast(s|ing)? (with|to)\b|\bunlike\b|\bopposed? to\b",
                re.I), "CONTRASTS_WITH"),
    (re.compile(r"\bsymbol(ize|ize s|ises|ic)?\b|\bsymbol of\b", re.I),
     "SYMBOLIZES"),
    (re.compile(r"\bdevelop(s|ment|ed)? (of|into|from)\b", re.I), "DEVELOPS"),
    (re.compile(r"\bintegrat(e|es|ion) (of|with|into)\b", re.I), "INTEGRATES"),
    (re.compile(r"\binfluence[sd]? by\b", re.I), "INFLUENCES"),
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

RELATION_WEIGHTS = {
    "pattern": 0.6,
    "sentence": 0.5,
    "block": 0.3,
    "chunk": 0.2,
}


def status_for(confidence: float) -> str:
    if confidence >= 0.70:
        return "TRUSTED"
    if confidence >= 0.40:
        return "WEAK"
    return "UNVERIFIED"


class RelationCandidate:
    __slots__ = ("a", "b", "relation_source", "rel_type", "signal",
                 "evidence_text", "heading_context")

    def __init__(self, a, b, relation_source, rel_type, signal,
                 evidence_text, heading_context=False):
        self.a = a
        self.b = b
        self.relation_source = relation_source
        self.rel_type = rel_type
        self.signal = signal
        self.evidence_text = evidence_text
        self.heading_context = heading_context


def _pair_key(a: str, b: str) -> tuple:
    return tuple(sorted([node_id_for(a), node_id_for(b)]))


def extract_relations_from_chunk(
    chunk,
    vocab: Vocabulary,
) -> List[RelationCandidate]:
    """Extract candidate relations from one canonical chunk."""
    text = " ".join(chunk.text.split())
    if not text:
        return []
    heading_context = bool(chunk.heading_path)

    # Split into pseudo-sentences and blocks by blank-line groups.
    raw_sentences = [s.strip() for s in SENTENCE_SPLIT.split(text)
                     if len(s.split()) >= 4]
    blocks: List[List[str]] = [[]]
    for s in raw_sentences:
        blocks[-1].append(s)
    candidates: Dict[tuple, RelationCandidate] = {}

    def consider(a, b, source, rel_type, signal, span):
        if node_id_for(a) == node_id_for(b):
            return
        key = (_pair_key(a, b))
        # keep the strongest signal per pair within this chunk
        rank = {"pattern": 0, "sentence": 1, "block": 2, "chunk": 3}
        existing = candidates.get(key)
        if existing is None or rank[source] < rank[existing.relation_source]:
            candidates[key] = RelationCandidate(
                a, b, source, rel_type, signal, span, heading_context)

    for block in blocks:
        for sent in block:
            mentions = vocab.find_mentions(sent)
            names = list(dict.fromkeys(m["canonical"] for m in mentions))
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    m = PATTERNS
                    matched = next(
                        (rt for pat, rt in m if pat.search(sent)), None)
                    if matched:
                        consider(a, b, "pattern", matched,
                                 f"explicit pattern in sentence: "
                                 f"'{sent[:80]}'", sent)
                    else:
                        consider(a, b, "sentence", "RELATED_TO",
                                 f"sentence co-occurrence: '{sent[:80]}'",
                                 sent)
        # block-level pairs not already sentence-linked
        block_names = set()
        for sent in block:
            for mm in vocab.find_mentions(sent):
                block_names.add(mm["canonical"])
        names = sorted(block_names)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                consider(names[i], names[j], "block", "ASSOCIATED_WITH",
                         "same-block co-occurrence", " ".join(block)[:120])

    return list(candidates.values())


def build_edges_and_evidence(chunks, vocab: Vocabulary):
    """Deterministically merge candidates across chunks into final graph.

    Returns (nodes_index, edges, evidence).
    """
    pair_candidates: Dict[tuple, List[RelationCandidate]] = {}
    chunk_ids_by_pair: Dict[tuple, set] = {}
    doc_ids_by_pair: Dict[tuple, set] = {}
    sentence_pair: Dict[tuple, bool] = {}
    heading_pair: Dict[tuple, bool] = {}

    for chunk in chunks:
        cands = extract_relations_from_chunk(chunk, vocab)
        for cand in cands:
            key = _pair_key(cand.a, cand.b)
            pair_candidates.setdefault(key, []).append(cand)
            chunk_ids_by_pair.setdefault(key, set()).add(chunk.chunk_id)
            doc_ids_by_pair.setdefault(key, set()).add(chunk.document_id)
            if cand.relation_source == "sentence":
                sentence_pair[key] = True
            if cand.heading_context:
                heading_pair[key] = True

    edges: Dict[str, GraphEdge] = {}
    evidence_list: List[GraphEvidence] = []
    nodes_index: Dict[str, dict] = {}

    def ensure_node(canonical: str):
        from jung_archive.graph.vocabulary import normalize_name

        nid = node_id_for(canonical)
        if nid not in nodes_index:
            canonical_key = vocab.alias_to_canonical.get(
                normalize_name(canonical), "")
            concept = vocab.by_normalized.get(
                normalize_name(canonical_key))
            nodes_index[nid] = {
                "node_id": nid,
                "canonical_name": canonical,
                "node_type": concept.node_type if concept else "CONCEPT",
                "aliases": list(concept.aliases) if concept else [],
                "description": concept.description if concept else "",
            }
        return nodes_index[nid]

    for key in sorted(pair_candidates):          # deterministic order
        cands = pair_candidates[key]
        # strongest relation source wins; pattern beats sentence etc.
        best = min(cands, key=lambda c: {"pattern": 0, "sentence": 1,
                                         "block": 2, "chunk": 3}[
                       c.relation_source])
        a_node = ensure_node(best.a)
        b_node = ensure_node(best.b)

        ev_ids: List[str] = []
        seen_spans = set()
        for cand in sorted(cands, key=lambda c: (c.evidence_text,)):
            span = " ".join(cand.evidence_text.split())[:400]
            digest_src = f"{key}|{span}|{cand.relation_source}"
            import hashlib

            eid = "ge-" + hashlib.sha256(
                digest_src.encode("utf-8")).hexdigest()[:16]
            if eid in seen_spans:
                continue
            seen_spans.add(eid)
            # find owning chunk for this candidate
            owner = None
            for chunk in chunks:
                if cand.evidence_text[:80] in " ".join(
                        chunk.text.split()):
                    owner = chunk
                    break
            if owner is None:
                continue
            evidence_list.append(GraphEvidence(
                evidence_id=eid,
                document_id=owner.document_id,
                chunk_id=owner.chunk_id,
                page_numbers=list(owner.page_numbers),
                source_block_ids=list(owner.source_block_ids),
                heading_path=list(owner.heading_path),
                evidence_text=span,
                relation_source=cand.relation_source,
                signal=cand.signal,
            ))
            ev_ids.append(eid)

        n_chunks = len(chunk_ids_by_pair[key])
        n_docs = len(doc_ids_by_pair[key])
        confidence = RELATION_WEIGHTS[best.relation_source]
        confidence += 0.05 * min(4, max(0, n_chunks - 1))
        if sentence_pair.get(key):
            confidence += 0.10
        if heading_pair.get(key):
            confidence += 0.05
        confidence += 0.02 * (n_docs - 1) if n_docs > 1 else 0.0
        confidence = round(min(1.0, confidence), 4)

        edge_hash = hashlib.sha256(
            "|".join(sorted(ev_ids)).encode("utf-8")).hexdigest()[:12]
        edge_id = f"edge-{edge_hash}"
        edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            source_node_id=a_node["node_id"],
            target_node_id=b_node["node_id"],
            relationship_type=best.rel_type,
            confidence=confidence,
            evidence_ids=ev_ids,
            evidence_count=len(ev_ids),
            status=status_for(confidence),
        )

    return nodes_index, list(edges.values()), evidence_list
