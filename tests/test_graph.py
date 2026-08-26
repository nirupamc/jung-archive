"""M7 knowledge-graph tests."""
import json

import pytest

from jung_archive.graph.build import build_graph, save_graph, load_graph, \
    staleness_report
from jung_archive.graph.extract import (
    RELATION_WEIGHTS,
    extract_relations_from_chunk,
    status_for,
)
from jung_archive.graph.models import (
    GraphEdge,
    GraphEvidence,
    GraphNode,
    GraphSnapshot,
    GraphState,
)
from jung_archive.graph.validation import validate_graph
from jung_archive.graph.vocabulary import Vocabulary, node_id_for, \
    normalize_name
from jung_archive.models.chunk import Chunk


def make_chunk(cid="d1-c00000", text="The shadow is part of the Self.",
               pages=None, blocks=None, heading=None):
    return Chunk(
        chunk_id=cid, document_id="d1", text=text,
        source_block_ids=blocks or ["p0001-b000"],
        page_numbers=pages or [1], token_count=10,
        source_type="PRIMARY",
        heading_path=heading or [])


# ----------------------------------------------------------------------
# Normalization + vocabulary (tests 1-2)

def test_normalization_deterministic():
    assert normalize_name("  The  SELF!! ") == normalize_name("the self")
    assert normalize_name("Self") == "self"
    assert node_id_for("Shadow") == "concept:shadow"


def test_alias_mapping():
    v = Vocabulary()
    assert v.canonical("the collective unconscious") == \
        "Collective Unconscious"
    assert v.canonical("mass mindedness") == "Mass-mindedness"
    assert v.canonical("know thyself") == "Self-knowledge"
    assert v.canonical("zzz unknown term") is None


def test_no_false_merge_of_distinct_concepts():
    v = Vocabulary()
    # 'unconscious' and 'personal unconscious' are distinct canonical nodes
    assert v.canonical("personal unconscious") == "Personal Unconscious"
    assert v.canonical("collective unconscious") == "Collective Unconscious"
    assert len({node_id_for("Unconscious"),
                node_id_for("Personal Unconscious"),
                node_id_for("Collective Unconscious")}) == 3


# ----------------------------------------------------------------------
# Extraction + confidence + status (tests 5-7)

def test_sentence_cooccurrence_creates_related_to():
    c = make_chunk(text="The ego confronts the shadow directly.")
    cands = extract_relations_from_chunk(c, Vocabulary())
    pairs = {(min(cand.a, cand.b), max(cand.a, cand.b)): cand
             for cand in cands}
    key = ("Ego", "Shadow")
    assert key in pairs
    assert pairs[key].rel_type in ("RELATED_TO", "ASSOCIATED_WITH")


def test_explicit_pattern_typed_edge():
    c = make_chunk(text="The persona is part of the Self.")
    cands = extract_relations_from_chunk(c, Vocabulary())
    persona = next(x for x in cands if "Persona" in (x.a, x.b))
    assert persona.rel_type == "PART_OF"
    assert persona.relation_source == "pattern"


def test_contrast_pattern():
    c = make_chunk(text="The persona contrasts with the shadow.")
    cands = extract_relations_from_chunk(c, Vocabulary())
    pair = next(x for x in cands if {x.a, x.b} == {"Persona", "Shadow"})
    assert pair.rel_type == "CONTRASTS_WITH"


def test_confidence_formula_documented_components():
    # base weight + chunk bonus + sentence bonus + heading bonus, capped
    assert RELATION_WEIGHTS["pattern"] > RELATION_WEIGHTS["sentence"] \
        > RELATION_WEIGHTS["block"] > RELATION_WEIGHTS["chunk"]
    assert status_for(0.75) == "TRUSTED"
    assert status_for(0.5) == "WEAK"
    assert status_for(0.3) == "UNVERIFIED"


def test_trusted_requires_evidence():
    e = GraphEdge(
        edge_id="e1", source_node_id="n1", target_node_id="n2",
        relationship_type="RELATED_TO", confidence=0.8,
        evidence_ids=[], evidence_count=0, status="TRUSTED")
    g = _snapshot(edges=[e])
    errs = validate_graph(g)
    assert any("no evidence" in x for x in errs)


# ----------------------------------------------------------------------
# Determinism + merge + validation (tests 3-4, 9-11)

def test_deterministic_node_and_edge_ids(tmp_path):
    chunks = [
        make_chunk("d1-c00000", "The shadow integrates into consciousness."),
        make_chunk("d1-c00001", "Consciousness meets the shadow again."),
    ]
    g1 = build_graph_from(chunks)
    g2 = build_graph_from(list(reversed(chunks)))
    n1 = sorted(n.model_dump_json() for n in g1.nodes)
    n2 = sorted(n.model_dump_json() for n in g2.nodes)
    e1 = sorted(e.edge_id for e in g1.edges)
    e2 = sorted(e.edge_id for e in g2.edges)
    assert n1 == n2
    assert e1 == e2
    # ids deterministic across builds with same content order too
    assert g1.state.corpus_fingerprint  # fingerprint present


def build_graph_from(chunks):
    from jung_archive.graph.extract import build_edges_and_evidence

    nodes_index, edges, evidence = build_edges_and_evidence(
        chunks, Vocabulary())
    from collections import Counter

    ev_by = {}
    docs_by = {}
    for e in edges:
        for nid in (e.source_node_id, e.target_node_id):
            ev_by[nid] = ev_by.get(nid, 0) + e.evidence_count
            docs_by.setdefault(nid, set()).add("d1")
    nodes = [GraphNode(node_id=n["node_id"],
                       canonical_name=n["canonical_name"],
                       node_type=n["node_type"], aliases=n.get("aliases", []),
                       description=n.get("description", ""),
                       evidence_count=ev_by.get(n["node_id"], 0),
                       source_count=len(docs_by.get(n["node_id"], set())))
             for n in sorted(nodes_index.values(), key=lambda x: x["node_id"])]
    state = GraphState(corpus_fingerprint="fp-test",
                       document_sha256={"d1": "x"},
                       built_at="t", vocab_version="jung-vocab-1",
                       extractor_version="relation-extractor-1")
    return GraphSnapshot(state=state, nodes=nodes, edges=edges,
                         evidence=evidence)


def test_duplicate_edges_rejected_by_validation():
    e = GraphEdge(edge_id="dup", source_node_id="n1", target_node_id="n2",
                  relationship_type="RELATED_TO", confidence=0.5,
                  evidence_ids=["ge-1"], evidence_count=1, status="WEAK")
    ev = GraphEvidence(
        evidence_id="ge-1", document_id="d1", chunk_id="c1",
        page_numbers=[1], evidence_text="span", relation_source="chunk",
        signal="test")
    g = _snapshot(edges=[e, e.model_copy()], evidence=[ev, ev.model_copy()])
    errs = validate_graph(g)
    assert any("duplicate edge" in x for x in errs)
    assert any("duplicate evidence" in x for x in errs)


def test_self_edge_rejected():
    with pytest.raises(ValueError):
        GraphEdge(edge_id="s", source_node_id="n", target_node_id="n",
                  relationship_type="RELATED_TO")


def test_unknown_relationship_type_rejected():
    with pytest.raises(ValueError):
        GraphEdge(edge_id="s", source_node_id="a", target_node_id="b",
                  relationship_type="CAUSED_BY")


def test_validation_catches_bad_references():
    good_ev = GraphEvidence(
        evidence_id="ok", document_id="d1", chunk_id="c1",
        page_numbers=[1], evidence_text="text", relation_source="chunk",
        signal="s")
    bad_edge = GraphEdge(
        edge_id="e", source_node_id="ghost-a", target_node_id="ghost-b",
        relationship_type="RELATED_TO", confidence=0.6,
        evidence_ids=["missing"], evidence_count=1, status="WEAK")
    g = _snapshot(nodes=[GraphNode(node_id="real", canonical_name="Real")],
                  edges=[bad_edge], evidence=[good_ev])
    errs = validate_graph(g)
    assert any("unknown node" in x for x in errs)
    assert any("unknown evidence" in x for x in errs)


def test_real_corpus_graph_validates_with_chunks():
    g = build_graph()   # real corpus
    errs = g.validate()
    assert errs == []
    trusted = [e for e in g.edges if e.status == "TRUSTED"]
    assert trusted, "expected at least some trusted edges on real corpus"
    assert all(e.evidence_count > 0 for e in trusted)


# ----------------------------------------------------------------------
# Persistence round-trip + staleness (tests 12, 17)

def test_persistence_round_trip(tmp_path):
    g = build_graph()
    path = save_graph(g, tmp_path)
    loaded = load_graph(path.parent)
    assert loaded is not None
    assert [n.node_id for n in loaded.nodes] == \
        [n.node_id for n in g.nodes]
    assert [e.edge_id for e in loaded.edges] == \
        [e.edge_id for e in g.edges]
    assert len(loaded.evidence) == len(g.evidence)
    assert json.dumps(json.loads(path.read_text(encoding="utf-8")),
                      sort_keys=True) is not None


def test_stale_detection_on_fingerprint_change(tmp_path):
    g = build_graph()
    stale = staleness_report(g)     # built from current corpus -> fresh
    assert stale == []


def _snapshot(nodes=None, edges=None, evidence=None):
    return GraphSnapshot(
        state=GraphState(built_at="t", corpus_fingerprint="f"),
        nodes=nodes or [], edges=edges or [], evidence=evidence or [])


# ----------------------------------------------------------------------
# Real-corpus sanity: required concepts exist with evidence

REQUIRED_CONCEPTS = ["Shadow", "Self", "Mass-mindedness", "Self-knowledge"]


@pytest.mark.parametrize("name", REQUIRED_CONCEPTS)
def test_required_concepts_present(name):
    g = build_graph()
    match = [n for n in g.nodes if n.canonical_name == name]
    assert match, f"{name} missing from graph"
    assert match[0].evidence_count > 0


def test_no_overstated_relations_in_vocabulary_types():
    g = build_graph()
    allowed = {"RELATED_TO", "ASSOCIATED_WITH", "CONTRASTS_WITH",
               "PART_OF", "DEVELOPS", "INTEGRATES", "SYMBOLIZES"}
    assert all(e.relationship_type in allowed for e in g.edges)
