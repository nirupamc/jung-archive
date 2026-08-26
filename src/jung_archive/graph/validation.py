"""Graph validation (M7). Fails loudly on provenance corruption."""
from typing import List

from jung_archive.graph.models import GraphSnapshot


def validate_graph(graph: GraphSnapshot,
                   valid_chunk_ids=None) -> List[str]:
    errors: List[str] = []

    node_ids = [n.node_id for n in graph.nodes]
    dup_nodes = {i for i in node_ids if node_ids.count(i) > 1}
    if dup_nodes:
        errors.append(f"duplicate node ids: {sorted(dup_nodes)}")
    node_set = set(node_ids)

    edge_ids = [e.edge_id for e in graph.edges]
    dup_edges = {i for i in edge_ids if edge_ids.count(i) > 1}
    if dup_edges:
        errors.append(f"duplicate edge ids: {sorted(dup_edges)}")

    evidence_by_id = {e.evidence_id: e for e in graph.evidence}
    ev_ids_all = [e.evidence_id for e in graph.evidence]
    dup_evid = {i for i in ev_ids_all if ev_ids_all.count(i) > 1}
    if dup_evid:
        errors.append(f"duplicate evidence ids: {sorted(dup_evid)}")

    seen_pairs = {}
    for edge in graph.edges:
        for nid in (edge.source_node_id, edge.target_node_id):
            if nid not in node_set:
                errors.append(
                    f"edge {edge.edge_id} references unknown node {nid}")
        if edge.source_node_id == edge.target_node_id:
            errors.append(f"self-edge {edge.edge_id}")
        if edge.status == "TRUSTED" and edge.evidence_count == 0:
            errors.append(
                f"trusted edge {edge.edge_id} has no evidence")
        pair = tuple(sorted([edge.source_node_id, edge.target_node_id]))
        seen_pairs.setdefault(pair, []).append(edge.edge_id)
        # duplicate merged edges (same pair + same relationship type)
        for eid in edge.evidence_ids:
            if eid not in evidence_by_id:
                errors.append(
                    f"edge {edge.edge_id} references unknown evidence {eid}")

    for pair, eids in seen_pairs.items():
        types = [e.relationship_type for e in graph.edges
                 if e.edge_id in eids]
        if len(set(types)) != len(types):
            errors.append(
                f"unmerged duplicate edges between {pair}: {types}")

    chunk_set = set(valid_chunk_ids) if valid_chunk_ids is not None else None
    for ev in graph.evidence:
        if not ev.document_id or not ev.chunk_id:
            errors.append(
                f"evidence {ev.evidence_id} missing document/chunk "
                f"provenance")
        if not ev.page_numbers:
            errors.append(f"evidence {ev.evidence_id} has no pages")
        if not ev.evidence_text.strip():
            errors.append(f"evidence {ev.evidence_id} has empty span")
        if chunk_set is not None and ev.chunk_id not in chunk_set:
            errors.append(
                f"evidence {ev.evidence_id} references unknown chunk "
                f"{ev.chunk_id}")

    return errors
