import type {
  GraphEdgeDetail,
  GraphNodeDetail,
  GraphOverview,
} from "@/lib/graph";

function ev(id: string, page: number) {
  return {
    evidence_id: id,
    document_id: "381d2da4b68e",
    chunk_id: `381d2da4b68e-c000${page}`,
    page_numbers: [page],
    source_block_ids: [`p000${page}-b000`],
    heading_path: ["Carl Gustav Jung"],
    evidence_text: `Evidence span ${id}: the shadow integrates into consciousness.`,
    relation_source: "sentence",
    signal: "sentence co-occurrence",
  };
}

const NODES = [
  {
    node_id: "concept:shadow",
    canonical_name: "Shadow",
    node_type: "ARCHETYPE",
    aliases: ["the shadow", "dark side"],
    description: "The unrecognized part of the personality.",
    source_count: 1,
    evidence_count: 148,
  },
  {
    node_id: "concept:self",
    canonical_name: "Self",
    node_type: "ARCHETYPE",
    aliases: ["the self"],
    description: "",
    source_count: 1,
    evidence_count: 35,
  },
  {
    node_id: "concept:self-knowledge",
    canonical_name: "Self-knowledge",
    node_type: "THEME",
    aliases: ["know thyself", "self knowledge"],
    description: "",
    source_count: 1,
    evidence_count: 48,
  },
];

const EDGES = [
  {
    edge_id: "edge-t1abc123",
    source_node_id: "concept:shadow",
    target_node_id: "concept:self-knowledge",
    relationship_type: "RELATED_TO",
    confidence: 0.82,
    evidence_ids: ["ge-aaa", "ge-bbb"],
    evidence_count: 2,
    status: "TRUSTED" as const,
  },
  {
    edge_id: "edge-w2xyz789",
    source_node_id: "concept:shadow",
    target_node_id: "concept:self",
    relationship_type: "INTEGRATES",
    confidence: 0.3,
    evidence_ids: ["ge-ccc"],
    evidence_count: 1,
    status: "UNVERIFIED" as const,
  },
];

export const GRAPH_OVERVIEW: GraphOverview = {
  state: {
    graph_schema_version: "graph-schema-1",
    vocab_version: "jung-vocab-1",
    extractor_version: "relation-extractor-1",
    corpus_fingerprint: "fp",
    built_at: "2026-08-25T00:00:00+00:00",
    build_time_s: 1.5,
  },
  stale: [],
  stats: {
    node_count: 3,
    edge_count: 2,
    evidence_count: 3,
    trusted_edges: 1,
    weak_edges: 0,
    unverified_edges: 1,
    orphan_nodes: 0,
    avg_evidence_per_trusted_edge: 2.0,
    evidence_backed_ratio: 1,
  },
  relation_counts: { RELATED_TO: 1, INTEGRATES: 1 },
  nodes: NODES,
  edges: EDGES,
  evidence: null,
};

export const NODE_DETAIL: GraphNodeDetail = {
  ...NODES[0],
  edges: [
    { ...EDGES[0], other_node_id: "concept:self-knowledge",
      other_name: "Self-knowledge", direction: "outgoing" },
    { ...EDGES[1], other_node_id: "concept:self",
      other_name: "Self", direction: "outgoing" },
  ],
  top_evidence: [ev("ge-aaa", 24), ev("ge-bbb", 68)],
};

export const EDGE_DETAIL: GraphEdgeDetail = {
  ...EDGES[0],
  source_name: "Shadow",
  target_name: "Self-knowledge",
  evidence: [ev("ge-aaa", 24), ev("ge-bbb", 68)],
};

