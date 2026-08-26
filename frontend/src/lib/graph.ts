// Typed contracts for the M7 knowledge graph API.

export interface GraphEvidenceDto {
  evidence_id: string;
  document_id: string;
  chunk_id: string;
  page_numbers: number[];
  source_block_ids: string[];
  heading_path: string[];
  evidence_text: string;
  relation_source: string;
  signal: string;
}

export interface GraphNodeDto {
  node_id: string;
  canonical_name: string;
  node_type: string;
  aliases: string[];
  description: string;
  source_count: number;
  evidence_count: number;
}

export interface GraphEdgeDto {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  relationship_type: string;
  confidence: number;
  evidence_ids: string[];
  evidence_count: number;
  status: "TRUSTED" | "WEAK" | "UNVERIFIED";
}

export interface GraphStateDto {
  graph_schema_version: string;
  vocab_version: string;
  extractor_version: string;
  corpus_fingerprint: string;
  built_at: string;
  build_time_s: number | null;
}

export interface GraphStats {
  node_count: number;
  edge_count: number;
  evidence_count: number;
  trusted_edges: number;
  weak_edges: number;
  unverified_edges: number;
  orphan_nodes: number;
  avg_evidence_per_trusted_edge: number;
  evidence_backed_ratio: number;
}

export interface GraphOverview {
  state: GraphStateDto;
  stale: string[];
  stats: GraphStats;
  relation_counts: Record<string, number>;
  nodes: GraphNodeDto[];
  edges: GraphEdgeDto[];
  evidence: GraphEvidenceDto[] | null;
}

export interface GraphNodeEdgeDto extends GraphEdgeDto {
  other_node_id: string;
  other_name: string;
  direction: "incoming" | "outgoing";
}

export interface GraphNodeDetail extends GraphNodeDto {
  edges: GraphNodeEdgeDto[];
  top_evidence: GraphEvidenceDto[];
}

export interface GraphEdgeDetail extends GraphEdgeDto {
  source_name: string;
  target_name: string;
  evidence: GraphEvidenceDto[];
}

export interface GraphSearchResult extends GraphNodeDto {
  score: number;
}
