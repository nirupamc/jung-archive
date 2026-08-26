// Pure, testable data adapter between the M7 graph API and the 3D
// force-graph renderer. No Three.js imports live here: this module is
// safe to unit-test in jsdom.

import type {
  GraphEdgeDto,
  GraphNodeDto,
  GraphOverview,
} from "@/lib/graph";

export type GraphStatus = "TRUSTED" | "WEAK" | "UNVERIFIED";

export interface GraphFilters {
  trustedOnly: boolean;
  minEvidence: number;
  /** empty/null = all relationship types */
  relationTypes: string[] | null;
  /** empty/null = all node types */
  nodeTypes: string[] | null;
}

export const DEFAULT_GRAPH_FILTERS: GraphFilters = {
  trustedOnly: false,
  minEvidence: 0,
  relationTypes: null,
  nodeTypes: null,
};

// Performance guards: the visible subgraph is always bounded.
export const MAX_VISIBLE_NODES = 80;
export const MAX_VISIBLE_LINKS = 160;

export interface Neighborhood {
  centerId: string;
  nodes: GraphNodeDto[];
  edges: GraphEdgeDto[];
  hiddenNodes: number;
  hiddenEdges: number;
}

function edgePassesFilters(e: GraphEdgeDto, f: GraphFilters): boolean {
  if (f.trustedOnly && e.status !== "TRUSTED") return false;
  if (e.evidence_count < f.minEvidence) return false;
  if (f.relationTypes && !f.relationTypes.includes(e.relationship_type)) {
    return false;
  }
  return true;
}

function nodePassesFilters(n: GraphNodeDto, f: GraphFilters): boolean {
  if (f.nodeTypes && !f.nodeTypes.includes(n.node_type)) return false;
  return true;
}

/**
 * Selected node + 1-hop neighborhood (optionally extended by `rings`),
 * filtered and capped. Neighbors are prioritized by evidence so a cap
 * keeps the best-supported part of the neighborhood.
 */
export function buildNeighborhood(
  overview: GraphOverview,
  centerId: string | null,
  filters: GraphFilters,
  rings = 1,
): Neighborhood {
  const nodeById = new Map(overview.nodes.map((n) => [n.node_id, n]));
  const edges = overview.edges.filter((e) => edgePassesFilters(e, filters));

  let frontier: Set<string>;
  let effectiveCenter: string;
  const centerOk =
    centerId &&
    nodeById.has(centerId) &&
    nodePassesFilters(nodeById.get(centerId)!, filters);
  if (centerOk) {
    frontier = new Set([centerId]);
    effectiveCenter = centerId;
  } else {
    // default center: best-supported node that survives filtering
    const ranked = [...overview.nodes]
      .filter((n) => nodePassesFilters(n, filters))
      .sort(
        (a, b) =>
          b.evidence_count - a.evidence_count ||
          a.canonical_name.localeCompare(b.canonical_name),
      );
    effectiveCenter = ranked.length ? ranked[0].node_id : "";
    frontier = new Set(effectiveCenter ? [effectiveCenter] : []);
  }

  const keep = new Set(frontier);
  // ring expansion: BFS over filtered edges
  for (let r = 0; r < Math.max(1, rings); r++) {
    const next = new Set<string>();
    for (const e of edges) {
      if (
        frontier.has(e.source_node_id) &&
        nodeById.has(e.target_node_id) &&
        nodePassesFilters(nodeById.get(e.target_node_id)!, filters)
      ) {
        next.add(e.target_node_id);
      }
      if (
        frontier.has(e.target_node_id) &&
        nodeById.has(e.source_node_id) &&
        nodePassesFilters(nodeById.get(e.source_node_id)!, filters)
      ) {
        next.add(e.source_node_id);
      }
    }
    next.forEach((id) => keep.add(id));
    if (next.size === 0) break;
    frontier = next;
  }

  // Cap: strongest-supported nodes first (center always kept).
  const candidates = [...keep]
    .map((id) => nodeById.get(id))
    .filter((n): n is GraphNodeDto => !!n);
  const ordered = [...candidates].sort(
    (a, b) =>
      b.evidence_count - a.evidence_count ||
      a.canonical_name.localeCompare(b.canonical_name),
  );
  const nodes =
    ordered.length > MAX_VISIBLE_NODES ? ordered.slice(0, MAX_VISIBLE_NODES) : ordered;
  const hiddenNodes = Math.max(0, ordered.length - nodes.length);
  const keptIds = new Set(nodes.map((n) => n.node_id));
  if (!keptIds.size || !effectiveCenter) {
    return { centerId: "", nodes: [], edges: [], hiddenNodes: 0, hiddenEdges: 0 };
  }

  // Edges inside the kept set; prioritize TRUSTED then evidence count.
  const inSet = edges.filter(
    (e) => keptIds.has(e.source_node_id) && keptIds.has(e.target_node_id),
  );
  const statusRank: Record<GraphStatus, number> = {
    TRUSTED: 0,
    WEAK: 1,
    UNVERIFIED: 2,
  };
  inSet.sort(
    (a, b) =>
      statusRank[a.status] - statusRank[b.status] ||
      b.evidence_count - a.evidence_count,
  );
  const shown =
    inSet.length > MAX_VISIBLE_LINKS ? inSet.slice(0, MAX_VISIBLE_LINKS) : inSet;

  return {
    centerId: effectiveCenter,
    nodes,
    edges: shown,
    hiddenNodes,
    hiddenEdges: Math.max(0, inSet.length - shown.length),
  };
}

/** Node visual weight from evidence count (sqrt keeps outliers sane). */
export function nodeVal(evidenceCount: number): number {
  return 0.5 + Math.sqrt(Math.max(0, evidenceCount)) / 2.2;
}

export const NODE_TYPE_COLORS: Record<string, string> = {
  ARCHETYPE: "#c8963e",
  CONCEPT: "#7fb4c9",
  THEME: "#9aa87f",
  PERSON: "#cf8f8f",
  WORK: "#a48fb8",
  SYMBOL: "#6fae8f",
  PLACE: "#8c9bab",
};

export function nodeColor(type: string): string {
  return NODE_TYPE_COLORS[type.toUpperCase()] ?? "#8b8778";
}

export const STATUS_COLORS: Record<GraphStatus, string> = {
  TRUSTED: "#d9a441",
  WEAK: "#8b8778",
  UNVERIFIED: "#55524a",
};

export const STATUS_WIDTHS: Record<GraphStatus, number> = {
  TRUSTED: 1.6,
  WEAK: 1.0,
  UNVERIFIED: 0.7,
};

export function edgeColor(status: string, selected: boolean): string {
  if (selected) return "#f3ecdd";
  return STATUS_COLORS[(status as GraphStatus)] ?? STATUS_COLORS.UNVERIFIED;
}

export interface ForceNode {
  id: string;
  name: string;
  nodeType: string;
  evidenceCount: number;
  sourceCount: number;
  isCenter: boolean;
  val: number;
  color: string;
}

export interface ForceLink {
  edgeId: string;
  source: string;
  target: string;
  relationType: string;
  status: GraphStatus;
  evidenceCount: number;
  color: string;
  width: number;
}

export interface Graph3DData {
  nodes: ForceNode[];
  links: ForceLink[];
}

/** Project API DTOs into renderer-shaped plain objects. */
export function toGraphData(
  nb: Neighborhood,
  selectedEdgeId: string | null = null,
): Graph3DData {
  return {
    nodes: nb.nodes.map((n) => ({
      id: n.node_id,
      name: n.canonical_name,
      nodeType: n.node_type,
      evidenceCount: n.evidence_count,
      sourceCount: n.source_count,
      isCenter: n.node_id === nb.centerId,
      val: nodeVal(n.evidence_count),
      color: nodeColor(n.node_type),
    })),
    links: nb.edges.map((e) => ({
      edgeId: e.edge_id,
      source: e.source_node_id,
      target: e.target_node_id,
      relationType: e.relationship_type,
      status: e.status,
      evidenceCount: e.evidence_count,
      color: edgeColor(e.status, e.edge_id === selectedEdgeId),
      width: STATUS_WIDTHS[e.status] ?? 1,
    })),
  };
}

export function availableNodeTypes(nodes: GraphNodeDto[]): string[] {
  return [...new Set(nodes.map((n) => n.node_type))].sort();
}

export function availableRelationTypes(edges: GraphEdgeDto[]): string[] {
  return [...new Set(edges.map((e) => e.relationship_type))].sort();
}
