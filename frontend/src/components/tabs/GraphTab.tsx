"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type {
  GraphEdgeDetail,
  GraphEvidenceDto,
  GraphOverview,
} from "@/lib/graph";
import {
  DEFAULT_GRAPH_FILTERS,
  MAX_VISIBLE_LINKS,
  MAX_VISIBLE_NODES,
  NODE_TYPE_COLORS,
  STATUS_COLORS,
  availableNodeTypes,
  availableRelationTypes,
  buildNeighborhood,
  toGraphData,
  type GraphFilters,
} from "@/lib/graph3d";
import { ForceGraphBoundary } from "@/components/graph/ForceGraphView";
import { useWorkspace } from "@/state/workspace";
import styles from "./GraphTab.module.css";

// Three.js/WebGL live behind this split boundary; nothing 3D initializes
// until the GRAPH tab mounts this component AND WebGL is available.
const ForceGraphView = dynamic(
  () => import("@/components/graph/ForceGraphView"),
  {
    ssr: false,
    loading: () => (
      <div className={styles.graphLoading}>loading 3D engine …</div>
    ),
  },
);

export default function GraphTab() {
  const { traceToSource } = useWorkspace();
  const [overview, setOverview] = useState<GraphOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [nodeDetail, setNodeDetail] = useState<Awaited<
    ReturnType<typeof api.graphNode>
  > | null>(null);
  const [edgeDetail, setEdgeDetail] = useState<GraphEdgeDetail | null>(null);
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<GraphFilters>(DEFAULT_GRAPH_FILTERS);
  const [rings, setRings] = useState(1);

  useEffect(() => {
    let cancelled = false;
    api
      .graphOverview()
      .then((g) => !cancelled && setOverview(g))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  // visible subgraph: selected node + capped neighborhood
  const neighborhood = useMemo(
    () =>
      overview ? buildNeighborhood(overview, selectedNodeId, filters, rings) : null,
    [overview, selectedNodeId, filters, rings],
  );
  const selectedEdgeId = edgeDetail?.edge_id ?? null;
  const graphData = useMemo(
    () => (neighborhood ? toGraphData(neighborhood, selectedEdgeId)
          : { nodes: [], links: [] }),
    [neighborhood, selectedEdgeId],
  );

  const selectNode = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
    setEdgeDetail(null);
    setNodeDetail(null);
    if (nodeId) {
      api
        .graphNode(nodeId)
        .then(setNodeDetail)
        .catch(() => setNodeDetail(null));
    }
  }, []);

  const selectEdge = useCallback((edgeId: string) => {
    setNodeDetail(null);
    api
      .graphEdge(edgeId)
      .then(setEdgeDetail)
      .catch(() => setEdgeDetail(null));
  }, []);

  const runSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setError(null);
    try {
      const res = await api.graphSearch(query.trim());
      if (res.results.length > 0) {
        setRings(1);
        selectNode(res.results[0].node_id);
      } else {
        setError(`no graph concept matches "${query}"`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const relTypes = useMemo(
    () => (overview ? availableRelationTypes(overview.edges) : []),
    [overview],
  );
  const nodeTypes = useMemo(
    () => (overview ? availableNodeTypes(overview.nodes) : []),
    [overview],
  );

  if (error && !overview) {
    return <div role="alert" className={styles.error}>{error}</div>;
  }
  if (!overview || !neighborhood) {
    return (
      <div role="status" className={styles.loading}>
        loading knowledge graph …
      </div>
    );
  }

  const toggleListFilter = (
    key: "relationTypes" | "nodeTypes",
    value: string,
  ) => {
    setFilters((f) => {
      const current = f[key] ?? [];
      const next = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];
      return { ...f, [key]: next.length === 0 ? null : next };
    });
  };

  return (
    <div className={styles.wrap}>
      <div className={styles.controls}>
        <form onSubmit={runSearch} className={styles.searchForm}>
          <label htmlFor="graph-search">concept search</label>
          <input
            id="graph-search"
            value={query}
            onChange={(e2) => setQuery(e2.target.value)}
            placeholder="shadow, know thyself…"
          />
          <button type="submit">search</button>
        </form>
        <label className={styles.checkLabel}>
          <input
            type="checkbox"
            checked={filters.trustedOnly}
            onChange={(e2) =>
              setFilters((f) => ({ ...f, trustedOnly: e2.target.checked }))
            }
            data-testid="graph-trusted-only"
          />{" "}
          trusted only
        </label>
        <label htmlFor="graph-min-ev" className={styles.checkLabel}>
          min evidence
          <input
            id="graph-min-ev"
            data-testid="graph-min-evidence"
            type="number"
            min={0}
            max={50}
            value={filters.minEvidence}
            onChange={(e2) =>
              setFilters((f) => ({
                ...f,
                minEvidence: Number(e2.target.value) || 0,
              }))
            }
            style={{ width: 56 }}
          />
        </label>
        <span className={styles.ringControls}>
          <button
            type="button"
            aria-label="expand neighborhood"
            disabled={rings >= 4}
            onClick={() => setRings((r) => Math.min(4, r + 1))}
          >
            expand +1 hop
          </button>
          <button
            type="button"
            aria-label="collapse neighborhood"
            disabled={rings <= 1}
            onClick={() => setRings(1)}
          >
            collapse
          </button>
        </span>
        <span className={styles.stats} aria-live="polite">
          {neighborhood.nodes.length}
          {neighborhood.hiddenNodes > 0 &&
            ` (+${neighborhood.hiddenNodes} capped)`}{" "}
          nodes · {neighborhood.edges.length}
          {neighborhood.hiddenEdges > 0 &&
            ` (+${neighborhood.hiddenEdges} capped)`}{" "}
          edges shown · corpus {overview.stats.node_count}n/
          {overview.stats.edge_count}e ·{" "}
          {overview.stats.trusted_edges}/{overview.stats.edge_count} trusted
          {overview.stale.length > 0 && (
            <em className={styles.stale}>
              {" "}· stale: {overview.stale.join("; ")}
            </em>
          )}
        </span>
      </div>

      <details className={styles.filterDrawer}>
        <summary>filters & legend</summary>
        <div className={styles.filterBody}>
          <fieldset>
            <legend>relationship types</legend>
            {relTypes.map((rt) => (
              <label key={rt} className={styles.checkLabel}>
                <input
                  type="checkbox"
                  checked={
                    filters.relationTypes?.includes(rt) ?? false
                  }
                  onChange={() => toggleListFilter("relationTypes", rt)}
                  data-testid={`rel-filter-${rt.toLowerCase()}`}
                />{" "}
                {rt.toLowerCase().replace(/_/g, " ")}
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>node types</legend>
            {nodeTypes.map((nt) => (
              <label key={nt} className={styles.checkLabel}>
                <input
                  type="checkbox"
                  checked={filters.nodeTypes?.includes(nt) ?? false}
                  onChange={() => toggleListFilter("nodeTypes", nt)}
                  data-testid={`node-filter-${nt.toLowerCase()}`}
                />{" "}
                <span
                  className={styles.legendSwatch}
                  style={{ background: NODE_TYPE_COLORS[nt] ?? "#8b8778" }}
                />
                {nt.toLowerCase()}
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>edge status</legend>
            {(Object.keys(STATUS_COLORS) as Array<keyof typeof STATUS_COLORS>)
              .map((s) => (
                <span key={s} className={styles.checkLabel}>
                  <span
                    className={`${styles.legendLine} ${styles[`lw_${s}`]}`}
                    style={{ background: STATUS_COLORS[s] }}
                  />
                  {s.toLowerCase()}
                </span>
              ))}
            <p className={styles.legendNote}>
              node size = evidence chunks backing the concept · edge style =
              relation status (heuristic, not probability)
            </p>
          </fieldset>
        </div>
      </details>

      <div className={styles.split}>
        <div className={styles.canvasPane}>
          <ForceGraphBoundary>
            <ForceGraphView
              data={graphData}
              selectedNodeId={selectedNodeId}
              onNodeClick={(id) => selectNode(id)}
              onLinkClick={(edgeId) => selectEdge(edgeId)}
            />
          </ForceGraphBoundary>

          {/* Accessible equivalents of canvas interactions */}
          <div className={styles.chipRows}>
            <div className={styles.chipRow} aria-label="visible concepts">
              {neighborhood.nodes.map((n) => (
                <button
                  key={n.node_id}
                  type="button"
                  data-testid={`gnode-${slug(n.canonical_name)}`}
                  aria-pressed={selectedNodeId === n.node_id}
                  className={[
                    styles.chip,
                    n.node_id === neighborhood.centerId ? styles.centerChip : "",
                    selectedNodeId === n.node_id ? styles.selected : "",
                  ].join(" ")}
                  onClick={() => selectNode(n.node_id)}
                >
                  <span
                    className={styles.legendSwatch}
                    style={{
                      background: NODE_TYPE_COLORS[n.node_type] ?? "#8b8778",
                    }}
                  />
                  {n.canonical_name}
                  <em className={styles.chipCount}>{n.evidence_count}</em>
                </button>
              ))}
            </div>
            <div className={styles.chipRow} aria-label="visible relationships">
              {neighborhood.edges.map((e) => (
                <button
                  key={e.edge_id}
                  type="button"
                  data-testid={`gedge-${short(e.edge_id)}`}
                  aria-pressed={edgeDetail?.edge_id === e.edge_id}
                  className={[
                    styles.edgeChip,
                    styles[`edge_${e.status}`],
                    edgeDetail?.edge_id === e.edge_id ? styles.selected : "",
                  ].join(" ")}
                  onClick={() => selectEdge(e.edge_id)}
                >
                  {shortName(neighborhood.nodes, e.source_node_id)}
                  {" —"}
                  {e.relationship_type.toLowerCase().replace(/_/g, " ")}
                  {"→ "}
                  {shortName(neighborhood.nodes, e.target_node_id)}
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className={styles.detailPane} aria-label="graph detail">
          {nodeDetail && (
            <GraphNodePanel
              detail={nodeDetail}
              onTrace={traceToSource}
              onOpenNode={(id) => selectNode(id)}
            />
          )}
          {edgeDetail && (
            <GraphEdgePanel
              detail={edgeDetail}
              onTrace={traceToSource}
              onClose={() => setEdgeDetail(null)}
            />
          )}
          {!nodeDetail && !edgeDetail && (
            <p className={styles.hint}>
              click a concept or relationship to inspect its evidence
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------
// Panels

function GraphNodePanel({
  detail,
  onTrace,
  onOpenNode,
}: {
  detail: Awaited<ReturnType<typeof api.graphNode>>;
  onTrace: (t: {
    pageNumbers: number[];
    blockIds: string[];
    chunkId?: string | null;
  }) => void;
  onOpenNode: (id: string) => void;
}) {
  const edges = [...(detail.edges ?? [])].sort(
    (a, b) => b.confidence - a.confidence,
  );
  return (
    <section aria-label="node detail" data-testid="node-detail">
      <h3 className={styles.panelTitle}>{detail.canonical_name}</h3>
      <dl className={styles.kv}>
        <dt>type</dt>
        <dd>{detail.node_type}</dd>
        <dt>aliases</dt>
        <dd>{detail.aliases.join(", ") || "—"}</dd>
        <dt>evidence</dt>
        <dd>{detail.evidence_count} chunks</dd>
        <dt>documents</dt>
        <dd>{detail.source_count}</dd>
      </dl>
      {detail.description && (
        <p className={styles.desc}>{detail.description}</p>
      )}
      <h4 className={styles.subTitle}>related concepts</h4>
      <ul className={styles.relatedList}>
        {edges.map((e) => (
          <li key={e.edge_id}>
            <button
              type="button"
              data-testid={`related-${slug(e.other_name)}`}
              className={styles.linkBtn}
              onClick={() => onOpenNode(e.other_node_id)}
            >
              {e.other_name}
            </button>{" "}
            <em className={`${styles.relTag} ${styles[`st_${e.status}`]}`}>
              {e.relationship_type.toLowerCase()} ({e.status.toLowerCase()})
            </em>
          </li>
        ))}
      </ul>
      <h4 className={styles.subTitle}>top supporting evidence</h4>
      <EvidenceList items={detail.top_evidence ?? []} onTrace={onTrace} />
    </section>
  );
}

function GraphEdgePanel({
  detail,
  onTrace,
  onClose,
}: {
  detail: GraphEdgeDetail;
  onTrace: (t: {
    pageNumbers: number[];
    blockIds: string[];
    chunkId?: string | null;
  }) => void;
  onClose: () => void;
}) {
  return (
    <section aria-label="edge detail" data-testid="edge-detail">
      <p className={styles.edgeSentence}>
        <strong>{detail.source_name}</strong>
        <span className={styles.relVerb}>
          {" —"}
          {detail.relationship_type.toLowerCase().replace(/_/g, " ")}

          →{" "}
        </span>
        <strong>{detail.target_name}</strong>
      </p>
      <dl className={styles.kv}>
        <dt>status</dt>
        <dd data-testid="edge-status">{detail.status}</dd>
        <dt>heuristic score</dt>
        <dd data-testid="edge-confidence">
          {detail.confidence.toFixed(2)}{" "}
          <span className={styles.notProb}>(not a probability)</span>
        </dd>
        <dt>support count</dt>
        <dd>{detail.evidence_count} evidence spans</dd>
      </dl>
      <button type="button" className={styles.closeBtn} onClick={onClose}>
        close
      </button>
      <h4 className={styles.subTitle}>supporting evidence</h4>
      <EvidenceList items={detail.evidence} onTrace={onTrace} />
    </section>
  );
}

function EvidenceList({
  items,
  onTrace,
}: {
  items: GraphEvidenceDto[];
  onTrace: (t: {
    pageNumbers: number[];
    blockIds: string[];
    chunkId?: string | null;
  }) => void;
}) {
  if (!items.length) {
    return <p className={styles.hint}>no evidence spans recorded</p>;
  }
  return (
    <ul className={styles.evList}>
      {items.map((ev) => (
        <li key={ev.evidence_id} className={styles.evCard}>
          <p className={styles.evText}>
            {ev.evidence_text.split(/\s+/).slice(0, 40).join(" ").slice(0, 200)}
          </p>
          <p className={styles.evMeta}>
            chunk <code>{shortChunk(ev.chunk_id)}</code> · pages{" "}
            {ev.page_numbers.join(", ")} · via {ev.relation_source}
          </p>
          <button
            type="button"
            className={styles.traceBtn}
            data-testid={`trace-${ev.evidence_id}`}
            onClick={() =>
              onTrace({
                pageNumbers: ev.page_numbers,
                blockIds: ev.source_block_ids,
                chunkId: ev.chunk_id,
              })
            }
          >
            view source →
          </button>
        </li>
      ))}
    </ul>
  );
}

// ----------------------------------------------------------------------
// helpers

function shortChunk(cid: string): string {
  const i = cid.lastIndexOf("-c");
  return i >= 0 ? cid.slice(i + 1) : cid;
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function short(edgeId: string): string {
  return edgeId.replace("edge-", "");
}

function shortName(
  nodes: Array<{ node_id: string; canonical_name: string }>,
  nodeId: string,
): string {
  const n = nodes.find((x) => x.node_id === nodeId);
  return n ? n.canonical_name : nodeId;
}

export { MAX_VISIBLE_LINKS, MAX_VISIBLE_NODES };
