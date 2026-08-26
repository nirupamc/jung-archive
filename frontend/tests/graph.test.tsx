import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import GraphTab from "@/components/tabs/GraphTab";
import { WorkspaceProvider } from "@/state/workspace";
import TabBar from "@/components/TabBar";
import { GRAPH_OVERVIEW, NODE_DETAIL, EDGE_DETAIL } from "./graphFixtures";
import {
  MAX_VISIBLE_LINKS,
  MAX_VISIBLE_NODES,
  DEFAULT_GRAPH_FILTERS,
  buildNeighborhood,
  nodeColor,
  nodeVal,
  toGraphData,
  availableNodeTypes,
  availableRelationTypes,
} from "@/lib/graph3d";

function mockFetch(url: string) {
  if (url.includes("/api/graph/search")) {
    const q = new URL(url).searchParams.get("q") ?? "";
    const hits = GRAPH_OVERVIEW.nodes.filter(
      (n) =>
        n.canonical_name.toLowerCase().includes(q.toLowerCase()) ||
        n.aliases.some((a) => a.toLowerCase().includes(q.toLowerCase())),
    );
    return new Response(
      JSON.stringify({ results: hits.map((h) => ({ ...h, score: 2 })) }),
      { status: 200 },
    );
  }
  if (url.includes("/api/graph/nodes/")) {
    return new Response(JSON.stringify(NODE_DETAIL), { status: 200 });
  }
  if (url.includes("/api/graph/edges/edge-t1")) {
    return new Response(JSON.stringify(EDGE_DETAIL), { status: 200 });
  }
  if (url.includes("/api/graph")) {
    return new Response(JSON.stringify(GRAPH_OVERVIEW), { status: 200 });
  }
  if (url.includes("/api/documents")) {
    return new Response(JSON.stringify([]), { status: 200 });
  }
  return new Response(JSON.stringify({ detail: "nf" }), { status: 404 });
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn((input: string | Request) =>
    mockFetch(String(input)),
  ));
});
afterEach(() => vi.unstubAllGlobals());

function renderGraph() {
  return render(
    <WorkspaceProvider>
      <GraphTab />
    </WorkspaceProvider>,
  );
}

describe("GraphTab", () => {
  it("renders the neighborhood graph with stats", async () => {
    renderGraph();
    await waitFor(() =>
      expect(screen.getByTestId("gnode-shadow")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("gnode-self")).toBeInTheDocument();
    // stats line shows visible vs corpus counts
    expect(screen.getByText(/edges shown/)).toBeInTheDocument();
  });

  it("selects a node and shows detail with evidence", async () => {
    const user = userEvent.setup();
    renderGraph();
    await screen.findByTestId("gnode-shadow");
    await user.click(screen.getByTestId("gnode-shadow"));
    const panel = await screen.findByTestId("node-detail");
    expect(panel.textContent).toContain("Shadow");
    expect(panel.textContent).toContain("ARCHETYPE");
    // top evidence carries trace buttons
    expect(screen.getAllByRole("button", { name: /view source/ }).length)
      .toBeGreaterThan(0);
  });

  it("traces node evidence to source via workspace state", async () => {
    const user = userEvent.setup();
    render(
      <WorkspaceProvider>
        <GraphTab />
        <TabBar />
      </WorkspaceProvider>,
    );
    await screen.findByTestId("gnode-shadow");
    await user.click(screen.getByTestId("gnode-shadow"));
    const btn = await screen.findByTestId("trace-ge-aaa");
    await user.click(btn);
    // traceToSource switches to the DOCUMENT tab
    await waitFor(() => {
      expect(screen.getByTestId("tab-document")).toHaveAttribute(
        "aria-selected",
        "true",
      );
    });
  });

  it("opens an edge and displays heuristic score with support count", async () => {
    const user = userEvent.setup();
    renderGraph();
    await screen.findByTestId("gnode-shadow");
    await user.click(screen.getByTestId("gedge-t1abc123"));
    const panel = await screen.findByTestId("edge-detail");
    expect(panel.textContent).toContain("TRUSTED");
    expect(screen.getByTestId("edge-confidence").textContent).toContain(
      "0.82",
    );
    expect(panel.textContent).toContain("(not a probability)");
    expect(screen.getByText(/supporting evidence/i)).toBeInTheDocument();
  });

  it("search finds a concept by alias and centers it", async () => {
    const user = userEvent.setup();
    renderGraph();
    await screen.findByLabelText(/concept search/i);
    await user.type(screen.getByLabelText(/concept search/i), "know thyself");
    await user.click(screen.getByRole("button", { name: /search/ }));
    const panel = await screen.findByTestId("node-detail");
    expect(panel.textContent).toContain("Self-knowledge");
  });

  it("trusted-only filter hides weak edges and their nodes", async () => {
    const user = userEvent.setup();
    renderGraph();
    await screen.findByTestId("graph-trusted-only");
    const before = screen.getByText(/edges shown/).textContent;
    await user.click(screen.getByTestId("graph-trusted-only"));
    const after = screen.getByText(/edges shown/).textContent;
    expect(after).not.toEqual(before);
  });

  it("relationship-type filter narrows the edge list", async () => {
    const user = userEvent.setup();
    renderGraph();
    await screen.findByTestId("rel-filter-related_to");
    await user.click(screen.getByTestId("rel-filter-related_to"));
    await waitFor(() => {
      expect(screen.queryByTestId("gedge-w2xyz789")).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("gedge-t1abc123")).toBeInTheDocument();
  });

  it("renders a WebGL fallback instead of crashing without 3D support", async () => {
    renderGraph();
    // jsdom has no WebGL; ForceGraphView must degrade honestly.
    await waitFor(() => {
      expect(screen.getByText(/3D graph unavailable/i)).toBeInTheDocument();
    });
    // ...while the accessible data remains fully usable
    expect(screen.getByTestId("gnode-shadow")).toBeInTheDocument();
  });
});

// ----------------------------------------------------------------------
// lib/graph3d adapter

const NODES = [
  {
    node_id: "concept:shadow",
    canonical_name: "Shadow",
    node_type: "ARCHETYPE",
    aliases: [],
    description: "",
    source_count: 2,
    evidence_count: 148,
  },
  {
    node_id: "concept:self",
    canonical_name: "Self",
    node_type: "ARCHETYPE",
    aliases: [],
    description: "",
    source_count: 1,
    evidence_count: 35,
  },
  {
    node_id: "concept:self-knowledge",
    canonical_name: "Self-knowledge",
    node_type: "THEME",
    aliases: [],
    description: "",
    source_count: 1,
    evidence_count: 48,
  },
];

const EDGES = [
  {
    edge_id: "edge-t1",
    source_node_id: "concept:shadow",
    target_node_id: "concept:self-knowledge",
    relationship_type: "RELATED_TO",
    confidence: 0.82,
    evidence_ids: ["ge-1"],
    evidence_count: 9,
    status: "TRUSTED" as const,
  },
  {
    edge_id: "edge-u2",
    source_node_id: "concept:self-knowledge",
    target_node_id: "concept:self",
    relationship_type: "PART_OF",
    confidence: 0.2,
    evidence_ids: ["ge-2"],
    evidence_count: 1,
    status: "UNVERIFIED" as const,
  },
];

const OVERVIEW = {
  state: GRAPH_OVERVIEW.state,
  stale: [],
  stats: GRAPH_OVERVIEW.stats,
  relation_counts: {},
  nodes: NODES,
  edges: EDGES,
  evidence: null,
};

describe("buildNeighborhood", () => {
  it("defaults to the best-supported hub as center plus its 1-hop ring", () => {
    const nb = buildNeighborhood(OVERVIEW, null, DEFAULT_GRAPH_FILTERS);
    expect(nb.centerId).toBe("concept:shadow"); // highest evidence count
    expect(nb.nodes.map((n) => n.node_id)).toEqual([
      "concept:shadow",
      "concept:self-knowledge",
    ]);
    expect(nb.edges.map((e) => e.edge_id)).toEqual(["edge-t1"]);
  });

  it("expands rings under control (+1 hop pulls in 2-hop nodes)", () => {
    const nb = buildNeighborhood(
      OVERVIEW,
      "concept:self-knowledge",
      DEFAULT_GRAPH_FILTERS,
      2,
    );
    expect(nb.centerId).toBe("concept:self-knowledge");
    expect(nb.nodes.map((n) => n.node_id).sort()).toEqual([
      "concept:self",
      "concept:self-knowledge",
      "concept:shadow",
    ]);
    expect(nb.edges).toHaveLength(2);
  });

  it("trusted-only filtering drops unverified edges and orphaned nodes", () => {
    const nb = buildNeighborhood(OVERVIEW, "concept:self-knowledge", {
      ...DEFAULT_GRAPH_FILTERS,
      trustedOnly: true,
    });
    expect(nb.edges.map((e) => e.edge_id)).toEqual(["edge-t1"]);
    expect(nb.nodes.map((n) => n.node_id)).not.toContain("concept:self");
  });

  it("min-evidence filter excludes weakly supported relations", () => {
    const nb = buildNeighborhood(OVERVIEW, "concept:self-knowledge", {
      ...DEFAULT_GRAPH_FILTERS,
      minEvidence: 5,
    });
    expect(nb.edges.map((e) => e.edge_id)).toEqual(["edge-t1"]);
  });

  it("caps visible nodes and links at the performance limits", () => {
    const manyNodes = Array.from({ length: 300 }, (_, i) => ({
      node_id: `concept:n${i}`,
      canonical_name: `N${i}`,
      node_type: "CONCEPT",
      aliases: [],
      description: "",
      source_count: 1,
      evidence_count: i + 1,
    }));
    const center = manyNodes[299]; // strongest
    const spokes = manyNodes.slice(0, 250).map((n) => ({
      edge_id: `edge-${n.node_id}`,
      source_node_id: center.node_id,
      target_node_id: n.node_id,
      relationship_type: "RELATED_TO",
      confidence: 0.8,
      evidence_ids: [],
      evidence_count: 10,
      status: "TRUSTED" as const,
    }));
    const bigOverview = {
      ...OVERVIEW,
      nodes: [...manyNodes],
      edges: [...spokes],
    };
    const nb = buildNeighborhood(bigOverview, center.node_id,
      DEFAULT_GRAPH_FILTERS, 4);
    expect(nb.nodes.length).toBeLessThanOrEqual(MAX_VISIBLE_NODES);
    expect(nb.edges.length).toBeLessThanOrEqual(MAX_VISIBLE_LINKS);
    expect(nb.hiddenNodes).toBeGreaterThan(0);
    // the center always survives capping
    expect(nb.nodes.some((n) => n.node_id === center.node_id)).toBe(true);
  });

  it("respects relationship-type filters", () => {
    const nb = buildNeighborhood(OVERVIEW, "concept:shadow", {
      ...DEFAULT_GRAPH_FILTERS,
      relationTypes: ["RELATED_TO"],
    });
    expect(nb.edges.map((e) => e.edge_id)).toEqual(["edge-t1"]);
  });

  it("respects node-type filters when picking neighbors", () => {
    const nb = buildNeighborhood(OVERVIEW, "concept:shadow", {
      ...DEFAULT_GRAPH_FILTERS,
      nodeTypes: ["ARCHETYPE"],
    });
    // THEME neighbor (Self-knowledge) is excluded by the filter; the only
    // ARCHETYPE-to-ARCHETYPE edge runs through it, so no edges survive.
    expect(nb.nodes.map((n) => n.node_id)).toEqual(["concept:shadow"]);
    expect(nb.edges).toHaveLength(0);
  });
});

describe("toGraphData / visual semantics", () => {
  it("maps DTOs into renderer shapes with type colors and status widths", () => {
    // two rings so both fixture edges are inside the visible subgraph
    const nb = buildNeighborhood(OVERVIEW, "concept:shadow",
      DEFAULT_GRAPH_FILTERS, 2);
    const data = toGraphData(nb, null);
    const shadow = data.nodes.find((n) => n.id === "concept:shadow")!;
    expect(shadow.nodeType).toBe("ARCHETYPE");
    expect(shadow.color).toBe(nodeColor("ARCHETYPE"));
    expect(shadow.val).toBe(nodeVal(148));
    const trusted = data.links.find((l) => l.edgeId === "edge-t1")!;
    expect(trusted.status).toBe("TRUSTED");
    expect(trusted.width).toBeGreaterThan(
      data.links.find((l) => l.edgeId === "edge-u2")!.width,
    );
  });

  it("highlights only the selected edge", () => {
    const nb = buildNeighborhood(OVERVIEW, "concept:shadow",
      DEFAULT_GRAPH_FILTERS, 2);
    const data = toGraphData(nb, "edge-t1");
    expect(data.links.find((l) => l.edgeId === "edge-t1")!.color)
      .toBe("#f3ecdd");
    expect(data.links.find((l) => l.edgeId === "edge-u2")!.color)
      .not.toBe("#f3ecdd");
  });

  it("exposes distinct palettes per node type", () => {
    const types = ["ARCHETYPE", "CONCEPT", "THEME", "PERSON", "WORK",
      "SYMBOL", "PLACE"];
    const colors = types.map(nodeColor);
    expect(new Set(colors).size).toBe(types.length);
    expect(nodeColor("SOMETHING_ELSE")).toBe("#8b8778"); // safe fallback
  });

  it("lists available facet values deterministically", () => {
    expect(availableNodeTypes(NODES)).toEqual(["ARCHETYPE", "THEME"]);
    expect(availableRelationTypes(EDGES)).toEqual(["PART_OF", "RELATED_TO"]);
  });
});
