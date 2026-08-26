"use client";

/**
 * Interactive 3D force view over the evidence-backed knowledge graph,
 * rendered with react-force-graph-3d.
 *
 * Loading discipline:
 *   - this module is itself lazily imported (next/dynamic) when the GRAPH
 *     tab opens;
 *   - react-force-graph-3d / Three.js are additionally imported at runtime
 *     below, and ONLY after a WebGL capability check succeeds, so browsers
 *     without WebGL never download or initialize the 3D stack.
 *
 * Performance safeguards:
 *   - capped subgraph produced upstream (lib/graph3d caps)
 *   - physics stops automatically after cooldown (cooldownTicks)
 *   - device pixel ratio clamped to 1.5
 *   - labels render as tooltips on hover only (never permanent overlays)
 *   - no post-processing pipeline (no bloom)
 */

import {
  Component,
  useEffect,
  useRef,
  useState,
  type ComponentType,
  type ReactNode,
} from "react";
import type { ForceGraphMethods } from "react-force-graph-3d";
import type { ForceLink, ForceNode } from "@/lib/graph3d";
import styles from "./ForceGraphView.module.css";

type FGComponent = ComponentType<{
  ref?: React.Ref<ForceGraphMethods<ForceNode, ForceLink>>;
  graphData: { nodes: ForceNode[]; links: ForceLink[] };
  width?: number;
  height?: number;
  backgroundColor?: string;
  showNavInfo?: boolean;
  nodeRelSize?: number;
  nodeVal?: (n: ForceNode) => number;
  nodeColor?: (n: ForceNode) => string;
  nodeLabel?: (n: ForceNode) => string;
  nodeResolution?: number;
  linkColor?: (l: ForceLink) => string;
  linkWidth?: (l: ForceLink) => number;
  linkOpacity?: number;
  linkLabel?: (l: ForceLink) => string;
  onNodeClick?: (node: ForceNode) => void;
  onLinkClick?: (link: ForceLink) => void;
  cooldownTicks?: number;
  cooldownTime?: number;
  warmupTicks?: number;
  enableNodeDrag?: boolean;
  onEngineStop?: () => void;
}>;

interface ForceGraphViewProps {
  data: { nodes: ForceNode[]; links: ForceLink[] };
  selectedNodeId: string | null;
  onNodeClick: (nodeId: string) => void;
  onLinkClick: (edgeId: string) => void;
}

function hasWebGL(): boolean {
  try {
    if (typeof document === "undefined") return false;
    const canvas = document.createElement("canvas");
    return !!(
      canvas.getContext("webgl2") ??
      canvas.getContext("webgl") ??
      canvas.getContext("experimental-webgl")
    );
  } catch {
    return false;
  }
}

export default function ForceGraphView({
  data,
  selectedNodeId,
  onNodeClick,
  onLinkClick,
}: ForceGraphViewProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraphMethods<ForceNode, ForceLink> | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [FG, setFG] = useState<FGComponent | null>(null);
  // Client-only component (ssr:false), so the capability check is safe in
  // the initializer: browsers without WebGL never load the 3D stack.
  const [unsupported] = useState(
    () => typeof window !== "undefined" && !hasWebGL(),
  );

  // Track the pane so the canvas always fills available space.
  // ResizeObserver fires on observe(), which provides the first measurement.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() =>
      setSize({ w: el.clientWidth, h: el.clientHeight }),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Runtime import second: capability check already passed here.
  useEffect(() => {
    if (unsupported) return;
    let cancelled = false;
    import("react-force-graph-3d")
      .then((m) => {
        if (!cancelled) setFG(m.default as unknown as FGComponent);
      })
      .catch(() => {
        /* import failure handled by boundary/fallback */
      });
    return () => {
      cancelled = true;
    };
  }, [unsupported]);

  // Clamp pixel ratio once the renderer exists.
  useEffect(() => {
    if (!FG) return;
    try {
      const renderer = fgRef.current?.renderer?.();
      renderer?.setPixelRatio?.(
        Math.min(window.devicePixelRatio || 1, 1.5),
      );
    } catch {
      /* renderer not ready yet; non-fatal */
    }
  }, [FG]);

  if (unsupported) {
    return (
      <div className={styles.fallback} role="note">
        <p>
          3D graph unavailable: this browser/device cannot initialize WebGL.
        </p>
        <p>
          Use the concept list and relationship list beside this pane to
          inspect the same evidence-backed data.
        </p>
      </div>
    );
  }

  return (
    <div ref={wrapRef} className={styles.wrap} data-testid="force-graph-wrap">
      {size.w > 0 && FG && (
        <FG
          ref={fgRef}
          graphData={data}
          width={size.w}
          height={size.h}
          backgroundColor="rgba(0,0,0,0)"
          showNavInfo={false}
          nodeRelSize={4}
          nodeVal={(n: ForceNode) => n.val}
          nodeColor={(n: ForceNode) =>
            n.isCenter || n.id === selectedNodeId ? "#f3ecdd" : n.color
          }
          nodeLabel={(n: ForceNode) =>
            `${n.name} · ${n.nodeType} · ${n.evidenceCount} evidence chunks`
          }
          nodeResolution={6}
          linkColor={(l: ForceLink) => l.color}
          linkWidth={(l: ForceLink) => l.width}
          linkOpacity={0.55}
          linkLabel={(l: ForceLink) =>
            `${l.relationType.replace(/_/g, " ")} (${l.status.toLowerCase()}) · ${l.evidenceCount} evidence`
          }
          onNodeClick={(node: ForceNode) => onNodeClick(node.id)}
          onLinkClick={(link: ForceLink) => onLinkClick(link.edgeId)}
          cooldownTicks={120}
          cooldownTime={3000}
          warmupTicks={0}
          enableNodeDrag={true}
          onEngineStop={() => {
            try {
              fgRef.current?.zoomToFit(400, 48);
            } catch {
              /* camera fit is best-effort */
            }
          }}
        />
      )}
      {!FG && (
        <div className={styles.loading} role="status">
          preparing 3D graph …
        </div>
      )}
    </div>
  );
}

/** Catches WebGL/driver crashes so the rest of the workspace survives. */
export class ForceGraphBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <div className={styles.fallback} role="note">
          <p>3D graph crashed while rendering; the concept lists below
            remain usable.</p>
        </div>
      );
    }
    return this.props.children;
  }
}
