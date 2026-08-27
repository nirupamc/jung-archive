"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./AskLoader.module.css";

type Node = { x: number; y: number; vx: number; vy: number; r: number };

function makeNodes(seed = 42): Node[] {
  // deterministic pseudo-random 12 nodes
  let s = seed;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  const nodes: Node[] = [];
  for (let i = 0; i < 12; i++) {
    nodes.push({
      x: 10 + rnd() * 80,
      y: 15 + rnd() * 70,
      vx: (rnd() - 0.5) * 0.9,
      vy: (rnd() - 0.5) * 0.9,
      r: i === 0 ? 4.2 : 1.7 + rnd() * 1.6,
    });
  }
  return nodes;
}

export default function AskLoader({ label = "SEARCHING THE ARCHIVE" }: { label?: string }) {
  const [reduced, setReduced] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const nodes = useMemo<Node[]>(() => makeNodes(), []);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number>(0);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- sync with media query
    setReduced(m.matches);
    const handler = (e: MediaQueryListEvent) => setReduced(e.matches);
    m.addEventListener?.("change", handler);
    return () => m.removeEventListener?.("change", handler);
  }, []);

  useEffect(() => {
    if (reduced) return;
    if (typeof window === "undefined") return;
    const svg = svgRef.current;
    if (!svg) return;
    const circles = Array.from(svg.querySelectorAll<SVGCircleElement>("[data-node]"));
    const linesG = svg.querySelector<SVGGElement>("[data-lines]");

    const tick = (now: number) => {
      if (document.hidden) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      if (!startRef.current) startRef.current = now;
      const t = now - startRef.current;
      const dt = 16; // approx

      // gentle motion + center pull after 4s
      const pull = t > 4000 ? 0.00045 : 0;
      for (const n of nodes) {
        n.x += n.vx * dt * 0.045;
        n.y += n.vy * dt * 0.045;
        if (pull) {
          n.x += (50 - n.x) * pull * dt;
          n.y += (50 - n.y) * pull * dt * 0.9;
        }
        if (n.x < 6 || n.x > 94) n.vx *= -1;
        if (n.y < 6 || n.y > 94) n.vy *= -1;
        n.x = Math.max(6, Math.min(94, n.x));
        n.y = Math.max(6, Math.min(94, n.y));
      }

      // update circles
      const appear = Math.min(1, t / 1100);
      circles.forEach((c, i) => {
        const n = nodes[i];
        c.setAttribute("cx", String(n.x));
        c.setAttribute("cy", String(n.y));
        // central pulse after 5.5s
        let r = n.r;
        let op = 0.62 + (i === 0 ? 0.12 : 0);
        if (i === 0 && t > 5500) {
          r += Math.sin(t * 0.0016) * 0.9;
          op += Math.sin(t * 0.0016) * 0.08;
        }
        c.setAttribute("r", r.toFixed(2));
        c.style.opacity = String(op * appear);
        // fade weaker outer nodes after 2.5s (reranking metaphor) – keep 8 strongest
        if (t > 2500 && i > 7) {
          c.style.opacity = String(appear * 0.22);
        }
      });

      // update lines – nearest-neighbor threshold 30
      if (linesG) {
        linesG.innerHTML = "";
        const thr = t < 1800 ? 22 + (t / 1800) * 10 : 31;
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < thr) {
              const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
              line.setAttribute("x1", String(nodes[i].x));
              line.setAttribute("y1", String(nodes[i].y));
              line.setAttribute("x2", String(nodes[j].x));
              line.setAttribute("y2", String(nodes[j].y));
              const strength = (1 - d / thr) * 0.38 * appear;
              // faint amber
              line.setAttribute("stroke", `rgba(200,150,62,${strength.toFixed(3)})`);
              line.setAttribute("stroke-width", i === 0 || j === 0 ? "0.7" : "0.45");
              linesG.appendChild(line);
            }
          }
        }
      }

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [reduced, nodes]);

  if (reduced) {
    // static fallback – archival, no motion
    const staticNodes = nodes;
    return (
      <div className={styles.wrap} role="status" aria-live="polite" data-testid="ask-loader">
        <svg viewBox="0 0 100 100" className={styles.svg} aria-hidden="true">
          <g data-lines>
            {staticNodes.slice(0, 8).map((_, i) =>
              staticNodes.slice(i + 1, 8).map((__, j) => {
                const a = staticNodes[i];
                const b = staticNodes[i + 1 + j];
                const d = Math.hypot(a.x - b.x, a.y - b.y);
                if (d > 32) return null;
                return (
                  <line
                    key={`${i}-${j}`}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={`rgba(200,150,62,${(0.28 * (1 - d / 32)).toFixed(2)})`}
                    strokeWidth={0.6}
                  />
                );
              }),
            )}
          </g>
          {staticNodes.map((n, i) => (
            <circle key={i} cx={n.x} cy={n.y} r={n.r} className={styles.node} opacity={i > 7 ? 0.28 : 0.72} />
          ))}
        </svg>
        <div className={styles.label}>{label}</div>
        <div className={styles.hint}>synthesizing from indexed passages</div>
      </div>
    );
  }

  return (
    <div className={styles.wrap} role="status" aria-live="polite" aria-label={label} data-testid="ask-loader">
      <svg ref={svgRef} viewBox="0 0 100 100" className={styles.svg} aria-hidden="true">
        <g data-lines />
        {nodes.map((n, i) => (
          <circle
            key={i}
            data-node
            cx={n.x}
            cy={n.y}
            r={n.r}
            className={i === 0 ? styles.centerNode : styles.node}
          />
        ))}
      </svg>
      <div className={styles.label}>{label}</div>
      <div className={styles.hint}>synthesizing from indexed passages</div>
    </div>
  );
}
