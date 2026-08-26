"use client";

import { useRef } from "react";
import { useWorkspace, type Tab } from "@/state/workspace";
import styles from "./TabBar.module.css";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "document", label: "DOCUMENT" },
  { id: "structure", label: "STRUCTURE" },
  { id: "chunks", label: "CHUNKS" },
  { id: "retrieval", label: "RETRIEVAL" },
  { id: "evaluation", label: "EVALUATION" },
  { id: "graph", label: "GRAPH" },
];

export default function TabBar() {
  const { state, dispatch } = useWorkspace();
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const idx = TABS.findIndex((t) => t.id === state.tab);
    let next = idx;
    if (e.key === "ArrowRight") next = (idx + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (idx - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    dispatch({ type: "set_tab", tab: TABS[next].id });
    refs.current[next]?.focus();
  };

  return (
    <nav className={styles.wrap} aria-label="inspector views">
      <div
        role="tablist"
        className={styles.tablist}
        onKeyDown={onKeyDown}
      >
        {TABS.map((t, i) => (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={state.tab === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={state.tab === t.id ? 0 : -1}
            data-testid={`tab-${t.id}`}
            className={[styles.tab, state.tab === t.id ? styles.active : ""].join(
              " ",
            )}
            onClick={() => dispatch({ type: "set_tab", tab: t.id })}
          >
            {t.label}
          </button>
        ))}
      </div>
      <span className={styles.meta}>
        {state.documentId
          ? `${state.documentId} · p.${state.page}`
          : "no document"}
      </span>
    </nav>
  );
}
