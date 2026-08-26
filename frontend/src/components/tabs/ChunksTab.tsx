"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { ChunkOut } from "@/lib/types";
import { useWorkspace } from "@/state/workspace";
import styles from "./ChunksTab.module.css";

export default function ChunksTab() {
  const { state, traceToSource, dispatch } = useWorkspace();
  const [pageFilter, setPageFilter] = useState("");
  const [sectionFilter, setSectionFilter] = useState("");
  const docId = state.documentId;

  const [loaded, setLoaded] = useState<{
    docId: string;
    chunks?: ChunkOut[];
    error?: string;
  } | null>(null);

  useEffect(() => {
    if (!docId) return;
    let cancelled = false;
    api
      .chunks(docId)
      .then((chunks) => !cancelled && setLoaded({ docId, chunks }))
      .catch((e: Error) => !cancelled && setLoaded({ docId, error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [docId]);

  const chunks =
    loaded && loaded.docId === docId ? (loaded.chunks ?? null) : null;
  const error = loaded && loaded.docId === docId ? (loaded.error ?? null) : null;

  const filtered = useMemo(() => {
    if (!chunks) return [];
    let out = chunks;
    const pf = Number(pageFilter);
    if (Number.isFinite(pf) && pageFilter !== "") {
      out = out.filter((c) => c.page_numbers.includes(pf));
    }
    if (sectionFilter.trim()) {
      const sf = sectionFilter.toLowerCase();
      out = out.filter((c) =>
        c.heading_path.join(" > ").toLowerCase().includes(sf),
      );
    }
    return out;
  }, [chunks, pageFilter, sectionFilter]);

  if (error)
    return <div role="alert" className={styles.error}>chunks unavailable: {error}</div>;
  if (!chunks)
    return <div role="status" className={styles.loading}>loading chunks …</div>;

  const selected =
    chunks.find((c) => c.chunk_id === state.selectedChunkId) ?? null;
  const idx = filtered.findIndex(
    (c) => c.chunk_id === state.selectedChunkId,
  );

  const step = (delta: number) => {
    const next = filtered[idx + delta];
    if (next) dispatch({ type: "select_chunk", id: next.chunk_id });
  };

  return (
    <div className={styles.split}>
      <div className={styles.listPane}>
        <div className={styles.controls}>
          <label htmlFor="chunk-page">page</label>
          <input
            id="chunk-page"
            value={pageFilter}
            onChange={(e) => setPageFilter(e.target.value)}
            placeholder="all"
            inputMode="numeric"
          />
          <label htmlFor="chunk-section">section</label>
          <input
            id="chunk-section"
            value={sectionFilter}
            onChange={(e) => setSectionFilter(e.target.value)}
            placeholder="filter heading path…"
          />
          <span>{filtered.length} chunks</span>
        </div>
        <ul className={styles.list} aria-label="chunks">
          {filtered.map((c) => (
            <li key={c.chunk_id}>
              <button
                type="button"
                data-testid={`chunk-${c.chunk_id}`}
                aria-pressed={state.selectedChunkId === c.chunk_id}
                className={[
                  styles.row,
                  state.selectedChunkId === c.chunk_id ? styles.selected : "",
                ].join(" ")}
                onClick={() =>
                  dispatch({ type: "select_chunk", id: c.chunk_id })
                }
              >
                <code className={styles.cid}>{c.chunk_id.slice(-7)}</code>
                <span className={styles.pages}>
                  p.{Math.min(...c.page_numbers)}–{Math.max(...c.page_numbers)}
                </span>
                <span className={styles.tokens}>{c.token_count} tok</span>
                <span className={styles.head}>
                  {c.heading_path.join(" › ") || "—"}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className={styles.detail} aria-live="polite">
        {!selected && <p className={styles.hint}>select a chunk to inspect provenance</p>}
        {selected && (
          <>
            <h3 className={styles.detailTitle}>
              <code>{selected.chunk_id}</code>
            </h3>
            <dl className={styles.meta}>
              <dt>heading path</dt>
              <dd>{selected.heading_path.join(" › ") || "—"}</dd>
              <dt>pages</dt>
              <dd>{selected.page_numbers.join(", ")}</dd>
              <dt>tokens</dt>
              <dd>{selected.token_count}</dd>
              <dt>source type</dt>
              <dd>{selected.source_type}</dd>
              <dt>strategy</dt>
              <dd>{selected.strategy ?? "—"}</dd>
              <dt>section</dt>
              <dd>{selected.section_id ?? "—"}</dd>
            </dl>
            <button
              type="button"
              data-testid="trace-chunk-source"
              className={styles.traceBtn}
              onClick={() =>
                traceToSource({
                  pageNumbers: selected.page_numbers,
                  blockIds: selected.source_block_ids,
                  chunkId: selected.chunk_id,
                })
              }
            >
              trace → source blocks & page
            </button>
            <p className={styles.blockIds}>
              source blocks:{" "}
              {selected.source_block_ids.map((b) => (
                <code key={b}>{b}</code>
              ))}
            </p>
            <pre className={styles.text}>{selected.text}</pre>
            <div className={styles.pager}>
              <button
                type="button"
                onClick={() => step(-1)}
                disabled={idx <= 0}
              >
                ‹ previous chunk
              </button>
              <button
                type="button"
                onClick={() => step(1)}
                disabled={idx < 0 || idx >= filtered.length - 1}
              >
                next chunk ›
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
