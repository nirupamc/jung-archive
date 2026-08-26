"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { isProcessedDoc } from "@/lib/types";
import type { BlockOut, DocumentSummary, PageInspection } from "@/lib/types";
import { useWorkspace } from "@/state/workspace";
import styles from "./InspectorPanel.module.css";

function orDash(v: unknown): string {
  if (v === null || v === undefined || v === "") return "—";
  return String(v);
}

export default function InspectorPanel() {
  const { state } = useWorkspace();
  const docId = state.documentId;
  const key = docId ? `${docId}:${state.page}` : "";

  const [loaded, setLoaded] = useState<{
    key: string;
    page?: PageInspection;
    error?: string;
  } | null>(null);

  const selectedDoc: DocumentSummary | null =
    state.documents?.find((d) => d.document_id === docId) ?? null;
  const processable = isProcessedDoc(selectedDoc);

  useEffect(() => {
    if (!docId || !processable) return;
    let cancelled = false;
    api
      .page(docId, state.page)
      .then((page) => !cancelled && setLoaded({ key, page }))
      .catch((e: Error) => !cancelled && setLoaded({ key, error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [docId, state.page, key, processable]);

  const current = loaded && loaded.key === key ? loaded : null;
  const page: PageInspection | null = current?.page ?? null;
  const error = current?.error ?? null;

  const selectedBlocks: BlockOut[] =
    (page?.blocks ?? []).filter((b) =>
      state.selectedBlockIds.includes(b.block_id),
    );

  return (
    <aside className={styles.wrap} aria-label="inspector">
      <h2 className={styles.heading}>
        inspector
        <span className={styles.pageTag}>PAGE {page?.page_number ?? state.page}</span>
      </h2>

      {!processable && (
        <div role="note" className={styles.loading}>
          {selectedDoc
            ? `${selectedDoc.status}: no processed pages to inspect yet`
            : "no document selected"}
        </div>
      )}

      {processable && error && (
        <div role="alert" className={styles.error}>page data unavailable: {error}</div>
      )}
      {processable && !error && !page && (
        <div role="status" className={styles.loading}>loading page …</div>
      )}

      {page && (
        <>
          <section className={styles.section} aria-label="classification">
            <h3>Classification</h3>
            <dl className={styles.meta}>
              <dt>class</dt>
              <dd>
                <em className={styles.classBadge}>{page.classification}</em>{" "}
                {page.classification_confidence !== null && (
                  <span className={styles.conf}>
                    {page.classification_confidence}
                  </span>
                )}
              </dd>
              <dt>reason</dt>
              <dd>{orDash(page.classification_reason)}</dd>
              <dt>layout</dt>
              <dd>
                {page.layout}{" "}
                {page.layout_confidence !== null && (
                  <span className={styles.conf}>{page.layout_confidence}</span>
                )}
              </dd>
              <dt>layout reason</dt>
              <dd>{orDash(page.layout_reason)}</dd>
              <dt>ocr confidence</dt>
              <dd data-testid="ocr-confidence">
                {page.ocr_confidence ?? "not measured"}
              </dd>
              <dt>blocks</dt>
              <dd>{page.blocks.length}</dd>
            </dl>
            {page.warnings.length > 0 && (
              <ul className={styles.warnings} aria-label="warnings">
                {page.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            )}
          </section>

          <section className={styles.section} aria-label="blocks">
            <h3>Blocks</h3>
            <ol className={styles.blockList}>
              {page.blocks.map((b) => (
                <li key={b.block_id}>
                  <details
                    open={state.selectedBlockIds.includes(b.block_id)}
                    data-testid={`inspector-${b.block_id}`}
                  >
                    <summary>
                      <code>{b.block_id}</code>
                      <em className={styles.typeBadge}>{b.block_type}</em>
                      <span className={styles.order}>ro {b.reading_order}</span>
                    </summary>
                    <dl className={styles.meta}>
                      <dt>extraction</dt>
                      <dd>{b.extraction_method}</dd>
                      <dt>confidence</dt>
                      <dd data-testid="block-confidence">
                        {b.confidence ?? "not measured"}
                      </dd>
                      <dt>heuristic quality</dt>
                      <dd>{b.heuristic_quality_score ?? "—"}</dd>
                      <dt>bbox</dt>
                      <dd className={styles.mono}>
                        [{b.bbox.x0.toFixed(1)}, {b.bbox.y0.toFixed(1)},{" "}
                        {b.bbox.x1.toFixed(1)}, {b.bbox.y1.toFixed(1)}]
                      </dd>
                      <dt>font</dt>
                      <dd>{orDash(b.font_name)} @ {orDash(b.font_size)}</dd>
                    </dl>
                    <p className={styles.blockText}>{b.text.slice(0, 400)}</p>
                  </details>
                </li>
              ))}
            </ol>
          </section>

          {selectedBlocks.length > 0 && (
            <section
              className={styles.section}
              aria-label="selected provenance"
              data-testid="selected-provenance"
            >
              <h3>Selected ({selectedBlocks.length})</h3>
              <ul className={styles.selectedList}>
                {selectedBlocks.map((b) => (
                  <li key={b.block_id}>
                    <code>{b.block_id}</code> — {b.block_type} — p.
                    {b.page_number}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </aside>
  );
}
