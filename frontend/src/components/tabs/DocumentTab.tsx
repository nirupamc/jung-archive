"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { PageInspection } from "@/lib/types";
import { useWorkspace } from "@/state/workspace";
import PdfViewer, { type FitMode } from "../PdfViewer";
import BlockOverlays from "../BlockOverlays";
import styles from "./DocumentTab.module.css";

export default function DocumentTab() {
  const { state, dispatch, setPage } = useWorkspace();
  const [zoom, setZoom] = useState(1.4);
  const [fitMode, setFitMode] = useState<FitMode>("width");
  const [showOverlays, setShowOverlays] = useState(true);
  const docId = state.documentId;
  // PDF load failures are tracked per document id, so switching to another
  // document does not inherit a previous document's failure (and returning to
  // the failed document still shows its fallback).
  const [pdfFailedDocIds, setPdfFailedDocIds] = useState<ReadonlySet<string>>(
    () => new Set<string>(),
  );

  const [loaded, setLoaded] = useState<{
    key: string;
    data?: PageInspection;
    error?: string;
  } | null>(null);

  useEffect(() => {
    if (!docId) return;
    let cancelled = false;
    api
      .page(docId, state.page)
      .then((data) => !cancelled && setLoaded({ key: `${docId}:${state.page}`, data }))
      .catch((e: Error) =>
        !cancelled && setLoaded({ key: `${docId}:${state.page}`, error: e.message }),
      );
    return () => {
      cancelled = true;
    };
  }, [docId, state.page]);

  const pageData =
    loaded && loaded.key === `${docId}:${state.page}` ? (loaded.data ?? null) : null;
  const pageError =
    loaded && loaded.key === `${docId}:${state.page}` ? (loaded.error ?? null) : null;
  const pageLoading =
    docId !== null && (loaded === null || loaded.key !== `${docId}:${state.page}`);

  if (!docId) {
    return <div className={styles.placeholder}>no document selected</div>;
  }

  const selectedDocument = state.documents?.find((d) => d.document_id === docId);
  const pageCount = selectedDocument?.page_count ?? 1;
  const hasPdf = selectedDocument?.has_pdf ?? false;
  const pdfFailed = pdfFailedDocIds.has(docId);
  const selectedSet = new Set(state.selectedBlockIds);
  const pdfUrl = api.pdfUrl(docId);

  return (
    <div className={styles.wrap}>
      <div className={styles.toolbar} role="toolbar" aria-label="page navigation">
        <button
          type="button"
          onClick={() => setPage(state.page - 1)}
          disabled={state.page <= 1}
          aria-label="previous page"
        >
          ‹ prev
        </button>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const input = new FormData(e.currentTarget).get("page");
            const n = Number(input);
            if (Number.isFinite(n)) setPage(Math.min(Math.max(1, Math.round(n)), pageCount));
          }}
        >
          <label htmlFor="page-input">page</label>
          <input
            id="page-input"
            name="page"
            type="number"
            min={1}
            max={pageCount}
            defaultValue={undefined}
            key={state.page}
            placeholder={String(state.page)}
            className={styles.pageInput}
          />
          <span aria-live="polite" className={styles.pageLabel}>
            {state.page} / {pageCount}
          </span>
        </form>
        <button
          type="button"
          onClick={() => setPage(state.page + 1)}
          disabled={state.page >= pageCount}
          aria-label="next page"
        >
          next ›
        </button>
        <span className={styles.divider} />
        {(["width", "page"] as FitMode[]).map((m) => (
          <button
            key={m}
            type="button"
            aria-pressed={fitMode === m}
            className={fitMode === m ? styles.active : ""}
            onClick={() => setFitMode(m)}
          >
            fit {m}
          </button>
        ))}
        <span className={styles.divider} />
        <button
          type="button"
          aria-label="zoom out"
          onClick={() => {
            setFitMode("custom");
            setZoom((z) => Math.max(0.4, +(z - 0.2).toFixed(2)));
          }}
        >
          −
        </button>
        <span className={styles.zoomLabel}>
          {(fitMode === "custom" ? zoom : 1).toFixed(1)}×
        </span>
        <button
          type="button"
          aria-label="zoom in"
          onClick={() => {
            setFitMode("custom");
            setZoom((z) => Math.min(4, +(z + 0.2).toFixed(2)));
          }}
        >
          +
        </button>
        <span className={styles.divider} />
        <span className={styles.overlayLegend}>
          overlays:
          <label>
            <input
              type="checkbox"
              checked={showOverlays}
              onChange={(e) => setShowOverlays(e.target.checked)}
            />{" "}
            blocks
          </label>
          {selectedSet.size > 0 && (
            <em className={styles.traced}>{selectedSet.size} traced</em>
          )}
        </span>
      </div>

      {pageError && (
        <div role="alert" className={styles.error}>
          failed to load page data: {pageError}
        </div>
      )}
      {!pageError && pageLoading && (
        <div role="status" className={styles.pageLoading}>
          loading page data …
        </div>
      )}

      <div className={styles.viewerArea}>
        {hasPdf ? (
          pdfFailed ? (
            <div className={styles.pdfFallback}>
              Original source PDF is not available in this deployment. Extracted document content and analysis remain available.
            </div>
          ) : (
            <PdfViewer
              pdfUrl={pdfUrl}
              pageNumber={state.page}
              zoom={zoom}
              fitMode={fitMode}
              pageWidthPt={pageData?.width ?? 360}
              pageHeightPt={pageData?.height ?? 576}
              onLoadError={() =>
                setPdfFailedDocIds((prev) => {
                  if (prev.has(docId)) return prev;
                  const next = new Set(prev);
                  next.add(docId);
                  return next;
                })
              }
            >
              {(scale) =>
                showOverlays && pageData ? (
                  <BlockOverlays
                    blocks={pageData.blocks}
                    scale={scale}
                    selectedIds={state.selectedBlockIds}
                    onSelect={(b) =>
                      dispatch({
                        type: "select_blocks",
                        ids: selectedSet.has(b.block_id)
                          ? state.selectedBlockIds.filter((x) => x !== b.block_id)
                          : [...state.selectedBlockIds, b.block_id],
                      })
                    }
                  />
                ) : null
              }
            </PdfViewer>
          )
        ) : (
          <div className={styles.pdfFallback}>
            Original source PDF is not available in this deployment. Extracted document content and analysis remain available.
          </div>
        )}
      </div>
    </div>
  );
}
