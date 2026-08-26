"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as pdfjs from "pdfjs-dist";
import { computeScale, type FitMode } from "@/lib/viewerMath";

pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";

export type { FitMode };

interface PdfViewerProps {
  pdfUrl: string;
  pageNumber: number;
  zoom: number;
  fitMode: FitMode;
  /** page size in PDF points (from canonical IR) */
  pageWidthPt: number;
  pageHeightPt: number;
  children: (scale: number) => React.ReactNode;
}

export default function PdfViewer({
  pdfUrl,
  pageNumber,
  zoom,
  fitMode,
  pageWidthPt,
  pageHeightPt,
  children,
}: PdfViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });

  // Keyed load state: doc is valid only for the current URL.
  const [docState, setDocState] = useState<{
    url: string;
    doc?: pdfjs.PDFDocumentProxy;
    error?: string;
  } | null>(null);

  // Keyed render state: done only after the current render task finishes.
  const [renderState, setRenderState] = useState<{
    key: string;
    error?: string;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const task = pdfjs.getDocument({ url: pdfUrl });
    task.promise
      .then((doc) => !cancelled && setDocState({ url: pdfUrl, doc }))
      .catch((e: Error) =>
        !cancelled && setDocState({ url: pdfUrl, error: `PDF load failed: ${e.message}` }),
      );
    return () => {
      cancelled = true;
      task.destroy();
    };
  }, [pdfUrl]);

  // Track BOTH axes of the viewer pane; every resize recomputes fit.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () =>
      setBox((prev) => {
        const w = el.clientWidth;
        const h = el.clientHeight;
        if (prev.w === w && prev.h === h) return prev;
        return { w, h };
      });
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const scale = computeScale(
    fitMode,
    zoom,
    pageWidthPt,
    pageHeightPt,
    box.w || 800,
    box.h || 600,
  );

  const viewportW = Math.round(pageWidthPt * scale);
  const viewportH = Math.round(pageHeightPt * scale);
  const doc = docState && docState.url === pdfUrl ? (docState.doc ?? null) : null;
  const loadError =
    docState && docState.url === pdfUrl ? (docState.error ?? null) : null;

  const renderKey = `${pdfUrl}#${pageNumber}@${scale.toFixed(4)}`;
  const rendering =
    !loadError && !!doc && (!renderState || renderState.key !== renderKey);
  const renderError =
    renderState && renderState.key === renderKey
      ? (renderState.error ?? null)
      : null;

  // Render the requested page.
  useEffect(() => {
    if (!doc || Number.isNaN(scale) || viewportW <= 0) return;
    let cancelled = false;
    let task: { cancel: () => void } | null = null;
    doc
      .getPage(pageNumber)
      .then((p) => {
        if (cancelled) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const t = p.render({
          canvasContext: canvas.getContext("2d")!,
          viewport: p.getViewport({ scale }),
        });
        task = t;
        return t.promise;
      })
      .then(() => !cancelled && setRenderState({ key: renderKey }))
      .catch((e: Error) => {
        if (!cancelled && e?.name !== "RenderingCancelledException") {
          setRenderState({
            key: renderKey,
            error: `page render failed: ${e.message}`,
          });
        }
      });
    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [doc, pageNumber, scale, renderKey, viewportW]);

  return (
    <div ref={containerRef} className="pdfViewerScroll">
      {(loadError || renderError) && (
        <div role="alert" className="viewerError">
          {loadError ?? renderError}
        </div>
      )}
      {!doc && !loadError && (
        <div className="viewerLoading" role="status">loading PDF …</div>
      )}
      <div
        className="pdfStage"
        style={{ width: viewportW, height: viewportH }}
        data-testid="pdf-stage"
        data-scale={Number.isNaN(scale) ? "" : scale.toFixed(4)}
        data-rendering={rendering ? "true" : "false"}
      >
        <canvas
          ref={canvasRef}
          width={viewportW}
          height={viewportH}
          aria-label={`PDF page ${pageNumber} rendering`}
        />
        {doc && !loadError && children(scale)}
      </div>
      {rendering && <span className="sr-only">rendering</span>}
    </div>
  );
}
