import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkspaceProvider, useWorkspace } from "@/state/workspace";
import DocumentTab from "@/components/tabs/DocumentTab";

/**
 * The real PdfViewer talks to pdf.js, which cannot load a document in jsdom.
 * It is replaced by a stub so these tests assert DocumentTab's PDF gating
 * (has_pdf, load-failure fallback) rather than pdf.js rendering.
 */
vi.mock("@/components/PdfViewer", () => ({
  default: ({
    pdfUrl,
    onLoadError,
  }: {
    pdfUrl: string;
    onLoadError?: (error: Error) => void;
  }) => (
    <div data-testid="pdf-viewer" data-url={pdfUrl}>
      <button type="button" onClick={() => onLoadError?.(new Error("boom"))}>
        fail pdf load
      </button>
    </div>
  ),
}));

const FALLBACK_TEXT = /original source pdf is not available/i;

function makeDoc(documentId: string, title: string, hasPdf: boolean) {
  return {
    document_id: documentId,
    title,
    author: null,
    source_type: "PRIMARY",
    index_status: "INCLUDE",
    page_count: 1,
    chunk_count: 1,
    source_path: `primary/${documentId}.pdf`,
    has_pdf: hasPdf,
    status: "INDEXED",
    section: "PRIMARY",
    registered: true,
    sha256: null,
  };
}

const PAGE = {
  document_id: "any",
  page_number: 1,
  width: 360,
  height: 576,
  classification: "NATIVE",
  classification_confidence: 0.9,
  classification_reason: "ok",
  layout: "SINGLE_COLUMN",
  layout_confidence: 0.9,
  layout_reason: "ok",
  ocr_confidence: null,
  warnings: [],
  blocks: [],
};

function json(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    }),
  );
}

function installFetch(docs: unknown[]) {
  const mock = vi.fn((url: string) => {
    if (url.includes("/pages/")) return json({ ...PAGE });
    if (url.includes("/chunks")) return json([]);
    if (url.endsWith("/pdf")) return json({}, 404);
    if (url.includes("/api/documents")) return json(docs);
    return json({}, 404);
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

/** Selects a document through the workspace API, like the sidebar does. */
function DocumentSwitcher() {
  const { state, selectDocument } = useWorkspace();
  return (
    <div>
      {(state.documents ?? []).map((d) => (
        <button
          key={d.document_id}
          type="button"
          onClick={() => selectDocument(d.document_id)}
        >
          select {d.document_id}
        </button>
      ))}
    </div>
  );
}

function renderTab() {
  return render(
    <WorkspaceProvider>
      <DocumentSwitcher />
      <DocumentTab />
    </WorkspaceProvider>,
  );
}

describe("DocumentTab PDF gating", () => {
  beforeEach(() => {
    // Silence expected workspace state updates during load assertions.
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("shows the fallback and never mounts the viewer when has_pdf is false", async () => {
    installFetch([makeDoc("no-pdf", "No Pdf Document", false)]);
    renderTab();

    await waitFor(() => expect(screen.getByText(FALLBACK_TEXT)).toBeTruthy());
    expect(screen.queryByTestId("pdf-viewer")).toBeNull();
  });

  it("mounts the viewer and hides the fallback when has_pdf is true", async () => {
    installFetch([makeDoc("has-pdf", "Has Pdf Document", true)]);
    renderTab();

    await waitFor(() => expect(screen.getByTestId("pdf-viewer")).toBeTruthy());
    expect(screen.getByTestId("pdf-viewer").getAttribute("data-url")).toContain(
      "/api/documents/has-pdf/pdf",
    );
    expect(screen.queryByText(FALLBACK_TEXT)).toBeNull();
  });

  it("shows the fallback after a PDF load failure", async () => {
    installFetch([makeDoc("broken", "Broken Pdf Document", true)]);
    renderTab();

    await screen.findByTestId("pdf-viewer");
    await userEvent.click(
      screen.getByRole("button", { name: /fail pdf load/i }),
    );

    await waitFor(() => expect(screen.getByText(FALLBACK_TEXT)).toBeTruthy());
    expect(screen.queryByTestId("pdf-viewer")).toBeNull();
  });

  it("keeps a PDF load failure scoped to the document that failed", async () => {
    installFetch([
      makeDoc("doc-a", "Doc A", true),
      makeDoc("doc-b", "Doc B", true),
    ]);
    renderTab();

    // A is preferred on mount; make its PDF load fail.
    await screen.findByTestId("pdf-viewer");
    await userEvent.click(
      screen.getByRole("button", { name: /fail pdf load/i }),
    );
    await waitFor(() => expect(screen.getByText(FALLBACK_TEXT)).toBeTruthy());

    // B has a valid PDF: the viewer must mount, not inherit A's failure.
    await userEvent.click(screen.getByRole("button", { name: /select doc-b/i }));
    await waitFor(() => expect(screen.getByTestId("pdf-viewer")).toBeTruthy());
    expect(screen.getByTestId("pdf-viewer").getAttribute("data-url")).toContain(
      "/api/documents/doc-b/pdf",
    );
    expect(screen.queryByText(FALLBACK_TEXT)).toBeNull();

    // Returning to A still shows A's own failure.
    await userEvent.click(screen.getByRole("button", { name: /select doc-a/i }));
    await waitFor(() => expect(screen.getByText(FALLBACK_TEXT)).toBeTruthy());
    expect(screen.queryByTestId("pdf-viewer")).toBeNull();
  });

  it("swaps between fallback and viewer as the selected document changes", async () => {
    installFetch([
      makeDoc("no-pdf", "No Pdf Document", false),
      makeDoc("has-pdf", "Has Pdf Document", true),
    ]);
    renderTab();

    // Preferred document on mount is the first one (no PDF).
    await waitFor(() => expect(screen.getByText(FALLBACK_TEXT)).toBeTruthy());
    expect(screen.queryByTestId("pdf-viewer")).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /select has-pdf/i }));
    await waitFor(() => expect(screen.getByTestId("pdf-viewer")).toBeTruthy());
    expect(screen.queryByText(FALLBACK_TEXT)).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /select no-pdf/i }));
    await waitFor(() => expect(screen.getByText(FALLBACK_TEXT)).toBeTruthy());
    expect(screen.queryByTestId("pdf-viewer")).toBeNull();
  });

  it("keeps page navigation and structure available without a PDF", async () => {
    installFetch([makeDoc("no-pdf", "No Pdf Document", false)]);
    renderTab();

    await waitFor(() => expect(screen.getByText(FALLBACK_TEXT)).toBeTruthy());
    expect(screen.getByRole("button", { name: /previous page/i })).toBeDisabled();
    expect(screen.getByLabelText(/^page$/i)).toBeTruthy();
    expect(screen.getByRole("checkbox", { name: /blocks/i })).toBeTruthy();
  });
});
