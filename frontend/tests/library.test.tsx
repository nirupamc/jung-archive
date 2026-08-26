import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import Workspace from "@/components/Workspace";
import { DOC, UNPROCESSED_DOC } from "./fixtures";

function fetchJson(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status });
}

const DOCS = [DOC, UNPROCESSED_DOC];

describe("library (multi-document corpus view)", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn((url: string) => {
      if (url.includes("/pages/")) {
        const m = url.match(/\/api\/documents\/([^/]+)\/pages\//);
        if (m && m[1] === DOC.document_id) {
          return Promise.resolve(fetchJson({
            document_id: DOC.document_id,
            page_number: 1,
            width: 360,
            height: 576,
            classification: "NATIVE",
            classification_confidence: 0.9,
            classification_reason: null,
            layout: "SINGLE_COLUMN",
            layout_confidence: 0.8,
            layout_reason: null,
            ocr_confidence: null,
            warnings: [],
            blocks: [],
          }));
        }
        return Promise.resolve(fetchJson({ detail: "no canonical JSON" }, 404));
      }
      if (url.includes("/api/documents")) {
        return Promise.resolve(fetchJson(DOCS));
      }
      return Promise.resolve(fetchJson({}, 404));
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  it("lists PRIMARY and SECONDARY sections with every document visible", async () => {
    render(<Workspace />);
    await waitFor(() =>
      expect(screen.getByTestId(`doc-${DOC.document_id}`)).toBeInTheDocument(),
    );
    expect(screen.getByRole("heading", { name: /primary/i })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /secondary/i }),
    ).toBeInTheDocument();
    // unprocessed documents are NOT hidden
    expect(
      screen.getByTestId(`doc-${UNPROCESSED_DOC.document_id}`),
    ).toBeInTheDocument();
  });

  it("shows pipeline status badges per document", async () => {
    render(<Workspace />);
    await screen.findByTestId(`doc-${DOC.document_id}`);
    expect(
      screen.getByTestId(`doc-status-${DOC.document_id}`).textContent,
    ).toBe("INDEXED");
    expect(
      screen.getByTestId(`doc-status-${UNPROCESSED_DOC.document_id}`)
        .textContent,
    ).toBe("REVIEW");
  });

  it("filters by processing status", async () => {
    const user = userEvent.setup();
    render(<Workspace />);
    await screen.findByTestId(`doc-${DOC.document_id}`);
    await user.click(screen.getByTestId("status-filter-review"));
    expect(
      screen.queryByTestId(`doc-${DOC.document_id}`),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId(`doc-${UNPROCESSED_DOC.document_id}`),
    ).toBeInTheDocument();
    // reset
    await user.click(screen.getByTestId("status-filter-review"));
    expect(screen.getByTestId(`doc-${DOC.document_id}`)).toBeInTheDocument();
  });

  it("selecting an unprocessed document shows an honest status screen", async () => {
    const user = userEvent.setup();
    render(<Workspace />);
    await screen.findByTestId(`doc-${UNPROCESSED_DOC.document_id}`);
    await user.click(screen.getByTestId(`doc-${UNPROCESSED_DOC.document_id}`));
    const notice = await screen.findByTestId("unprocessed-notice");
    expect(notice.textContent).toContain("REVIEW");
    expect(notice.textContent).toContain("Held for review");
    // no crash + no page fetch loop against a missing artifact
    const pageCalls = fetchMock.mock.calls.filter((c) =>
      String(c[0]).includes("/pages/"),
    );
    expect(pageCalls.every((c) =>
      String(c[0]).includes(DOC.document_id),
    )).toBe(true);
  });

  it("topbar reflects the honest pipeline status of the selection", async () => {
    const user = userEvent.setup();
    render(<Workspace />);
    await screen.findByTestId(`doc-${UNPROCESSED_DOC.document_id}`);
    await user.click(screen.getByTestId(`doc-${UNPROCESSED_DOC.document_id}`));
    await waitFor(() => {
      expect(screen.getByTestId("topbar-status").textContent).toContain(
        "REVIEW",
      );
    });
  });
});
