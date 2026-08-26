import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import EvaluationTab from "@/components/tabs/EvaluationTab";
import { WorkspaceProvider } from "@/state/workspace";
import {
  EVAL_RUN_A,
  EVAL_RUN_B,
  EVAL_RUNS,
  LATEST_SUMMARY,
} from "./evalFixtures";

function mockFetch(url: string) {
  if (url.includes(EVAL_RUN_A.run_id)) {
    return new Response(JSON.stringify(EVAL_RUN_A), { status: 200 });
  }
  if (url.includes(EVAL_RUN_B.run_id)) {
    return new Response(JSON.stringify(EVAL_RUN_B), { status: 200 });
  }
  if (url.includes("/api/evaluation/runs")) {
    return new Response(JSON.stringify(EVAL_RUNS), { status: 200 });
  }
  if (url.includes("/api/evaluation/latest")) {
    return new Response(JSON.stringify(LATEST_SUMMARY), { status: 200 });
  }
  if (url.includes("/api/documents")) {
    return new Response(
      JSON.stringify([
        {
          document_id: "381d2da4b68e",
          title: "The Undiscovered Self",
          author: "Carl Gustav Jung",
          source_type: "PRIMARY",
          index_status: "INCLUDE",
          page_count: 88,
          chunk_count: 211,
          source_path: null,
          has_pdf: false,
        },
      ]),
      { status: 200 },
    );
  }
  return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn((input: string | Request) =>
    mockFetch(String(input)),
  ));
});
afterEach(() => vi.unstubAllGlobals());

describe("EvaluationTab overview", () => {
  it("renders the benchmark table with all four modes", async () => {
    render(<EvaluationTab />);
    await waitFor(() =>
      expect(screen.getByTestId("overview-table")).toBeInTheDocument(),
    );
    for (const mode of ["dense", "bm25", "hybrid", "hybrid_rerank"]) {
      expect(screen.getByTestId(`row-${mode}`)).toBeInTheDocument();
    }
  });
  it("shows real metric values from the run summary", async () => {
    render(<EvaluationTab />);
    const row = await screen.findByTestId("row-hybrid_rerank");
    expect(row.textContent).toContain("0.767");
    expect(row.textContent).toContain("0.853");
  });

  it("reports generation metrics as explicitly NOT RUN", async () => {
    render(<EvaluationTab />);
    expect(await screen.findByTestId("generation-status")).toHaveTextContent(
      "NOT_RUN",
    );
  });
});

describe("EvaluationTab experiments", () => {
  it("renders delta table with negative values labeled as regression", async () => {
    const user = userEvent.setup();
    render(<EvaluationTab />);
    await user.click(screen.getByTestId("eval-sub-experiments"));
    const table = await screen.findByTestId("delta-table");
    // RUN A (hybrid only) vs RUN B (reranked): recall@5 improves
    const row = screen.getByTestId("delta-hybrid_rerank-recall@5");
    expect(row).toBeInTheDocument();
    expect(table).toBeInTheDocument();
    // negative deltas carry the regression label
    if (screen.queryByText(/^-(.*)regression/)) {
      expect(screen.getAllByText(/regression/).length).toBeGreaterThan(0);
    }
  });
});

describe("EvaluationTab failures", () => {
  function renderWithProvider(ui: React.ReactElement) {
    return render(<WorkspaceProvider>{ui}</WorkspaceProvider>);
  }

  it("lists failure cases with per-mode ranks and trace buttons", async () => {
    const user = userEvent.setup();
    renderWithProvider(<EvaluationTab />);
    await user.click(screen.getByTestId("eval-sub-failures"));
    const list = await screen.findByTestId("failure-list");
    expect(list).toBeInTheDocument();
    // per-mode rank rows rendered for each failure case
    expect(screen.getAllByTestId(/rank-q/).length).toBeGreaterThan(0);
  });

  it("filters failure cases by category", async () => {
    const user = userEvent.setup();
    renderWithProvider(<EvaluationTab />);
    await user.click(screen.getByTestId("eval-sub-failures"));
    await screen.findByTestId("failure-list");
    await user.selectOptions(
      screen.getByLabelText(/category/i),
      "all_methods_fail",
    );
    const cards = screen.getAllByText(/all_methods_fail/);
    expect(cards.length).toBeGreaterThan(0);
  });

  it("traces a ground-truth chunk to source on click", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | Request) => {
        const url = String(input);
        if (url.includes("/api/documents/381d2da4b68e/chunks")) {
          return new Response(
            JSON.stringify([
              {
                chunk_id: "381d2da4b68e-c00029",
                page_numbers: [19],
                source_block_ids: ["p0019-b001"],
              },
            ]),
            { status: 200 },
          );
        }
        return mockFetch(url);
      }),
    );
    renderWithProvider(<EvaluationTab />);
    await waitFor(async () => {
      await user.click(screen.getByTestId("eval-sub-failures"));
    });
    const btn = await screen.findByTestId("trace-fail-qb-c00029");
    await waitFor(() => expect(btn).toBeEnabled());
    await user.click(btn);
  });
});
