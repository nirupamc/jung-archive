import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ScorePathTable from "@/components/ScorePathTable";
import BlockOverlays from "@/components/BlockOverlays";
import { EvidenceView } from "@/components/tabs/RetrievalTab";
import { EVIDENCE_PACK, HYBRID_RESPONSE, RERANK_RESPONSE, PAGE_17 } from "./fixtures";
import type { RetrievalResult } from "@/lib/types";

// ---------------------------------------------------------------------
// Score path rendering

describe("ScorePathTable", () => {
  it("renders all four stages for a reranked result", () => {
    const r = RERANK_RESPONSE.results[1];
    render(<ScorePathTable result={r} />);
    expect(screen.getByText("dense")).toBeInTheDocument();
    expect(screen.getByText("bm25")).toBeInTheDocument();
    expect(screen.getByText("fusion (RRF)")).toBeInTheDocument();
    expect(screen.getByText("reranker (cross-encoder logit)")).toBeInTheDocument();
  });

  it("renders nullable scores as an em dash, never fake values", () => {
    const r = RERANK_RESPONSE.results[0]; // dense leg null
    render(<ScorePathTable result={r} />);
    const row = screen.getByTestId("score-row-dense");
    expect(row.textContent).toContain("—");
  });

  it("shows raw cross-encoder logits without normalizing", () => {
    const r: RetrievalResult = {
      ...RERANK_RESPONSE.results[1],
      reranker_score: -6.089736,
    };
    render(<ScorePathTable result={r} />);
    expect(
      screen.getByTestId("score-row-reranker").textContent,
    ).toContain("-6.089736");
  });
});

// ---------------------------------------------------------------------
// Pipeline / mode rendering

function ResultProbe({ r }: { r: RetrievalResult }) {
  return <div data-testid="probe">{r.reranker_rank ?? "none"}</div>;
}

describe("retrieval responses", () => {
  it("plain hybrid results carry no reranker fields", () => {
    render(
      <>
        {HYBRID_RESPONSE.results.map((r) => (
          <ResultProbe key={r.chunk_id} r={r} />
        ))}
      </>,
    );
    const probes = screen.getAllByTestId("probe");
    probes.forEach((p) => expect(p).toHaveTextContent("none"));
  });

  it("reranked results expose reranker ranks and preserve fusion ranks", () => {
    render(
      <>
        {RERANK_RESPONSE.results.map((r) => (
          <ResultProbe key={`${r.chunk_id}-${r.fusion_rank}`} r={r} />
        ))}
      </>,
    );
    expect(screen.getAllByTestId("probe").map((p) => p.textContent)).toEqual([
      "2",
      "1",
    ]);
  });
});

// ---------------------------------------------------------------------
// Block overlays

describe("BlockOverlays", () => {
  const scale = 2;

  function geom(id: string) {
    return document.querySelector(`[data-testid="overlay-${id}"]`) as HTMLElement;
  }

  it("positions boxes by bbox scaled to the rendered page", () => {
    render(
      <BlockOverlays
        blocks={PAGE_17.blocks}
        scale={scale}
        selectedIds={[]}
      />,
    );
    const b = PAGE_17.blocks[0];
    const el = geom(b.block_id);
    expect(el.style.left).toBe(`${b.bbox.x0 * scale}px`);
    expect(el.style.top).toBe(`${b.bbox.y0 * scale}px`);
    expect(el.style.width).toBe(`${(b.bbox.x1 - b.bbox.x0) * scale}px`);
  });

  it("marks selected blocks with aria-pressed and the selected class", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <BlockOverlays
        blocks={PAGE_17.blocks}
        scale={scale}
        selectedIds={["p0017-b000"]}
        onSelect={onSelect}
      />,
    );
    expect(geom("p0017-b000")).toHaveAttribute("aria-pressed", "true");
    expect(geom("p0017-b001")).toHaveAttribute("aria-pressed", "false");
    await user.click(geom("p0017-b001"));
    expect(onSelect).toHaveBeenCalledWith(PAGE_17.blocks[1]);
  });
});

// ---------------------------------------------------------------------
// Evidence view

describe("EvidenceView", () => {
  const onTrace = vi.fn();

  it("renders evidence id, budget and duplicate suppression", () => {
    render(<EvidenceView pack={EVIDENCE_PACK} onTrace={onTrace} />);
    expect(screen.getByText("[S1]")).toBeInTheDocument();
    expect(screen.getByTestId("evidence-budget").textContent).toContain(
      "1634 / 2500 tokens",
    );
    expect(screen.getByTestId("suppressed-duplicates").textContent).toContain(
      "c00027",
    );
    expect(screen.getByTestId("suppressed-duplicates").textContent).toContain(
      "0.87",
    );
  });

  it("defaults to clean text and toggles to original", async () => {
    const user = userEvent.setup();
    render(<EvidenceView pack={EVIDENCE_PACK} onTrace={onTrace} />);
    const pre = screen.getByTestId("evtext-S1");
    expect(pre.textContent).not.toContain("the undiscovered self");
    await user.click(screen.getByLabelText(/show original text/i));
    expect(pre.textContent).toContain("the undiscovered self");
  });

  it("lists cleanup operations only when cleaned", () => {
    render(<EvidenceView pack={EVIDENCE_PACK} onTrace={onTrace} />);
    expect(screen.getByText(/removed_running_header/)).toBeInTheDocument();
    expect(screen.getByText(/removed_folio/)).toBeInTheDocument();
  });

  it("traces evidence to source on click", async () => {
    const user = userEvent.setup();
    render(<EvidenceView pack={EVIDENCE_PACK} onTrace={onTrace} />);
    await user.click(screen.getByTestId("trace-evidence-S1"));
    expect(onTrace).toHaveBeenCalledWith({
      pageNumbers: [18, 19],
      blockIds: ["p0018-b002", "p0019-b001"],
      chunkId: "381d2da4b68e-c00028",
    });
  });
});
