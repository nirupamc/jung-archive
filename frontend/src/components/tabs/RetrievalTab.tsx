"use client";

import { useState } from "react";
import type { EvidencePack, RetrievalMode, RetrievalResult } from "@/lib/types";
import { RETRIEVAL_MODES } from "@/lib/types";
import { useWorkspace } from "@/state/workspace";
import ScorePathTable from "../ScorePathTable";
import styles from "./RetrievalTab.module.css";

const MODE_LABELS: Record<RetrievalMode, string> = {
  dense: "DENSE",
  bm25: "BM25",
  hybrid: "HYBRID",
  hybrid_rerank: "HYBRID + RERANKER",
};

export default function RetrievalTab() {
  const {
    state,
    dispatch,
    runSearch,
    traceToSource,
    assembleEvidence,
  } = useWorkspace();
  const [openChunk, setOpenChunk] = useState<string | null>(null);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!state.query.trim()) return;
    void runSearch();
  };

  return (
    <div className={styles.wrap}>
      <form className={styles.queryBox} onSubmit={submit}>
        <label htmlFor="retrieval-query" className={styles.queryLabel}>
          query
        </label>
        <input
          id="retrieval-query"
          value={state.query}
          onChange={(e) => dispatch({ type: "set_query", query: e.target.value })}
          placeholder='e.g. "What protects an individual from mass psychology?"'
        />
        <div
          className={styles.modeRow}
          role="radiogroup"
          aria-label="retrieval mode"
        >
          {RETRIEVAL_MODES.map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={state.mode === m}
              data-testid={`mode-${m}`}
              className={[
                styles.modeBtn,
                state.mode === m ? styles.modeActive : "",
              ].join(" ")}
              onClick={() => dispatch({ type: "set_mode", mode: m })}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
        </div>
        <div className={styles.actions}>
          <button
            type="submit"
            disabled={!state.query.trim() || state.retrievalLoading}
          >
            {state.retrievalLoading ? "searching…" : "run retrieval"}
          </button>
          <button
            type="button"
            disabled={!state.query.trim() || state.evidenceLoading}
            data-testid="assemble-evidence"
            onClick={() => void assembleEvidence(state.query)}
          >
            {state.evidenceLoading ? "assembling…" : "assemble evidence"}
          </button>
        </div>
      </form>

      {state.retrievalError && (
        <div role="alert" className={styles.error}>
          retrieval failed: {state.retrievalError}
        </div>
      )}
      {state.retrieval && (
        <PipelineFlow
          response={state.retrieval}
          openChunk={openChunk}
          onToggle={(cid) => setOpenChunk(openChunk === cid ? null : cid)}
          onTrace={(r) =>
            traceToSource({
              pageNumbers: r.page_numbers,
              blockIds: r.source_block_ids,
              chunkId: r.chunk_id,
            })
          }
        />
      )}

      {state.evidenceError && (
        <div role="alert" className={styles.error}>
          evidence assembly failed: {state.evidenceError}
        </div>
      )}

      {state.evidence && (
        <EvidenceView pack={state.evidence} onTrace={traceToSource} />
      )}
    </div>
  );
}

function PipelineFlow({
  response,
  openChunk,
  onToggle,
  onTrace,
}: {
  response: import("@/lib/types").RetrievalResponse;
  openChunk: string | null;
  onToggle: (chunkId: string) => void;
  onTrace: (r: RetrievalResult) => void;
}) {
  return (
    <section aria-label="retrieval pipeline" className={styles.flowSection}>
      <p className={styles.stageLabel}>QUERY</p>
      <blockquote className={styles.queryEcho}>{response.query}</blockquote>

      <Stage title="FUSED CANDIDATES (RRF)" note={`${response.results.length} results · ${response.latency_ms ?? "?"} ms`}>
        {response.results.map((r) => (
          <ResultRow
            key={r.chunk_id}
            r={r}
            rank={r.fusion_rank}
            scoreLabel={`rrf ${r.fusion_score}`}
            expanded={openChunk === r.chunk_id}
            onToggle={() => onToggle(r.chunk_id)}
            onTrace={() => onTrace(r)}
          />
        ))}
      </Stage>

      {(response.mode === "hybrid_rerank") && (
        <>
          <div className={styles.arrow} aria-hidden="true">↓</div>
          <Stage
            title="RERANKER (cross-encoder)"
            note={`reranked ${response.candidates_reranked} candidates${
              response.pairs_truncated ? ` · ${response.pairs_truncated} pairs truncated` : ""
            }`}
          >
            {[...response.results]
              .sort((a, b) => (a.reranker_rank ?? 99) - (b.reranker_rank ?? 99))
              .map((r) => (
                <ResultRow
                  key={r.chunk_id}
                  r={r}
                  rank={r.reranker_rank}
                  scoreLabel={`ce ${r.reranker_score?.toFixed(3)}`}
                  muted
                  expanded={openChunk === r.chunk_id}
                  onToggle={() => onToggle(r.chunk_id)}
                  onTrace={() => onTrace(r)}
                />
              ))}
          </Stage>
        </>
      )}
    </section>
  );
}

function Stage({
  title,
  note,
  children,
}: {
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={styles.stage}>
      <h4 className={styles.stageTitle}>
        {title} {note && <span className={styles.stageNote}>{note}</span>}
      </h4>
      <ol className={styles.resultList}>{children}</ol>
    </div>
  );
}

function ResultRow({
  r,
  rank,
  scoreLabel,
  muted = false,
  expanded,
  onToggle,
  onTrace,
}: {
  r: RetrievalResult;
  rank: number | null;
  scoreLabel: string;
  muted?: boolean;
  expanded?: boolean;
  onToggle: () => void;
  onTrace: () => void;
}) {
  return (
    <li className={styles.resultItem}>
      <div className={styles.resultHead}>
        <span className={muted ? styles.rankMuted : styles.rank}>#{rank}</span>
        <code className={styles.cid}>{r.chunk_id}</code>
        <span className={styles.pages}>
          p.{Math.min(...(r.page_numbers.length ? r.page_numbers : [0]))}
          {r.page_numbers.length > 1 ? `–${Math.max(...r.page_numbers)}` : ""}
        </span>
        <span className={styles.score}>{scoreLabel}</span>
        <button type="button" className={styles.miniBtn} onClick={onToggle} aria-expanded={expanded}>
          {expanded ? "hide details" : "details"}
        </button>
        <button type="button" data-testid={`trace-${r.chunk_id}`} className={styles.traceMini} onClick={onTrace}>
          trace →
        </button>
      </div>
      <p className={styles.preview}>{r.text.slice(0, 180)}</p>
      {expanded && (
        <ScorePathTable result={r} />
      )}
    </li>
  );
}

export function EvidenceView({
  pack,
  onTrace,
}: {
  pack: EvidencePack;
  onTrace: (t: { pageNumbers: number[]; blockIds: string[]; chunkId?: string | null }) => void;
}) {
  const [showOriginal, setShowOriginal] = useState<Record<string, boolean>>({});
  return (
    <section aria-label="evidence pack" className={styles.evidence} data-testid="evidence-pack">
      <h3 className={styles.stageTitle}>EVIDENCE PACK</h3>
      <div className={styles.budget} data-testid="evidence-budget">
        budget: {pack.tokens_used} / {pack.max_evidence_tokens} tokens ·{" "}
        {pack.items.length} / {pack.max_evidence_items} items
      </div>
      {pack.suppressed_duplicates.length > 0 && (
        <div className={styles.dedup} data-testid="suppressed-duplicates">
          {pack.suppressed_duplicates.length} duplicate(s) removed:{" "}
          <ul>
            {pack.suppressed_duplicates.map((s) => (
              <li key={s.chunk_id}>
                <code>{s.chunk_id}</code> — {s.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
      <ol className={styles.evidenceList}>
        {pack.items.map((item) => (
          <li key={item.evidence_id} className={styles.evidenceItem}>
            <div className={styles.evHead}>
              <span className={styles.evId}>[{item.evidence_id}]</span>
              <span>{item.title ?? item.document_id}</span>
              <span>
                pages {item.page_numbers.join(", ")} · {item.token_count} tok
              </span>
              <button
                type="button"
                data-testid={`trace-evidence-${item.evidence_id}`}
                className={styles.traceMini}
                onClick={() =>
                  onTrace({
                    pageNumbers: item.page_numbers,
                    blockIds: item.source_block_ids,
                    chunkId: item.chunk_id,
                  })
                }
              >
                trace → source
              </button>
            </div>
            <dl className={styles.evMeta}>
              <dt>chunk</dt>
              <dd><code>{item.chunk_id}</code></dd>
              <dt>cleanup</dt>
              <dd>
                {item.was_cleaned ? item.cleanup_operations.join(", ") : "none"}
              </dd>
              <dt>selection</dt>
              <dd>{item.selection_reason}</dd>
            </dl>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={!!showOriginal[item.evidence_id]}
                onChange={(e) =>
                  setShowOriginal((s) => ({
                    ...s,
                    [item.evidence_id]: e.target.checked,
                  }))
                }
              />{" "}
              show original text
            </label>
            <pre className={styles.evText} data-testid={`evtext-${item.evidence_id}`}>
              {showOriginal[item.evidence_id] ? item.text : item.clean_text}
            </pre>
          </li>
        ))}
      </ol>
    </section>
  );
}
