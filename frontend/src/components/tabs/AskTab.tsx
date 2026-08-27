"use client";

import { useState } from "react";
import { useWorkspace } from "@/state/workspace";
import type { AskResponse, SourceCard } from "@/lib/types";
import AskLoader from "../AskLoader";
import styles from "./AskTab.module.css";

function renderAnswer(text: string, onCiteClick: (id: string) => void) {
  const parts = text.split(/(\[S\d+\])/g);
  return parts.map((part, i) => {
    const m = part.match(/^\[(S\d+)\]$/);
    if (m) {
      return (
        <button
          key={i}
          type="button"
          className={styles.citeLink}
          onClick={() => onCiteClick(m[1])}
          aria-label={`Jump to source ${m[1]}`}
        >
          {part}
        </button>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function toSourceCards(resp: AskResponse): SourceCard[] {
  const byId = new Map<string, SourceCard>();
  for (const item of resp.evidence_pack.items) {
    const lo = item.page_numbers.length ? Math.min(...item.page_numbers) : 0;
    const hi = item.page_numbers.length ? Math.max(...item.page_numbers) : 0;
    byId.set(item.evidence_id, {
      evidence_id: item.evidence_id,
      document_id: item.document_id,
      title: item.title ?? null,
      author: item.author ?? null,
      pages: lo === hi ? String(lo) : `${lo}-${hi}`,
      section: item.section_id ?? null,
      excerpt: item.clean_text,
    });
  }
  return Array.from(byId.values());
}

export default function AskTab() {
  const { state, dispatch, traceToSource, ask } = useWorkspace();
  const [showTrace, setShowTrace] = useState(false);
  const [showOriginal, setShowOriginal] = useState<Record<string, boolean>>({});

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!state.query.trim() || state.askLoading) return;
    void ask();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!state.askLoading && state.query.trim()) {
        void ask();
      }
    }
  };

  const resp = state.ask;
  const sourceCards = resp ? toSourceCards(resp) : [];

  return (
    <div className={styles.wrap}>
      <form className={styles.header} onSubmit={submit}>
        <h2 className={styles.title}>ASK JUNG ARCHIVE</h2>
        <p className={styles.subtitle}>
          Ask questions across Jung&rsquo;s indexed works
        </p>

        <div className={styles.inputRow}>
          <textarea
            id="ask-query"
            value={state.query}
            onChange={(e) =>
              dispatch({ type: "set_query", query: e.target.value })
            }
            onKeyDown={handleKeyDown}
            placeholder="Ask across the Jung archive..."
            rows={3}
            disabled={state.askLoading}
            className={styles.textarea}
            data-testid="ask-input"
          />
          <button
            type="submit"
            disabled={!state.query.trim() || state.askLoading}
            className={styles.submitBtn}
            data-testid="ask-submit"
          >
            {state.askLoading ? "ASKING…" : "ASK"}
          </button>
        </div>

        <div className={styles.metaRow}>
          {resp && (
            <span
              className={styles.providerBadge}
              data-testid="provider-mode"
            >
              {resp.local_or_remote}
              {" · "}
              {resp.provider}
              {resp.model ? ` · ${resp.model}` : ""}
            </span>
          )}
        </div>
      </form>

      {state.askError && (
        <div role="alert" className={styles.error}>
          {state.askError}
        </div>
      )}

      {state.askLoading && (
        <div className={styles.loadingWrap}>
          <AskLoader />
        </div>
      )}

      {resp && !state.askLoading && (
        <>
          <section className={styles.answerSection} data-testid="answer">
            <h3 className={styles.stageTitle}>ANSWER</h3>
            <p className={styles.answerText}>
              {renderAnswer(resp.answer, (id) => {
                const el = document.getElementById(`source-${id}`);
                el?.scrollIntoView({ behavior: "smooth", block: "center" });
                el?.classList.add(styles.flash);
                setTimeout(() => el?.classList.remove(styles.flash), 900);
              })}
            </p>
          </section>

          {resp.citations.length > 0 && (
            <section className={styles.sourcesSection}>
              <h3 className={styles.stageTitle}>SOURCES</h3>
              <ol className={styles.sourceList}>
                {sourceCards.map((card) => {
                  const cit = resp.citations.find(
                    (c) => c.evidence_id === card.evidence_id,
                  );
                  return (
                    <li
                      key={card.evidence_id}
                      id={`source-${card.evidence_id}`}
                      className={styles.sourceCard}
                      data-testid={`source-${card.evidence_id}`}
                    >
                      <div className={styles.sourceHead}>
                        <span className={styles.sourceId}>
                          [{card.evidence_id}]
                        </span>
                        <span className={styles.sourceTitle}>
                          {card.title ?? card.document_id}
                        </span>
                        {cit && (
                          <span
                            className={[
                              styles.sourceStatus,
                              cit.status === "valid"
                                ? styles.statusValid
                                : styles.statusUnknown,
                            ].join(" ")}
                          >
                            {cit.status}
                          </span>
                        )}
                      </div>
                      <div className={styles.sourceMeta}>
                        p. {card.pages}
                        {card.section ? ` · ${card.section}` : ""}
                      </div>
                      <pre className={styles.sourceExcerpt}>
                        {card.excerpt.slice(0, 220)}
                        {card.excerpt.length > 220 ? "…" : ""}
                      </pre>
                      <button
                        type="button"
                        className={styles.traceBtn}
                        onClick={() =>
                          traceToSource({
                            pageNumbers: resp.evidence_pack.items.find(
                              (i) => i.evidence_id === card.evidence_id,
                            )?.page_numbers ?? [],
                            blockIds: resp.evidence_pack.items.find(
                              (i) => i.evidence_id === card.evidence_id,
                            )?.source_block_ids ?? [],
                            chunkId: resp.evidence_pack.items.find(
                              (i) => i.evidence_id === card.evidence_id,
                            )?.chunk_id,
                          })
                        }
                      >
                        trace → source
                      </button>
                    </li>
                  );
                })}
              </ol>
            </section>
          )}

          <details
            className={styles.traceDetails}
            open={showTrace}
            onToggle={(e) => setShowTrace((e.target as HTMLDetailsElement).open)}
          >
            <summary className={styles.traceSummary}>
              VIEW RETRIEVAL TRACE
            </summary>
            <div className={styles.traceGrid}>
              <div className={styles.traceItem}>
                <span className={styles.traceLabel}>Query</span>
                <span className={styles.traceValue}>{resp.retrieval_metadata.mode ?? "hybrid"}</span>
              </div>
              <div className={styles.traceItem}>
                <span className={styles.traceLabel}>Results</span>
                <span className={styles.traceValue}>
                  {resp.retrieval_metadata.results ?? 0}
                </span>
              </div>
              <div className={styles.traceItem}>
                <span className={styles.traceLabel}>Latency</span>
                <span className={styles.traceValue}>
                  {resp.retrieval_metadata.latency_ms ?? "?"} ms
                </span>
              </div>
              <div className={styles.traceItem}>
                <span className={styles.traceLabel}>Evidence</span>
                <span className={styles.traceValue}>
                  {resp.evidence_pack.items?.length ?? 0} /{" "}
                  {resp.evidence_pack.max_evidence_items ?? 0} items
                </span>
              </div>
              <div className={styles.traceItem}>
                <span className={styles.traceLabel}>Tokens</span>
                <span className={styles.traceValue}>
                  {resp.evidence_pack.tokens_used ?? 0} /{" "}
                  {resp.evidence_pack.max_evidence_tokens ?? 0}
                </span>
              </div>
            </div>
          </details>

          {resp.warnings.length > 0 && (
            <div className={styles.warnings} role="status">
              {resp.warnings.map((w, i) => (
                <div key={i} className={styles.warningItem}>
                  {w}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
