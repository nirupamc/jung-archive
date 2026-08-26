"use client";

import { useMemo, useState } from "react";
import type { CorpusStatus, DocumentSummary } from "@/lib/types";
import { useWorkspace } from "@/state/workspace";
import styles from "./LibrarySidebar.module.css";

const STATUS_ORDER: CorpusStatus[] = [
  "INDEXED",
  "CHUNKED",
  "PROCESSED",
  "DISCOVERED",
  "REVIEW",
  "EXCLUDED",
  "ERROR",
];

const SECTIONS: Array<{ key: string; label: string }> = [
  { key: "PRIMARY", label: "PRIMARY" },
  { key: "SECONDARY", label: "SECONDARY" },
];

export default function LibrarySidebar() {
  const { state, selectDocument } = useWorkspace();
  const [statusFilter, setStatusFilter] = useState<CorpusStatus | null>(null);

  const { bySection, presentStatuses } = useMemo(() => {
    const docs = state.documents ?? [];
    const map = new Map<string, DocumentSummary[]>();
    for (const s of SECTIONS) map.set(s.key, []);
    for (const d of docs) {
      const lane = SECTIONS.some((s) => s.key === d.section)
        ? d.section
        : "UNKNOWN";
      if (!map.has(lane)) map.set(lane, []);
      map.get(lane)!.push(d);
    }
    for (const [, list] of map) {
      list.sort(
        (a, b) =>
          STATUS_ORDER.indexOf(a.status as CorpusStatus) -
            STATUS_ORDER.indexOf(b.status as CorpusStatus) ||
          (a.title ?? "").localeCompare(b.title ?? ""),
      );
    }
    return {
      bySection: map,
      presentStatuses: STATUS_ORDER.filter((s) =>
        docs.some((d) => d.status === s),
      ),
    };
  }, [state.documents]);

  if (state.documentsError) {
    return (
      <aside className={styles.wrap} aria-label="document library">
        <h2 className={styles.heading}>library</h2>
        <div role="alert" className={styles.error}>
          backend unreachable: {state.documentsError}
        </div>
      </aside>
    );
  }

  if (!state.documents) {
    return (
      <aside className={styles.wrap} aria-label="document library">
        <h2 className={styles.heading}>library</h2>
        <div role="status" className={styles.loading}>
          loading documents …
        </div>
      </aside>
    );
  }

  const matches = (d: DocumentSummary) =>
    statusFilter === null || d.status === statusFilter;

  return (
    <aside className={styles.wrap} aria-label="document library">
      <h2 className={styles.heading}>library</h2>

      <div
        className={styles.filters}
        role="group"
        aria-label="filter by processing status"
        data-testid="library-filters"
      >
        <button
          type="button"
          aria-pressed={statusFilter === null}
          className={[
            styles.filterChip,
            statusFilter === null ? styles.filterActive : "",
          ].join(" ")}
          onClick={() => setStatusFilter(null)}
        >
          all ({state.documents.length})
        </button>
        {presentStatuses.map((s) => (
          <button
            key={s}
            type="button"
            aria-pressed={statusFilter === s}
            data-testid={`status-filter-${s.toLowerCase()}`}
            className={[
              styles.filterChip,
              styles[`f_${s}`],
              statusFilter === s ? styles.filterActive : "",
            ].join(" ")}
            onClick={() => setStatusFilter(statusFilter === s ? null : s)}
          >
            {s.toLowerCase()}{" "}
            {state.documents?.filter((d) => d.status === s).length ?? 0}
          </button>
        ))}
      </div>

      {SECTIONS.map(({ key, label }) => {
        const docs = (bySection.get(key) ?? []).filter(matches);
        const total = (bySection.get(key) ?? []).length;
        if (total === 0) return null;
        return (
          <section key={key} aria-label={`${label.toLowerCase()} sources`}>
            <h3 className={styles.sectionHeading}>
              {label} <span>{docs.length}/{total}</span>
            </h3>
            {docs.length === 0 && (
              <p className={styles.noneNote}>no documents match filter</p>
            )}
            <ul className={styles.list}>
              {docs.map((d) => (
                <li key={d.document_id}>
                  <button
                    type="button"
                    data-testid={`doc-${d.document_id}`}
                    aria-pressed={state.documentId === d.document_id}
                    className={[
                      styles.card,
                      state.documentId === d.document_id ? styles.selected : "",
                    ].join(" ")}
                    onClick={() => selectDocument(d.document_id)}
                  >
                    <span className={styles.title}>
                      {d.title ?? d.document_id}
                    </span>
                    <span className={styles.author}>
                      {d.author ?? "author unverified"}
                    </span>
                    <span className={styles.tags}>
                      <em
                        className={[
                          styles.tag,
                          d.source_type === "PRIMARY" ? styles.primary : "",
                        ].join(" ")}
                      >
                        {d.source_type}
                      </em>
                      <em
                        data-testid={`doc-status-${d.document_id}`}
                        className={[
                          styles.tag,
                          styles[`st_${d.status}`],
                        ].join(" ")}
                      >
                        {d.status}
                      </em>
                      <span className={styles.counts}>
                        {d.page_count > 0 && `${d.page_count} pp`}
                        {d.chunk_count > 0 && ` · ${d.chunk_count} chunks`}
                        {!d.has_pdf && " · no pdf"}
                        {!d.registered && " · unregistered"}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </aside>
  );
}

export type { DocumentSummary };
