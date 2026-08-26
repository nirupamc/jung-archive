"use client";

import type { DocumentSummary } from "@/lib/types";
import styles from "./UnprocessedNotice.module.css";

const STATUS_EXPLANATIONS: Record<string, string> = {
  DISCOVERED:
    "This file was found by the corpus scan but has not been processed yet: no canonical page inspection, chunks, or index vectors exist for it.",
  REVIEW:
    "Held for review. No explicit curation decision is recorded (or the registry says REVIEW), so the safety policy keeps it out of the pipeline until a human decides.",
  EXCLUDED:
    "Excluded by an explicit registry decision. This document will never be chunked, embedded, or indexed.",
  ERROR:
    "The PDF could not be opened during discovery. It is shown here rather than hidden so the problem stays visible.",
};

export default function UnprocessedNotice({
  doc,
}: {
  doc: DocumentSummary;
}) {
  const explanation =
    STATUS_EXPLANATIONS[doc.status] ?? "This document is not processable yet.";
  return (
    <div className={styles.wrap} data-testid="unprocessed-notice">
      <h2 className={styles.title}>{doc.title ?? doc.document_id}</h2>
      <p className={styles.author}>
        {doc.author ?? "author unverified"} · {doc.section} ·{" "}
        {doc.page_count} pages
      </p>
      <p className={styles.statusLine}>
        status:{" "}
        <strong
          className={[
            styles.badge,
            styles[`b_${doc.status}`],
          ].join(" ")}
        >
          {doc.status}
        </strong>
      </p>
      <p className={styles.explanation}>{explanation}</p>
      {doc.registered_reason && (
        <blockquote className={styles.reason}>
          registry note: {doc.registered_reason}
        </blockquote>
      )}
      {!doc.registered && (
        <p className={styles.policyNote}>
          Folder location alone never grants trust. Register an explicit
          decision in <code>config/document_metadata.json</code> to change
          this document&apos;s status.
        </p>
      )}
      {(doc.status === "DISCOVERED") && (
        <p className={styles.howTo}>
          Process approved documents with:{" "}
          <code>python -m jung_archive.cli corpus ingest</code>
        </p>
      )}
    </div>
  );
}
