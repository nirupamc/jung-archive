"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { BlockOut } from "@/lib/types";
import { useWorkspace } from "@/state/workspace";
import styles from "./StructureTab.module.css";

const FURNITURE_TYPES = new Set(["HEADER", "FOOTER", "PAGE_NUMBER"]);

export default function StructureTab() {
  const { state, traceToSource } = useWorkspace();
  const [filter, setFilter] = useState("");
  const [showFurniture, setShowFurniture] = useState(false);
  const docId = state.documentId;

  // Keyed load state: avoids resetting state synchronously inside effects.
  const [loaded, setLoaded] = useState<{
    docId: string;
    items?: BlockOut[];
    error?: string;
  } | null>(null);

  useEffect(() => {
    if (!docId) return;
    let cancelled = false;
    api
      .structure(docId)
      .then((items) => !cancelled && setLoaded({ docId, items }))
      .catch((e: Error) =>
        !cancelled && setLoaded({ docId, error: e.message }),
      );
    return () => {
      cancelled = true;
    };
  }, [docId]);

  const items =
    loaded && loaded.docId === docId ? (loaded.items ?? null) : null;
  const error =
    loaded && loaded.docId === docId ? (loaded.error ?? null) : null;

  const filtered = useMemo(() => {
    if (!items) return [];
    let out = items;
    if (!showFurniture) {
      out = out.filter((b) => !FURNITURE_TYPES.has(b.block_type));
    }
    if (filter.trim()) {
      const f = filter.toLowerCase();
      out = out.filter(
        (b) =>
          b.block_type.toLowerCase().includes(f) ||
          b.text.toLowerCase().includes(f),
      );
    }
    return out;
  }, [items, filter, showFurniture]);

  // Group by page so long documents scan like a table of contents.
  const pageGroups = useMemo(() => {
    const groups: Array<{ page: number; blocks: BlockOut[] }> = [];
    for (const b of filtered) {
      const last = groups[groups.length - 1];
      if (last && last.page === b.page_number) {
        last.blocks.push(b);
      } else {
        groups.push({ page: b.page_number, blocks: [b] });
      }
    }
    return groups;
  }, [filtered]);

  if (error)
    return <div role="alert" className={styles.error}>structure unavailable: {error}</div>;
  if (!items)
    return <div role="status" className={styles.loading}>loading structure …</div>;

  return (
    <div className={styles.wrap}>
      <div className={styles.controls}>
        <label htmlFor="struct-filter">filter</label>
        <input
          id="struct-filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="type or text…"
        />
        <label className={styles.checkLabel}>
          <input
            type="checkbox"
            checked={showFurniture}
            onChange={(e) => setShowFurniture(e.target.checked)}
            data-testid="show-furniture"
          />{" "}
          page furniture
        </label>
        <span>
          {filtered.length} / {items.length} blocks · {pageGroups.length} pages
        </span>
      </div>
      <ol className={styles.flow} aria-label="canonical document structure">
        {pageGroups.map((group) => (
          <li key={group.page}>
            <div className={styles.pageDivider} role="separator">
              <span>page {group.page}</span>
              <span className={styles.pageCount}>
                {group.blocks.length} blocks
              </span>
            </div>
            <ol className={styles.pageBlocks}>
              {group.blocks.map((b) => (
                <li key={b.block_id}>
                  <button
                    type="button"
                    className={[
                      styles.item,
                      state.selectedBlockIds.includes(b.block_id)
                        ? styles.selected
                        : "",
                    ].join(" ")}
                    data-testid={`structure-${b.block_id}`}
                    onClick={() =>
                      traceToSource({
                        pageNumbers: [b.page_number],
                        blockIds: [b.block_id],
                      })
                    }
                  >
                    <span
                      className={`${styles.badge} ${styles[`t_${b.block_type}`] ?? ""}`}
                    >
                      {b.block_type}
                    </span>
                    {b.block_type === "HEADING" &&
                      typeof b.font_size === "number" && (
                        <em
                          className={styles.hLevel}
                          title={`font size ${b.font_size}`}
                        >
                          h{b.font_size >= 14 ? "1" : "2"}
                        </em>
                      )}
                    <span className={styles.text}>
                      {b.text.slice(0, 220)}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </li>
        ))}
      </ol>
    </div>
  );
}
