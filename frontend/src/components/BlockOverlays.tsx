import type { BlockOut } from "@/lib/types";
import styles from "./BlockOverlays.module.css";

const BLOCK_CLASS: Record<string, string> = {
  TITLE: "tTitle",
  HEADING: "tHeading",
  PARAGRAPH: "tParagraph",
  LIST: "tList",
  TABLE: "tTable",
  FIGURE: "tFigure",
  CAPTION: "tCaption",
  HEADER: "tHeader",
  FOOTER: "tFooter",
  PAGE_NUMBER: "tPageNumber",
  UNKNOWN: "tUnknown",
};

interface Props {
  blocks: BlockOut[];
  scale: number;
  selectedIds: string[];
  onSelect?: (block: BlockOut) => void;
}

export default function BlockOverlays({
  blocks,
  scale,
  selectedIds,
  onSelect,
}: Props) {
  const selected = new Set(selectedIds);
  return (
    <div className={styles.layer} aria-label="canonical block overlays">
      {blocks.map((b) => (
        <button
          key={b.block_id}
          type="button"
          data-testid={`overlay-${b.block_id}`}
          data-block-type={b.block_type}
          aria-pressed={selected.has(b.block_id)}
          className={[
            styles.box,
            styles[BLOCK_CLASS[b.block_type] ?? "tUnknown"],
            selected.has(b.block_id) ? styles.selected : "",
          ].join(" ")}
          style={{
            left: b.bbox.x0 * scale,
            top: b.bbox.y0 * scale,
            width: (b.bbox.x1 - b.bbox.x0) * scale,
            height: (b.bbox.y1 - b.bbox.y0) * scale,
          }}
          onClick={() => onSelect?.(b)}
        >
          <span className={styles.tag}>{b.block_type.toLowerCase()}</span>
        </button>
      ))}
    </div>
  );
}
