"""
Structure-aware chunker.

Principles:
  - preserve paragraph/block boundaries; never concatenate across sections
  - headings stay attached to the content that follows them
  - reading order and provenance are preserved exactly
  - token budget: pack toward target, enforce max, avoid sub-min chunks
    except at structural boundaries where merging is impossible
  - limited overlap carried from the previous chunk within a section
"""
import re
from typing import List, Optional, Tuple

from jung_archive.chunking.tokenizer import (
    count_tokens,
    split_text_into_token_windows,
    truncate_to_tokens,
)
from jung_archive.models.chunk import Chunk, ChunkingConfig
from jung_archive.models.document import BlockType, Document
from jung_archive.structure.sections import SectionTree, build_section_tree

_SENTENCE_RE = re.compile(r"(?<=[.!?:;\u201d\u2019])\s+")


class StructureAwareChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk_document(self, document: Document) -> List[Chunk]:
        tree = build_section_tree(document)
        return self._chunk_tree(document, tree)

    # ------------------------------------------------------------------
    def _chunk_tree(self, document: Document, tree: SectionTree) -> List[Chunk]:
        chunks: List[Chunk] = []
        cfg = self.config
        # Heading anchors whose section had no content: carried into the
        # next section rather than emitted as meaningless tiny chunks.
        carried: List[Tuple[int, object]] = []

        for section in tree.sections:
            heading = section.anchor  # (page_number, block) or None
            # Carried orphan headings from previous contentless sections
            # ride at the front of this section's first chunk.
            if carried and section.items:
                pending: List[Tuple[int, object]] = list(carried)
                carried.clear()
            else:
                pending = []
            prev_tail = ""          # overlap carrier within this section

            def token_total() -> int:
                t = sum(count_tokens(b.text) for _, b in pending)
                if heading is not None:
                    t += count_tokens(heading[1].text)
                # account for "\n\n" separators (~1 token each)
                parts = len(pending) + (1 if heading is not None else 0)
                return t + max(0, parts - 1)

            def emit(block_items, head_item, extra_prefix):
                nonlocal prev_tail
                # Merge heading + content and order strictly by document
                # position so IDs/text follow true reading order (carried
                # headings from earlier pages may precede this section's
                # anchor).
                combined = list(block_items)
                if head_item is not None:
                    combined.append(head_item)
                combined.sort(key=lambda pb: (pb[0], round(pb[1].reading_order, 1)))

                ids = [b.block_id for _, b in combined]
                pages = {p for p, _ in combined}
                parts = []
                if extra_prefix:
                    parts.append(extra_prefix)
                parts.extend(b.text.strip() for _, b in combined)
                text = "\n\n".join(p for p in parts if p)
                page_list = sorted(pages)
                return text, ids, page_list

            def flush(force: bool = False):
                nonlocal heading, pending, prev_tail
                if not pending and heading is None:
                    return
                if not force and token_total() < cfg.min_tokens:
                    return
                text, ids, page_list = emit(pending, heading, prev_tail)
                if not text.strip():
                    heading, pending = None, []
                    return
                chunks.append(Chunk(
                    chunk_id=f"{document.document_id}-c{len(chunks):05d}",
                    document_id=document.document_id,
                    text=text,
                    source_block_ids=ids,
                    page_numbers=page_list,
                    heading_path=list(section.heading_path),
                    token_count=count_tokens(text),
                    source_type=document.source_type,
                    section_id=section.section_id,
                    start_page=min(page_list),
                    end_page=max(page_list),
                    char_count=len(text),
                    strategy=cfg.strategy_name,
                    created_from_blocks=list(ids),
                    metadata={
                        "section_title": section.title,
                        "heading_level": section.level,
                    },
                ))
                body_text = " ".join(b.text.strip() for _, b in pending)
                prev_tail = self._tail_overlap(body_text)
                heading, pending = None, []

            for i, (page_number, block) in enumerate(section.items):
                if block.block_type in (BlockType.TITLE, BlockType.HEADING):
                    # Defensive: sections normally own headings; if one leaks
                    # through, treat it as a new heading context.
                    flush(force=True)
                    heading = (page_number, block)
                    continue

                btoks = count_tokens(block.text)
                if btoks > cfg.max_tokens:
                    # Oversized block: flush current state, then emit windows
                    # sized so that heading/overlap prefixes never push the
                    # final chunk over max_tokens.
                    flush(force=True)
                    prefix_parts = []
                    if heading is not None:
                        prefix_parts.append(heading[1].text.strip())
                    if prev_tail:
                        prefix_parts.append(prev_tail)
                    prefix_tokens = count_tokens("\n\n".join(prefix_parts))
                    # +1 per join separator so emitted chunks stay <= max
                    join_overhead = len(prefix_parts)  # prefix parts + window - 1
                    window_tokens = max(
                        1, cfg.max_tokens - prefix_tokens - join_overhead
                    )
                    windows = split_text_into_token_windows(
                        block.text.strip(), window_tokens, cfg.overlap_tokens
                    )
                    for w_i, window in enumerate(windows):
                        ids, pages = [block.block_id], {page_number}
                        parts = []
                        if w_i == 0 and heading is not None:
                            parts.append(heading[1].text.strip())
                            ids.insert(0, heading[1].block_id)
                            pages.add(heading[0])
                        if w_i == 0 and prev_tail:
                            parts.append(prev_tail)
                        parts.append(window)
                        text = "\n\n".join(p for p in parts if p)
                        page_list = sorted(pages)
                        chunks.append(Chunk(
                            chunk_id=f"{document.document_id}-c{len(chunks):05d}",
                            document_id=document.document_id,
                            text=text,
                            source_block_ids=ids,
                            page_numbers=page_list,
                            heading_path=list(section.heading_path),
                            token_count=count_tokens(text),
                            source_type=document.source_type,
                            section_id=section.section_id,
                            start_page=min(page_list),
                            end_page=max(page_list),
                            char_count=len(text),
                            strategy=cfg.strategy_name,
                            created_from_blocks=[block.block_id],
                            metadata={
                                "section_title": section.title,
                                "split_part": w_i + 1,
                                "split_parts": len(windows),
                            },
                        ))
                        prev_tail = self._tail_overlap(window)
                    heading = None
                    continue

                if token_total() + btoks + 1 > cfg.max_tokens and (pending or heading):
                    # +1 budgets the "\n\n" separator that joining will add
                    flush(force=True)

                pending.append((page_number, block))

                if token_total() >= cfg.target_tokens:
                    flush(force=False)

            # End of section: flush remaining content; an anchor heading
            # with no content is carried into the next section instead of
            # being emitted as a meaningless tiny chunk.
            if pending:
                flush(force=True)
            elif heading is not None:
                carried.append(heading)
                heading = None

        # Document ended with contentless heading sections: emit them as a
        # single trailing chunk so no source block is silently lost.
        if carried:
            ids = [b.block_id for _, b in carried]
            pages = sorted({p for p, _ in carried})
            text = "\n\n".join(b.text.strip() for _, b in carried)
            chunks.append(Chunk(
                chunk_id=f"{document.document_id}-c{len(chunks):05d}",
                document_id=document.document_id,
                text=text,
                source_block_ids=ids,
                page_numbers=pages,
                heading_path=[],
                token_count=count_tokens(text),
                source_type=document.source_type,
                start_page=min(pages),
                end_page=max(pages),
                char_count=len(text),
                strategy=cfg.strategy_name,
                created_from_blocks=list(ids),
                metadata={"trailing_headings": True},
            ))

        # Final deterministic numbering over the whole document
        for idx, ch in enumerate(chunks):
            ch.chunk_index = idx
            ch.chunk_id = f"{document.document_id}-c{idx:05d}"
        return chunks

    def _tail_overlap(self, text: str) -> str:
        """Last ~overlap_tokens worth of sentence tail, for continuity."""
        budget = self.config.overlap_tokens
        if budget <= 0 or not text:
            return ""
        sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
        tail: List[str] = []
        used = 0
        for s in reversed(sentences):
            t = count_tokens(s)
            if used + t > budget:
                if not tail:
                    tail.insert(0, truncate_to_tokens(s.strip(), budget))
                break
            tail.insert(0, s.strip())
            used += t
        return " ".join(tail)
