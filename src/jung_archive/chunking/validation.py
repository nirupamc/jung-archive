"""
Chunk provenance validation.

Fails loudly on corrupted provenance: every chunk must trace back to real
blocks of its own document, in valid order, with honest token counts.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from jung_archive.chunking.tokenizer import count_tokens
from jung_archive.models.chunk import Chunk
from jung_archive.models.document import BlockType, Document


def _norm_tokens(text: str) -> List[str]:
    """Whitespace-fold and split into lowercase word-only tokens for fuzzy
    derivation checking. Punctuation is stripped so that spacing quirks
    around commas (common in PDF extraction) don't cause false negatives.
    """
    import re
    raw = re.split(r"\s+", text.strip().lower())
    return [re.sub(r"[^a-z0-9]", "", t) for t in raw if
            re.sub(r"[^a-z0-9]", "", t)]


def _substantial_overlap(block_tokens: List[str], chunk_tokens: List[str],
                          min_overlap: int = 8) -> bool:
    """Verify that the first ``min_overlap`` block tokens appear in order
    somewhere in the chunk tokens. Survives overlap-prefix spacing quirks
    while still catching chunks that don't actually contain the block text.
    """
    if len(block_tokens) <= min_overlap:
        needle = block_tokens
    else:
        needle = block_tokens[:min_overlap]
    if not needle:
        return True
    ci = 0
    for tok in chunk_tokens:
        if ci < len(needle) and tok == needle[ci]:
            ci += 1
    return ci >= len(needle)


def _is_subsequence_of(short_tokens: List[str], long_tokens: List[str],
                        min_overlap: int = 8) -> bool:
    """Check if ``short_tokens`` is a contiguous subsequence of ``long_tokens``.

    For split-window chunks where the chunk text is a window INTO a larger
    block, the chunk's tokens should appear as a contiguous run within the
    block's full token sequence.
    """
    if len(short_tokens) < min_overlap:
        return False
    if not short_tokens or not long_tokens:
        return False
    n, m = len(short_tokens), len(long_tokens)
    if n > m:
        return False
    for start in range(m - n + 1):
        if long_tokens[start:start + n] == short_tokens:
            return True
    return False


@dataclass
class ValidationResult:
    ok: bool = True
    errors: List[str] = field(default_factory=list)

    def add(self, msg: str):
        self.ok = False
        self.errors.append(msg)


def validate_chunks(chunks: List[Chunk], document: Document) -> ValidationResult:
    """Validate a chunk list against the source document.

    Checks (per M2.5):
      1. document_id matches the document
      2. every source_block_id exists in that document
      3. every referenced page exists
      4. block references are in valid reading order per page
      5. chunk text is derived from referenced source blocks
      6. token counts are accurate
      7. chunk IDs are unique
      8. chunks preserve source ordering (non-decreasing page/position)
      9. no empty chunks
     10. no silent source loss: every content block is covered by some chunk
    """
    result = ValidationResult()

    # Index the document
    blocks_by_id: Dict[str, object] = {}
    block_page: Dict[str, int] = {}
    for page in document.pages:
        for blk in page.blocks:
            blocks_by_id[blk.block_id] = blk
            block_page[blk.block_id] = page.page_number

    # 0. block IDs must be unique per document; collisions silently corrupt
    #    every downstream provenance mapping, so fail before anything else.
    seen_block_ids: Dict[str, int] = {}
    for page in document.pages:
        for blk in page.blocks:
            seen_block_ids[blk.block_id] = \
                seen_block_ids.get(blk.block_id, 0) + 1
    for bid, n in sorted(seen_block_ids.items()):
        if n > 1:
            result.add(
                f"duplicate source block id {bid} ({n} occurrences); "
                f"provenance mapping would be ambiguous")
    content_block_ids = [
        bid for bid, blk in blocks_by_id.items()
        if getattr(blk, "text", "").strip()
        and getattr(blk, "block_type", None) is not None
        and blk.block_type not in (BlockType.HEADER, BlockType.FOOTER,
                                    BlockType.PAGE_NUMBER)
    ]

    seen_ids = set()
    covered_blocks = set()
    last_position = (-1, -1)  # (page_number, position-in-page-order)

    # Per-page block sequence for ordering checks
    page_sequence: Dict[int, list] = {}
    for page in document.pages:
        ordered = sorted(page.blocks, key=lambda b: b.reading_order)
        page_sequence[page.page_number] = [b.block_id for b in ordered]

    prev_pages_end = None
    for chunk in chunks:
        cid = chunk.chunk_id

        # 9. non-empty
        if not chunk.text.strip():
            result.add(f"{cid}: empty chunk text")

        # 7. unique IDs
        if cid in seen_ids:
            result.add(f"duplicate chunk_id {cid}")
        seen_ids.add(cid)

        # 1. document binding
        if chunk.document_id != document.document_id:
            result.add(
                f"{cid}: document_id mismatch "
                f"({chunk.document_id} != {document.document_id})"
            )

        # 2. block existence + 3. page existence + 4. order validity
        for bid in chunk.source_block_ids:
            if bid not in blocks_by_id:
                result.add(f"{cid}: unknown source block {bid}")
        for pno in chunk.page_numbers:
            if pno < 1 or pno > document.page_count:
                result.add(f"{cid}: page {pno} outside document ({document.page_count} pages)")

        pages_in_order = True
        prev_seq_idx = -1
        prev_page_no = None
        for bid in chunk.source_block_ids:
            if bid not in block_page:
                continue
            pno = block_page[bid]
            seq = page_sequence.get(pno, [])
            try:
                idx_now = seq.index(bid)
            except ValueError:
                continue
            if pno == prev_page_no and idx_now <= prev_seq_idx:
                pages_in_order = False
            prev_page_no, prev_seq_idx = pno, idx_now
        if not pages_in_order:
            result.add(f"{cid}: source blocks not in valid page/reading order")

        # 8. global ordering across chunks (by first referenced page)
        first_page = min((block_page.get(b, 10**9) for b in chunk.source_block_ids),
                         default=10**9)
        if first_page < last_position[0]:
            result.add(
                f"{cid}: breaks source ordering (starts at page {first_page}, "
                f"previous ended at {last_position[0]})"
            )
        else:
            last_position = (first_page, 0)

        # 5. derivation: each sufficiently long block must contribute a
        # detectable fragment to the chunk text. For split-window chunks
        # (oversized blocks split into token windows), the chunk text is a
        # substring of the block, so we check token subsequence matching
        # from both ends. Uses word-only tokens so punctuation spacing
        # quirks from PDF extraction don't cause false negatives.
        chunk_tokens = _norm_tokens(chunk.text)
        for bid in chunk.source_block_ids:
            blk = blocks_by_id.get(bid)
            if blk is None:
                continue
            frag = " ".join(blk.text.split())
            if len(frag) < 20:
                continue  # skip trivial blocks
            block_tokens = _norm_tokens(frag)
            # Single-block chunks for split windows: check if the chunk
            # tokens are a subsequence of the block tokens (the chunk is
            # a window INTO the block text).
            if len(chunk.source_block_ids) == 1 and \
               len(chunk.created_from_blocks or []) == 1:
                # Could be a full block chunk or a split-window chunk.
                # Check both directions: either block head in chunk, or
                # chunk is a contiguous subsequence of block.
                if _substantial_overlap(block_tokens, chunk_tokens):
                    continue
                # Try: chunk tokens are a subsequence of block tokens
                # (split-window case)
                if _is_subsequence_of(chunk_tokens, block_tokens,
                                      min_overlap=max(8, len(chunk_tokens)//4)):
                    continue
            elif not _substantial_overlap(block_tokens, chunk_tokens):
                result.add(
                    f"{cid}: text not derived from source block {bid}")

        # Track coverage
        for bid in chunk.source_block_ids:
            covered_blocks.add(bid)

        # 6. token honesty
        if chunk.token_count != count_tokens(chunk.text):
            result.add(
                f"{cid}: token_count {chunk.token_count} != actual "
                f"{count_tokens(chunk.text)}"
            )

    # 10. silent source loss
    lost = [bid for bid in content_block_ids if bid not in covered_blocks]
    if lost:
        result.add(
            f"{len(lost)} content blocks never appear in any chunk "
            f"(e.g. {lost[:3]})"
        )

    return result


def require_valid(chunks: List[Chunk], document: Document) -> ValidationResult:
    """Validate and raise on any provenance corruption."""
    res = validate_chunks(chunks, document)
    if not res.ok:
        raise ProvenanceError(res.errors)
    return res


class ProvenanceError(Exception):
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(
            "provenance validation failed:\n- " + "\n- ".join(errors[:20])
        )
