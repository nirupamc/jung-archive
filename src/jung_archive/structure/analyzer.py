import re
import statistics
from typing import List, Optional
from jung_archive.models.document import Block, BlockType

_BULLET_RE = re.compile(r"^\s*([-•*·—–]|\(?\d{1,2}[.)]|[a-z][.)]|i{1,3}[.)])\s+", re.IGNORECASE)
_DIGITS_ONLY_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE)
_ROMAN_RE = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)


class StructureAnalyzer:
    """
    Heuristic block-type classification for M1.

    Evidence used, in priority order:
      1. Position: header/footer/page-number strips at page edges
      2. Font size relative to the page's median body size
      3. Shape: line count, character count
      4. Content: bullet/numbering patterns

    Conservative by design: when evidence is weak the block stays
    PARAGRAPH only with clear body-text evidence, otherwise UNKNOWN.
    """

    def analyze(self, blocks: List[Block], page) -> List[Block]:
        """Classify each block's type; returns blocks sorted top-to-bottom."""
        ordered = sorted(blocks, key=lambda b: (round(b.bbox.y0, 1), round(b.bbox.x0, 1)))
        if not ordered:
            return ordered

        body_size = self._median_body_size(ordered)
        page_height = page.rect.height if page is not None else max(b.bbox.y1 for b in ordered)

        for block in ordered:
            block.block_type = self._classify(block, body_size, page_height)
            block.heuristic_quality_score = self._quality_score(
                block.block_type, block, body_size, page_height
            )
        return ordered

    @staticmethod
    def _quality_score(
        block_type: BlockType, block: Block, body_size: Optional[float], page_height: float
    ) -> Optional[float]:
        """Explicitly HEURISTIC evidence rating for the type assignment.

        This is NOT extraction certainty; it grades how strong the
        structural evidence behind the label was. None = no meaningful
        evidence (UNKNOWN).
        """
        if block_type == BlockType.UNKNOWN:
            return None
        text = block.text.strip()
        bbox = block.bbox
        # Positional edge-strip evidence is strong
        in_top_strip = bbox.y1 <= page_height * 0.08
        in_bottom_strip = bbox.y0 >= page_height * 0.93
        if in_top_strip or in_bottom_strip:
            return 0.9
        size_ratio = (
            block.font_size / body_size
            if block.font_size and body_size and body_size > 0
            else None
        )
        if block_type in (BlockType.TITLE, BlockType.HEADING):
            return 0.9 if (size_ratio is not None and size_ratio >= 1.35) else 0.7
        if block_type == BlockType.LIST:
            return 0.85  # bullet pattern is a strong signal
        if block_type == BlockType.CAPTION:
            return 0.6  # requires keyword + small text, still weakish
        if block_type == BlockType.PARAGRAPH:
            multi_line = len([ln for ln in block.text.splitlines() if ln.strip()]) >= 2
            return 0.7 if multi_line or len(text) >= 200 else 0.5
        return 0.5

    @staticmethod
    def _median_body_size(blocks: List[Block]) -> Optional[float]:
        """Median font size weighted toward larger blocks (body text dominates)."""
        sizes = []
        for b in blocks:
            if b.font_size:
                weight = max(1, len(b.text) // 100)  # long blocks count more
                sizes.extend([b.font_size] * min(weight, 5))
        return statistics.median(sizes) if sizes else None

    def _classify(self, block: Block, body_size: Optional[float], page_height: float) -> BlockType:
        text = block.text.strip()
        lines = [ln for ln in block.text.splitlines() if ln.strip()]
        n_lines = len(lines)
        n_chars = len(text)
        bbox = block.bbox

        # --- 1. Edge strips: header / footer / page number ---
        in_top_strip = bbox.y1 <= page_height * 0.08
        in_bottom_strip = bbox.y0 >= page_height * 0.93
        if in_top_strip or in_bottom_strip:
            if _DIGITS_ONLY_RE.match(text) or (_ROMAN_RE.match(text) and n_chars <= 8):
                return BlockType.PAGE_NUMBER
            if n_chars <= 80:
                return BlockType.FOOTER if in_bottom_strip else BlockType.HEADER
            return BlockType.UNKNOWN

        # --- 2. Font-size evidence vs page median body size ---
        size_ratio = None
        if block.font_size and body_size:
            size_ratio = block.font_size / body_size

        # --- 3. Content patterns ---
        bulleted = sum(1 for ln in lines if _BULLET_RE.match(ln))
        mostly_list = n_lines > 0 and bulleted >= max(1, len(lines) - 1) and bulleted >= 2
        if mostly_list:
            return BlockType.LIST

        # --- Title/heading via size + brevity + position ---
        near_top = bbox.y0 < page_height * 0.18
        large_text = size_ratio is not None and size_ratio >= 1.35
        heading_sized = size_ratio is not None and size_ratio >= 1.15

        if large_text and n_chars <= 150 and n_lines <= 3:
            return BlockType.TITLE if near_top else BlockType.HEADING
        if heading_sized and n_chars <= 120 and n_lines <= 2:
            return BlockType.HEADING

        # Caption: short line(s) adjacent-to-figure-sized gap cannot be proven
        # here; require small text and brevity as weak evidence.
        small_text = size_ratio is not None and size_ratio <= 0.85
        if small_text and n_chars <= 200 and n_lines <= 2 and re.search(
            r"(figure|fig\.|plate|table)\s*\d*", text, re.IGNORECASE
        ):
            return BlockType.CAPTION

        # --- 4. Body paragraph evidence ---
        ends_sentence = bool(re.search(r"[.!?:;\u201d\u2019'\"]\s*$", text))
        if n_lines >= 2 or ends_sentence or (n_chars >= 200):
            return BlockType.PARAGRAPH
        # Body-sized text of reasonable length is treated as a paragraph
        # even without terminal punctuation (common in fragmented native
        # block segmentation); anything else stays UNKNOWN.
        body_sized = size_ratio is not None and 0.85 < size_ratio < 1.15
        if body_sized and n_chars >= 40:
            return BlockType.PARAGRAPH
        if size_ratio is None and n_chars >= 40:
            # No font metadata (OCR path): moderate evidence
            return BlockType.PARAGRAPH

        return BlockType.UNKNOWN
