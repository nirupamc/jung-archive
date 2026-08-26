import statistics
from typing import List, Optional, Tuple
from jung_archive.models.document import Block, LayoutType


class _Rect:
    """Lightweight internal rectangle for geometry analysis."""

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def width(self) -> float:
        return self.x1 - self.x0


class LayoutAnalyzer:
    """
    Detects page layout from line geometry using an x-axis occupancy
    profile (gutter detection).

    Method:
      1. Collect line rectangles (native PDF lines when available,
         otherwise block bounding boxes).
      2. Bin the horizontal extent of the text area.
      3. A "gutter" is an interior band covered by very few lines while
         both flanking bands are well covered.
      4. Lines crossing the gutter are counted; many crossings on a
         guttered page indicate MIXED rather than strict TWO_COLUMN.
    """

    N_BINS = 50
    GUTTER_MAX_COVERAGE = 0.25   # bin coverage below this = gutter candidate
    FLANK_MIN_COVERAGE = 0.45    # flanking bins must exceed this
    MIN_GUTTER_BINS = 2          # gutter must be >= 4% of text width
    CROSS_FRACTION_MIXED = 0.25  # crossing lines above this -> MIXED
    SIDE_MIN_FRACTION = 0.15     # each side needs this share of lines

    def detect(
        self,
        blocks: List[Block],
        page_rect=None,
        page=None,
    ) -> Tuple[LayoutType, float, str]:
        if not blocks:
            return LayoutType.UNKNOWN, 0.0, "no blocks to analyze"

        rects = self._collect_line_rects(blocks, page)
        if len(rects) < 3:
            # Too little geometry for column inference
            return LayoutType.UNKNOWN, 0.3, f"insufficient geometry ({len(rects)} lines)"

        xs0 = [r.x0 for r in rects]
        xs1 = [r.x1 for r in rects]
        area_x0, area_x1 = min(xs0), max(xs1)
        area_w = area_x1 - area_x0
        if area_w <= 1.0:
            return LayoutType.UNKNOWN, 0.3, "degenerate text width"

        nbins = self.N_BINS
        coverage = [0.0] * nbins
        for r in rects:
            b0 = int((r.x0 - area_x0) / area_w * nbins)
            b1 = int((r.x1 - area_x0) / area_w * nbins)
            b0 = max(0, min(nbins - 1, b0))
            b1 = max(0, min(nbins - 1, b1))
            for b in range(b0, b1 + 1):
                coverage[b] += 1.0
        n = len(rects)
        coverage = [c / n for c in coverage]

        gutter = self._find_gutter(coverage)
        if gutter is None:
            return LayoutType.SINGLE_COLUMN, 0.85, "no interior gutter in x-occupancy profile"

        g0, g1 = gutter
        mid_left = area_x0 + g1 * area_w / nbins  # gutter right edge
        mid_right_of_gutter_start = area_x0 + g0 * area_w / nbins

        left_lines = sum(1 for r in rects if r.x1 <= mid_right_of_gutter_start)
        right_lines = sum(1 for r in rects if r.x0 >= mid_left)
        crossing = n - sum(1 for r in rects if r.x1 <= mid_right_of_gutter_start or r.x0 >= mid_left)

        left_frac = left_lines / n
        right_frac = right_lines / n
        cross_frac = crossing / n

        balanced = left_frac >= self.SIDE_MIN_FRACTION and right_frac >= self.SIDE_MIN_FRACTION
        if not balanced:
            return (
                LayoutType.MIXED,
                0.6,
                f"gutter present but unbalanced columns (L={left_frac:.2f}, R={right_frac:.2f})",
            )
        if cross_frac >= self.CROSS_FRACTION_MIXED:
            return (
                LayoutType.MIXED,
                0.7,
                f"column structure with {cross_frac:.0%} full-width lines",
            )

        conf = min(0.95, 0.6 + (1.0 - max(coverage[g0:g1])) * 0.35)
        return (
            LayoutType.TWO_COLUMN,
            round(conf, 2),
            f"interior gutter bins {g0}-{g1}/{nbins}; L={left_frac:.0%} R={right_frac:.0%} "
            f"cross={cross_frac:.0%}",
        )

    def _find_gutter(self, coverage: List[float]) -> Optional[Tuple[int, int]]:
        nbins = len(coverage)
        best: Optional[Tuple[int, int]] = None
        i = 1  # skip extreme left bin (margins cause noise)
        while i < nbins - 1:
            if coverage[i] <= self.GUTTER_MAX_COVERAGE:
                j = i
                while j + 1 < nbins - 1 and coverage[j + 1] <= self.GUTTER_MAX_COVERAGE:
                    j += 1
                width = j - i + 1
                left_flank_ok = any(c >= self.FLANK_MIN_COVERAGE for c in coverage[max(0, i - 10):i])
                right_flank_ok = any(
                    c >= self.FLANK_MIN_COVERAGE for c in coverage[j + 1:j + 11]
                )
                if (
                    width >= self.MIN_GUTTER_BINS
                    and i > 0
                    and j < nbins - 1
                    and left_flank_ok
                    and right_flank_ok
                ):
                    if best is None or width > (best[1] - best[0] + 1):
                        best = (i, j)
                i = j + 1
            else:
                i += 1
        return best

    def _collect_line_rects(self, blocks: List[Block], page) -> List[_Rect]:
        """Prefer true line geometry from the PDF page; fall back to blocks."""
        if page is not None:
            try:
                rects = []
                d = page.get_text("dict")
                for b in d.get("blocks", []):
                    if b.get("type") != 0:
                        continue
                    for ln in b.get("lines", []):
                        x0, y0, x1, y1 = ln["bbox"]
                        if x1 > x0 and y1 > y0:
                            rects.append(_Rect(x0, y0, x1, y1))
                if rects:
                    return rects
            except Exception:
                pass
        return [_Rect(b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1) for b in blocks]
