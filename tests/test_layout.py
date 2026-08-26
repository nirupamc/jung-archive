from jung_archive.layout.analyzer import LayoutAnalyzer
from jung_archive.models.document import Block, LayoutType


def block(i, x0, y0, x1, y1, text="content"):
    return Block(
        block_id=f"b{i:03d}",
        block_type="UNKNOWN",
        text=text,
        bbox={"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        reading_order=0,
        extraction_method="NATIVE",
        confidence=0.9,
    )


class TestLayoutClassification:
    def setup_method(self):
        self.analyzer = LayoutAnalyzer()

    def test_empty_blocks_unknown(self):
        layout, conf, reason = self.analyzer.detect([], None)
        assert layout == LayoutType.UNKNOWN
        assert 0.0 <= conf <= 1.0

    def test_single_column_full_width_blocks(self):
        # Full-width stacked blocks -> no interior gutter
        blocks = [block(i, 50, 60 + i * 100, 545, 130 + i * 100) for i in range(6)]
        layout, conf, reason = self.analyzer.detect(blocks, None)
        assert layout == LayoutType.SINGLE_COLUMN

    def test_two_column_gutter_detection(self):
        # Alternating left/right column blocks with clear gutter
        blocks = []
        n = 8
        for i in range(n):
            y = 60 + i * 90
            blocks.append(block(2 * i, 45, y, 250, y + 70))       # left col
            blocks.append(block(2 * i + 1, 330, y + 10, 550, y + 80))  # right col
        layout, conf, reason = self.analyzer.detect(blocks, None)
        assert layout == LayoutType.TWO_COLUMN
        assert "gutter" in reason.lower()

    def test_layout_confidence_bounded(self):
        blocks = [block(i, 50, 60 + i * 100, 300, 130 + i * 100) for i in range(4)]
        _, conf, _ = self.analyzer.detect(blocks, None)
        assert 0.0 <= conf <= 1.0

    def test_deterministic_result(self):
        blocks = [
            block(1, 45, 60, 250, 130),
            block(2, 330, 60, 550, 130),
            block(3, 45, 200, 250, 270),
            block(4, 330, 200, 550, 270),
        ]
        r1 = self.analyzer.detect(blocks, None)
        r2 = self.analyzer.detect(blocks, None)
        assert r1 == r2
