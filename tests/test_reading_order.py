from jung_archive.ingestion.pdf import PDFIngestor
from jung_archive.models.document import (
    Block,
    BoundingBox,
    LayoutType,
)


def make_block(i, x0, y0, x1, y1):
    return Block(
        block_id=f"b{i}",
        block_type="UNKNOWN",
        text=f"block {i}",
        bbox={"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        reading_order=0,
        extraction_method="NATIVE",
        confidence=0.9,
    )


class TestReadingOrder:
    def test_single_column_top_to_bottom(self):
        blocks = [
            make_block(3, 50, 400, 545, 470),
            make_block(1, 50, 100, 545, 170),
            make_block(2, 50, 250, 545, 320),
        ]
        ordered = PDFIngestor._assign_reading_order(blocks, 595.0, LayoutType.SINGLE_COLUMN)
        assert [b.block_id for b in ordered] == ["b1", "b2", "b3"]
        assert [b.reading_order for b in ordered] == [1, 2, 3]

    def test_two_column_left_then_right(self):
        # Left column rows interleaved in creation order with right column
        blocks = [
            make_block("r1", 330, 60, 550, 130),   # right col row 1
            make_block("l1", 45, 60, 250, 130),    # left col row 1
            make_block("r2", 330, 200, 550, 270),  # right col row 2
            make_block("l2", 45, 200, 250, 270),   # left col row 2
        ]
        ordered = PDFIngestor._assign_reading_order(blocks, 595.0, LayoutType.TWO_COLUMN)
        ids = [b.block_id for b in ordered]
        assert ids == ["bl1", "bl2", "br1", "br2"]
        assert [b.reading_order for b in ordered] == [1, 2, 3, 4]

    def test_deterministic_across_runs(self):
        blocks = [
            make_block(9, 50, 700, 545, 770),
            make_block(4, 50, 100, 545, 170),
            make_block(7, 50, 300, 545, 370),
        ]
        a = PDFIngestor._assign_reading_order([*blocks], 595.0, LayoutType.SINGLE_COLUMN)
        b = PDFIngestor._assign_reading_order(
            [
                make_block(9, 50, 700, 545, 770),
                make_block(4, 50, 100, 545, 170),
                make_block(7, 50, 300, 545, 370),
            ],
            595.0,
            LayoutType.SINGLE_COLUMN,
        )
        assert [(x.block_id, x.reading_order) for x in a] == [
            (x.block_id, x.reading_order) for x in b
        ]

    def test_reading_orders_are_sequential_from_one(self):
        blocks = [make_block(i, 50 + i * 10, 60 + i * 90, 500 + i * 10, 130 + i * 90)
                  for i in range(6)]
        ordered = PDFIngestor._assign_reading_order(blocks, 595.0, LayoutType.MIXED)
        assert sorted(b.reading_order for b in ordered) == list(range(1, 7))
