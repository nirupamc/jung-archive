import fitz  # PyMuPDF
from collections import Counter
from typing import List
from jung_archive.models.document import Block, BoundingBox, BlockType, ExtractionMethod


class NativeExtractor:
    """
    Extracts text blocks from a PDF page using native PDF text extraction.

    Uses PyMuPDF's structured 'dict' output to preserve per-block font
    metadata (dominant font name/size/flags) for downstream structural
    classification.
    """

    def extract(self, page: fitz.Page) -> List[Block]:
        """
        Extract text blocks with bounding boxes and font metadata.
        Deterministic: blocks are emitted in PDF content order and IDs
        are stable indices within the page.
        """
        blocks: List[Block] = []
        page_dict = page.get_text("dict")

        index = 0
        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:  # 0 = text block
                continue

            # Assemble line texts and collect span statistics
            line_texts = []
            font_sizes: List[float] = []
            font_names: List[str] = []
            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(s.get("text", "") for s in spans)
                if line_text.strip():
                    line_texts.append(line_text)
                for s in spans:
                    if s.get("size"):
                        font_sizes.append(float(s["size"]))
                    if s.get("font"):
                        font_names.append(s["font"])

            text = "\n".join(line_texts).strip()
            if not text:
                continue

            x0, y0, x1, y1 = raw_block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            # Some corpus PDFs report block rects that overflow the page by
            # a few points (font ascender quirks); clamp into the page rect
            # so coordinates stay honest AND valid. Values already inside
            # the page are untouched.
            page_rect = page.rect
            x0 = min(max(float(x0), 0.0), float(page_rect.width))
            y0 = min(max(float(y0), 0.0), float(page_rect.height))
            x1 = min(max(float(x1), 0.0), float(page_rect.width))
            y1 = min(max(float(y1), 0.0), float(page_rect.height))
            if x1 <= x0 or y1 <= y0:
                continue  # degenerate after clamping; no fabricable geometry
            bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

            # Dominant font = most common name; median size is robust to outliers
            font_name = Counter(font_names).most_common(1)[0][0] if font_names else None
            font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else None

            # 1-based page prefix to match all other provenance surfaces
            # (fitz page.number is 0-based).
            block = Block(
                block_id=f"p{page.number + 1:04d}-b{index:03d}",
                block_type=BlockType.UNKNOWN,  # classified by structure analyzer
                text=text,
                bbox=bbox,
                reading_order=0,  # assigned after sorting
                extraction_method=ExtractionMethod.NATIVE,
                # Native PDF text extraction is lossless w.r.t. the embedded
                # text layer, so there is no per-block measured certainty to
                # report; leaving None instead of a fabricated prior.
                confidence=None,
                font_name=font_name,
                font_size=font_size,
                flags=None,
                number=raw_block.get("number"),
            )
            blocks.append(block)
            index += 1

        return blocks
