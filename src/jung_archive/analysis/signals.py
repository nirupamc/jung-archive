import fitz  # PyMuPDF
from typing import NamedTuple

class PageSignals(NamedTuple):
    """
    Signals extracted from a PDF page used for classification.
    """
    text_length: int  # Number of text characters (non-whitespace)
    text_block_count: int  # Number of text blocks
    image_count: int  # Number of images
    image_area_ratio: float  # Ratio of image area to page area
    printable_char_ratio: float  # Ratio of printable characters to total characters
    avg_font_size: float  # Average font size in text blocks
    # Add more signals as needed

class PageSignalExtractor:
    """
    Extracts signals from a PDF page.
    """

    def extract(self, page: fitz.Page) -> PageSignals:
        """
        Extract signals from a given page.
        """
        # Get text and text blocks
        text = page.get_text("text")
        text_blocks = page.get_text("blocks")  # Each block is (x0, y0, x1, y1, text, block_no, block_type)

        # Calculate text length (non-whitespace)
        non_whitespace = sum(1 for c in text if not c.isspace())

        # Count text blocks (those with block_type 0 in PyMuPDF? Actually, block_type 0 is text)
        # In PyMuPDF's get_text("blocks"), the last element is block type: 0 for text, 1 for image, etc.
        text_block_count = sum(1 for block in text_blocks if block[6] == 0) if text_blocks else 0

        # Get images
        image_list = page.get_images()
        image_count = len(image_list)

        # Calculate image area ratio using actual image placements
        page_area = page.rect.width * page.rect.height
        image_area = 0.0
        seen_rects = []
        for img in image_list:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for rect in rects:
                # Avoid double-counting overlapping placements
                overlaps = any(rect.intersects(r) for r in seen_rects)
                if not overlaps:
                    seen_rects.append(rect)
                    inter = rect & page.rect  # clip to page
                    image_area += max(0.0, inter.width) * max(0.0, inter.height)
        image_area_ratio = min(1.0, image_area / page_area) if page_area > 0 else 0.0

        # Printable character ratio (printable vs total)
        printable = sum(1 for c in text if c.isprintable())
        total_chars = len(text)
        printable_char_ratio = printable / total_chars if total_chars > 0 else 0.0

        # Average font size from span data (real measurement, 0.0 when no text)
        avg_font_size = 0.0
        sizes = []
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size")
                    if size and size > 0:
                        sizes.append(size)
        if sizes:
            avg_font_size = sum(sizes) / len(sizes)

        return PageSignals(
            text_length=non_whitespace,
            text_block_count=text_block_count,
            image_count=image_count,
            image_area_ratio=image_area_ratio,
            printable_char_ratio=printable_char_ratio,
            avg_font_size=avg_font_size
        )