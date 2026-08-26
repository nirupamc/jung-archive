import fitz  # PyMuPDF
from typing import List
from jung_archive.models.document import Block, BoundingBox, BlockType, ExtractionMethod


class OCRExtractor:
    """
    Extracts text from PDF pages using OCR (Tesseract).

    OCR is optional and failure-safe: if pytesseract/Tesseract is missing,
    the extractor reports itself unavailable instead of raising.
    """

    def __init__(self):
        self.available = False
        self.unavailable_reason = None
        self.last_error = None
        self.last_mean_confidence = None
        try:
            import pytesseract
            from PIL import Image

            # Verify the Tesseract binary is actually reachable, not just pytesseract
            pytesseract.get_tesseract_version()
            self.pytesseract = pytesseract
            self.Image = Image
            self.available = True
        except Exception as e:
            self.pytesseract = None
            self.Image = None
            self.unavailable_reason = str(e)

    def extract(self, page: fitz.Page) -> List[Block]:
        """
        Extract text from a page using OCR.

        Uses word-level OCR data so that both the text and a MEASURED mean
        word confidence can be reported. Returns a single page-covering
        block (conservative M1 approach), or an empty list when OCR is
        unavailable or fails. Failures are recorded on the extractor via
        last_error rather than fabricated text.
        """
        self.last_error = None
        self.last_mean_confidence = None
        if not self.available:
            self.last_error = f"OCR backend unavailable: {self.unavailable_reason}"
            return []

        try:
            mat = fitz.Matrix(2, 2)  # 2x zoom for OCR quality
            pix = page.get_pixmap(matrix=mat)
            img = self.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            data = self.pytesseract.image_to_data(
                img, output_type=self.pytesseract.Output.DICT
            )
        except Exception as e:
            self.last_error = f"OCR failed: {e}"
            return []

        words = []
        confidences = []
        n = len(data["text"])
        for i in range(n):
            raw = (data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not raw or conf < 0:
                continue
            words.append(raw)
            confidences.append(conf)

        if not words:
            self.last_error = "OCR produced no text"
            return []

        text = " ".join(words)
        self.last_mean_confidence = sum(confidences) / len(confidences) / 100.0

        bbox = BoundingBox(x0=0.0, y0=0.0, x1=page.rect.width, y1=page.rect.height)
        block = Block(
            # Page-scoped ID: provenance requires block IDs to be unique
            # within the whole document, not just within one page.
            block_id=f"p{page.number + 1:04d}-o000",
            block_type=BlockType.UNKNOWN,  # classified later by structure analyzer
            text=text,
            bbox=bbox,
            reading_order=0,  # assigned later
            extraction_method=ExtractionMethod.OCR,
            confidence=round(self.last_mean_confidence, 4),  # measured, not prior
        )
        return [block]
