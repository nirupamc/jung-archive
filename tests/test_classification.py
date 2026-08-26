import pytest

from jung_archive.analysis.signals import PageSignalExtractor
from jung_archive.analysis.classifier import PageClassifier
from jung_archive.models.document import PageClassification


def classify_page(page):
    signals = PageSignalExtractor().extract(page)
    return PageClassifier().classify(signals)


class TestSyntheticClassification:
    """Classifier behavior on unit-level synthetic signals."""

    def setup_method(self):
        self.classifier = PageClassifier()

    def test_empty_page_classification(self):
        from jung_archive.analysis.signals import PageSignals

        signals = PageSignals(
            text_length=0,
            text_block_count=0,
            image_count=0,
            image_area_ratio=0.0,
            printable_char_ratio=0.0,
            avg_font_size=0.0,
        )
        cls, conf, reason = self.classifier.classify(signals)
        assert cls == PageClassification.EMPTY
        assert 0.0 <= conf <= 1.0
        assert reason

    def test_native_text_page(self):
        from jung_archive.analysis.signals import PageSignals

        signals = PageSignals(
            text_length=2500,
            text_block_count=8,
            image_count=0,
            image_area_ratio=0.0,
            printable_char_ratio=1.0,
            avg_font_size=12.0,
        )
        cls, conf, reason = self.classifier.classify(signals)
        assert cls == PageClassification.NATIVE
        assert conf >= 0.5

    def test_scanned_page_detection(self):
        from jung_archive.analysis.signals import PageSignals

        signals = PageSignals(
            text_length=3,
            text_block_count=0,
            image_count=1,
            image_area_ratio=0.92,
            printable_char_ratio=1.0,
            avg_font_size=0.0,
        )
        cls, conf, _ = self.classifier.classify(signals)
        assert cls == PageClassification.OCR_REQUIRED

    def test_confidence_always_bounded(self):
        from jung_archive.analysis.signals import PageSignals

        candidates = [
            dict(text_length=0, text_block_count=0, image_count=0,
                 image_area_ratio=0.0, printable_char_ratio=0.0, avg_font_size=0.0),
            dict(text_length=5, text_block_count=1, image_count=2,
                 image_area_ratio=0.8, printable_char_ratio=1.0, avg_font_size=10.0),
            dict(text_length=300, text_block_count=4, image_count=1,
                 image_area_ratio=0.3, printable_char_ratio=1.0, avg_font_size=11.0),
            dict(text_length=2000, text_block_count=6, image_count=0,
                 image_area_ratio=0.0, printable_char_ratio=0.99, avg_font_size=12.0),
        ]
        for kwargs in candidates:
            s = PageSignals(**kwargs)
            cls, conf, reason = self.classifier.classify(s)
            assert 0.0 <= conf <= 1.0
            assert isinstance(cls, PageClassification)
            assert len(reason) > 0


class TestGeneratedPdfClassification:
    """End-to-end classification on generated fixture PDFs."""

    def test_blank_pdf_is_empty(self, empty_pdf):
        import fitz

        doc = fitz.open(empty_pdf)
        cls, _, _ = classify_page(doc[0])
        doc.close()
        assert cls == PageClassification.EMPTY

    def test_text_pdf_is_native(self, text_pdf):
        import fitz

        doc = fitz.open(text_pdf)
        for i in range(len(doc)):
            cls, _, _ = classify_page(doc[i])
            assert cls == PageClassification.NATIVE, f"page {i + 1} classified {cls}"
        doc.close()

    def test_scanned_pdf_requires_ocr(self, scanned_pdf):
        import fitz

        doc = fitz.open(scanned_pdf)
        cls, conf, reason = classify_page(doc[0])
        doc.close()
        assert cls in (PageClassification.OCR_REQUIRED, PageClassification.HYBRID)
        if cls == PageClassification.OCR_REQUIRED:
            assert "image" in reason.lower() or "text" in reason.lower()
