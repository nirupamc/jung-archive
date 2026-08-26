from jung_archive.models.document import PageClassification
from jung_archive.analysis.signals import PageSignals, PageSignalExtractor
from typing import Tuple

class PageClassifier:
    """
    Classifies a page as NATIVE, OCR_REQUIRED, HYBRID, EMPTY, SUSPICIOUS, or FAILED
    based on page signals.
    """

    # Thresholds for classification
    EMPTY_TEXT_LENGTH = 20  # Pages with less than this much text are considered empty
    OCR_THRESHOLD_TEXT = 1000  # Low text but high image area suggests OCR
    OCR_IMAGE_COVERAGE = 0.5  # If images cover more than this ratio, need OCR

    def classify(self, signals: PageSignals) -> Tuple[PageClassification, float, str]:
        """
        Classify a page based on its signals.
        Returns (classification, confidence, reason).
        """
        # Check for empty page
        if signals.text_length == 0 and signals.image_count == 0:
            return PageClassification.EMPTY, 1.0, "no text and no images detected"

        # Check for very low text with images - likely scanned/OCR needed
        if signals.text_length < self.EMPTY_TEXT_LENGTH:
            if signals.image_count > 0:
                # This page has images but very little text
                return PageClassification.OCR_REQUIRED, 0.9, "very low native text with images"
            return PageClassification.EMPTY, 0.8, "very low text volume, no images"

        # Check for suspiciously low text density
        if signals.text_block_count == 0 and signals.text_length > 0:
            return PageClassification.SUSPICIOUS, 0.7, "text detected but no text blocks"

        # Check for hybrid pages (text + significant images)
        if signals.image_count > 0 and signals.image_area_ratio > 0.1:
            # Has images covering at least 10% of the page
            if signals.text_length < 500:  # But not much text
                return PageClassification.HYBRID, 0.85, "mixed text and images"

        # Default to native
        return PageClassification.NATIVE, 0.95, "sufficient native text detected"