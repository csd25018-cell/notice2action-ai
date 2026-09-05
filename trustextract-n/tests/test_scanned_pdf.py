"""
Test Suite for Scanned PDF Pipeline — TrustExtract-N
=====================================================
Tests the complete pipeline against scanned/image-based PDFs.
Includes two mandatory demonstration scenarios:

DEMO 1: High-quality notification → expect HIGH CONFIDENCE
DEMO 2: Low-quality/ambiguous scan → expect mixed results with abstention

Also tests individual components:
- PDF type detection
- OCR quality scoring
- Bounding box propagation
- Multi-date extraction and classification
- Dual-text preservation
- Image preprocessing
- Layout reconstruction
"""

import os
import json
import tempfile
import pytest
from pathlib import Path

# Test utilities
try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz as pymupdf
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

from src.preprocessing.pdf_parser import PDFParser
from src.preprocessing.ocr import OCREngine
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.ocr_quality import OCRQualityScorer
from src.preprocessing.layout import LayoutReconstructor
from src.inference.date_classifier import DateClassifier
from src.inference.confidence import estimate_field_confidence, get_ocr_confidence_for_span
from src.inference.consistency import run_consistency_checks, check_field_consistency
from src.inference.evidence import extract_evidence


# ========================================================
# Helper: Create a synthetic scanned PDF (image-only)
# ========================================================

def create_scanned_pdf(output_path: str, text: str = None):
    """
    Creates a synthetic scanned PDF by rendering text to an image
    and embedding it as an image-only page (no text layer).
    """
    if not PYMUPDF_AVAILABLE:
        pytest.skip("PyMuPDF not available for test PDF creation")

    if text is None:
        text = """GOVERNMENT OF INDIA
MINISTRY OF HEALTH AND FAMILY WELFARE
(Department of Health and Family Welfare)

NOTIFICATION

New Delhi, the 19th August, 2026

G.S.R. 745(E).— The following draft of certain rules further to amend the
Drugs Rules, 1945, which the Central Government proposes to make, in exercise
of the powers conferred by sub-section (1) of section 12 and sub-section (1) of
section 33 of the Drugs and Cosmetics Act, 1940 (23 of 1940).

Objections and suggestions may be addressed to the Under Secretary (Drugs),
Ministry of Health and Family Welfare, Government of India,
Kartavya Bhawan-1, New Delhi, 110001 or emailed at drugsdiv-mohfw@gov.in.

Last date for submission: 19th September, 2026

Eligible persons: All registered drug manufacturers, distributors, and
pharmacy practitioners holding valid licenses.

Required documents:
1. Copy of drug manufacturing license
2. Registration certificate
3. Identity proof of authorized signatory

Sd/-
(Dr. R.K. Sharma)
Under Secretary to the Government of India"""

    doc = pymupdf.open()
    # Create a text page first, then render it to image, then embed as image-only
    page = doc.new_page(width=595, height=842)  # A4 size

    # Write text on page
    text_rect = pymupdf.Rect(50, 50, 545, 792)
    page.insert_textbox(text_rect, text, fontsize=11, fontname="helv")

    # Render to pixmap
    pix = page.get_pixmap(dpi=200)

    # Create new document with image-only page
    doc2 = pymupdf.open()
    img_page = doc2.new_page(width=pix.width, height=pix.height)

    # Insert the rendered image (this creates a scanned-like PDF with no text layer)
    img_page.insert_image(img_page.rect, pixmap=pix)

    doc2.save(output_path)
    doc2.close()
    doc.close()


def create_digital_pdf(output_path: str, text: str = None):
    """
    Creates a digital (machine-readable text) PDF.
    """
    if not PYMUPDF_AVAILABLE:
        pytest.skip("PyMuPDF not available for test PDF creation")

    if text is None:
        text = """TEZPUR UNIVERSITY
(A Central University established by an Act of Parliament)

NOTIFICATION

No. TU/Exam/2026/1234                              Dated: 5th September, 2026

Subject: Notification regarding examination schedule for Autumn Semester 2026

All students of Tezpur University are hereby notified that the end-semester
examinations for the Autumn Semester 2026 will commence from 15th November, 2026.

The last date for submission of examination forms is 30th September, 2026.

Contact: Controller of Examinations, Tezpur University
Phone: 03712-275001
Email: coe@tezu.ernet.in

Sd/-
Controller of Examinations
Tezpur University"""

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    text_rect = pymupdf.Rect(50, 50, 545, 792)
    page.insert_textbox(text_rect, text, fontsize=11, fontname="helv")
    doc.save(output_path)
    doc.close()


# ========================================================
# Test: PDF Type Detection
# ========================================================

class TestPDFTypeDetection:
    """Tests that the PDF parser correctly classifies pages as TEXT, SCANNED, or MIXED."""

    def test_digital_pdf_classified_as_text(self, tmp_path):
        """A digital PDF with embedded text should be classified as TEXT."""
        pdf_path = str(tmp_path / "digital.pdf")
        create_digital_pdf(pdf_path)

        result = PDFParser.parse_pdf(pdf_path)
        assert result["document_type_summary"] == "TEXT"
        for page in result["pages"]:
            assert page["page_type"] == "TEXT"
            assert page["needs_ocr"] is False

    def test_scanned_pdf_classified_as_scanned(self, tmp_path):
        """A scanned (image-only) PDF should be classified as SCANNED."""
        pdf_path = str(tmp_path / "scanned.pdf")
        create_scanned_pdf(pdf_path)

        result = PDFParser.parse_pdf(pdf_path)
        assert result["document_type_summary"] in ("SCANNED", "MIXED")
        # At least one page should need OCR
        assert any(p["needs_ocr"] for p in result["pages"])

    def test_page_analysis_fields_present(self, tmp_path):
        """Each page should have type analysis fields."""
        pdf_path = str(tmp_path / "test.pdf")
        create_digital_pdf(pdf_path)

        result = PDFParser.parse_pdf(pdf_path)
        page = result["pages"][0]
        assert "page_type" in page
        assert "has_images" in page
        assert "has_fonts" in page
        assert "image_count" in page
        assert "text_quality" in page
        assert "needs_ocr" in page


# ========================================================
# Test: OCR Engine
# ========================================================

class TestOCREngine:
    """Tests the OCR engine produces structured output with bounding boxes."""

    def test_ocr_engine_initialization(self):
        """Test OCR engine initializes with correct defaults."""
        engine = OCREngine(lang="en", dpi=250)
        assert engine.lang == "en"
        assert engine.dpi == 250
        assert engine.enable_preprocessing is True

    def test_structured_ocr_output_format(self):
        """OCR output should contain text, words, avg_confidence, word_count."""
        from PIL import Image
        # Create a simple test image with text
        img = Image.new("RGB", (400, 100), "white")
        engine = OCREngine(lang="en", dpi=250)
        result = engine.ocr_image_structured(img, preprocess=False)

        assert "text" in result
        assert "words" in result
        assert "avg_confidence" in result
        assert "word_count" in result
        assert "engine" in result
        assert isinstance(result["words"], list)

    def test_process_document_preserves_structure(self, tmp_path):
        """Document OCR should preserve page structure and add OCR data."""
        pdf_path = str(tmp_path / "test.pdf")
        create_digital_pdf(pdf_path)

        doc = PDFParser.parse_pdf(pdf_path)
        engine = OCREngine(lang="en", dpi=250)
        result = engine.process_document_ocr(doc, pdf_path)

        assert "pages" in result
        assert "full_text" in result
        assert "ocr_quality" in result
        assert "ocr_language" in result

        for page in result["pages"]:
            assert "page_number" in page
            assert "original_ocr_text" in page
            assert "ocr_quality" in page

    def test_legacy_compatibility(self):
        """process_document_ocr_selective should still work (backward compat)."""
        engine = OCREngine()
        doc = {
            "document_id": "TEST",
            "file_path": "nonexistent.pdf",
            "pages": [
                {"page_number": 1, "text": "Test text " * 20, "source": "pdf_text", "page_type": "TEXT"}
            ],
            "full_text": "Test text " * 20,
        }
        result = engine.process_document_ocr_selective(doc)
        assert "pages" in result


# ========================================================
# Test: Image Preprocessor
# ========================================================

class TestImagePreprocessor:
    """Tests image preprocessing operations."""

    def test_preprocessor_initialization(self):
        preprocessor = ImagePreprocessor()
        assert preprocessor.deskew is True
        assert preprocessor.denoise is True
        assert preprocessor.target_dpi == 250

    def test_preprocess_returns_image_and_metadata(self):
        from PIL import Image
        img = Image.new("RGB", (200, 300), "white")
        preprocessor = ImagePreprocessor()
        result_img, metadata = preprocessor.preprocess(img)

        assert isinstance(result_img, Image.Image)
        assert "original_size" in metadata
        assert "operations_applied" in metadata
        assert isinstance(metadata["operations_applied"], list)

    def test_preprocess_for_ocr_convenience(self):
        from PIL import Image
        img = Image.new("RGB", (200, 300), "white")
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess_for_ocr(img)
        assert isinstance(result, Image.Image)


# ========================================================
# Test: OCR Quality Scorer
# ========================================================

class TestOCRQualityScorer:
    """Tests OCR quality scoring."""

    def test_empty_words_returns_zero(self):
        result = OCRQualityScorer.score_page([])
        assert result["ocr_quality_score"] == 0.0
        assert result["quality_label"] == "NO_TEXT"

    def test_high_quality_words(self):
        words = [
            {"text": "MINISTRY", "confidence": 0.98, "bbox": [10, 10, 100, 30]},
            {"text": "OF", "confidence": 0.99, "bbox": [110, 10, 130, 30]},
            {"text": "HEALTH", "confidence": 0.97, "bbox": [140, 10, 200, 30]},
            {"text": "AND", "confidence": 0.95, "bbox": [210, 10, 240, 30]},
            {"text": "FAMILY", "confidence": 0.96, "bbox": [250, 10, 310, 30]},
            {"text": "WELFARE", "confidence": 0.94, "bbox": [320, 10, 400, 30]},
        ]
        result = OCRQualityScorer.score_page(words, page_width=800, page_height=1200)
        assert result["avg_confidence"] > 0.90
        assert result["token_validity_ratio"] > 0.80
        assert result["quality_label"] in ("GOOD", "MODERATE")

    def test_low_quality_words(self):
        words = [
            {"text": "x!@#", "confidence": 0.3, "bbox": [10, 10, 50, 30]},
            {"text": "$$%", "confidence": 0.2, "bbox": [60, 10, 90, 30]},
            {"text": "??", "confidence": 0.1, "bbox": [100, 10, 120, 30]},
        ]
        result = OCRQualityScorer.score_page(words, page_width=800, page_height=1200)
        assert result["avg_confidence"] < 0.5
        assert result["quality_label"] in ("POOR", "VERY_POOR")

    def test_document_level_scoring(self):
        page_scores = [
            {"ocr_quality_score": 0.9, "word_count": 100, "quality_label": "GOOD", "language_detected": "en"},
            {"ocr_quality_score": 0.5, "word_count": 50, "quality_label": "POOR", "language_detected": "en"},
        ]
        result = OCRQualityScorer.score_document(page_scores)
        assert 0.0 <= result["ocr_confidence"] <= 1.0
        assert result["language"] == "en"
        assert result["page_count"] == 2


# ========================================================
# Test: Layout Reconstruction
# ========================================================

class TestLayoutReconstructor:
    """Tests layout reconstruction from OCR words."""

    def test_empty_words(self):
        recon = LayoutReconstructor()
        result = recon.reconstruct([], page_width=800, page_height=1200)
        assert result["lines"] == []
        assert result["paragraphs"] == []

    def test_words_grouped_into_lines(self):
        words = [
            {"text": "Hello", "bbox": [10, 10, 60, 30], "confidence": 0.9},
            {"text": "World", "bbox": [70, 12, 130, 32], "confidence": 0.9},
            {"text": "Next", "bbox": [10, 50, 50, 70], "confidence": 0.9},
            {"text": "Line", "bbox": [60, 52, 100, 72], "confidence": 0.9},
        ]
        recon = LayoutReconstructor()
        result = recon.reconstruct(words, page_width=400, page_height=300)
        assert len(result["lines"]) >= 2

    def test_zones_detected(self):
        words = [
            # Header zone (top 12%)
            {"text": "HEADER", "bbox": [10, 5, 100, 20], "confidence": 0.9},
            # Body zone
            {"text": "Body", "bbox": [10, 200, 80, 220], "confidence": 0.9},
            {"text": "text", "bbox": [90, 200, 140, 220], "confidence": 0.9},
            # Footer zone (bottom 10%)
            {"text": "Page", "bbox": [10, 550, 50, 570], "confidence": 0.9},
            {"text": "1", "bbox": [55, 550, 65, 570], "confidence": 0.9},
        ]
        recon = LayoutReconstructor()
        result = recon.reconstruct(words, page_width=400, page_height=600)
        assert "zones" in result
        assert "header" in result["zones"]
        assert "footer" in result["zones"]


# ========================================================
# Test: Text Cleaner — Dual Text Preservation
# ========================================================

class TestTextCleanerDualText:
    """Tests that text cleaning preserves original OCR text."""

    def test_original_text_preserved(self):
        doc = {
            "pages": [
                {
                    "page_number": 1,
                    "text": "  GOVERNMENT   OF  INDIA  \n\n\nPage 1 of 3\n\n  NOTIFICATION  ",
                    "original_ocr_text": "  GOVERNMENT   OF  INDIA  \n\n\nPage 1 of 3\n\n  NOTIFICATION  ",
                    "source": "ocr_paddleocr",
                    "page_type": "SCANNED",
                    "words": [{"text": "GOVERNMENT", "bbox": [10, 10, 100, 30], "confidence": 0.95}],
                    "ocr_quality": {"ocr_quality_score": 0.9},
                }
            ]
        }
        result = TextCleaner.clean_document_object(doc)

        page = result["pages"][0]
        # Cleaned text should have normalized whitespace and removed page artifacts
        assert "Page 1 of 3" not in page["text"]
        assert "GOVERNMENT" in page["text"]

        # Original OCR text should be untouched
        assert "Page 1 of 3" in page["original_ocr_text"]
        assert "  GOVERNMENT   OF  INDIA  " in page["original_ocr_text"]

        # Words should be preserved through cleaning
        assert len(page["words"]) > 0

    def test_document_level_original_text(self):
        doc = {
            "pages": [
                {
                    "page_number": 1,
                    "text": "Original  text  here",
                    "original_ocr_text": "Original  text  here",
                    "source": "ocr_paddleocr",
                }
            ]
        }
        result = TextCleaner.clean_document_object(doc)
        assert "original_text" in result
        assert "Original  text  here" in result["original_text"]

    def test_entity_preservation(self):
        """Dates, emails, phones, DIN numbers must NOT be destroyed."""
        raw = """DIN-2026-993821-GST
Date: 15-09-2026
Email: contact@tax.gov.in
Phone: +91-9876543210
Page 1 of 3"""
        cleaned = TextCleaner.clean_text(raw)
        assert "DIN-2026-993821-GST" in cleaned
        assert "15-09-2026" in cleaned
        assert "contact@tax.gov.in" in cleaned
        assert "+91-9876543210" in cleaned
        assert "Page 1 of 3" not in cleaned


# ========================================================
# Test: Date Classification
# ========================================================

class TestDateClassifier:
    """Tests contextual date classification."""

    def test_issue_date_detection(self):
        text = "New Delhi, the 19th August, 2026\nG.S.R. 745(E)"
        dates = DateClassifier.extract_and_classify_dates(text)
        assert len(dates) >= 1
        assert any(d["type"] == "ISSUE_DATE" for d in dates)

    def test_deadline_detection(self):
        text = "The notice is dated 1st January, 2026. Last date for submission: 15th February, 2026."
        dates = DateClassifier.extract_and_classify_dates(text)
        assert len(dates) >= 2
        deadlines = [d for d in dates if d["type"] == "DEADLINE"]
        assert len(deadlines) >= 1

    def test_multiple_dates_preserved(self):
        text = """Dated: 5th September, 2026
Application start date: 10th September, 2026
Last date for application: 30th September, 2026
Examination commences: 15th November, 2026"""
        dates = DateClassifier.extract_and_classify_dates(text)
        assert len(dates) >= 3  # At least 3 dates should be found

    def test_hindi_date_support(self):
        text = "दिनांक: 15 अगस्त, 2026 को यह अधिसूचना जारी की गई।"
        dates = DateClassifier.extract_and_classify_dates(text)
        # Should find the Hindi date
        assert len(dates) >= 1

    def test_no_phantom_dates(self):
        text = "This document contains no dates whatsoever."
        dates = DateClassifier.extract_and_classify_dates(text)
        assert len(dates) == 0


# ========================================================
# Test: Multi-Signal Confidence
# ========================================================

class TestMultiSignalConfidence:
    """Tests the multi-signal confidence estimation."""

    def test_high_confidence_all_signals_strong(self):
        result = estimate_field_confidence(
            label="AUTHORITY",
            text="Ministry of Health and Family Welfare",
            token_probabilities=[0.95, 0.93, 0.96, 0.91, 0.94, 0.92],
            ocr_word_confidences=[0.98, 0.97, 0.99, 0.96, 0.95, 0.97],
            evidence_available=True,
            evidence_span_complete=True,
            consistency_ok=True,
            page_ocr_quality=0.95,
        )
        assert result["confidence"] > 0.85
        assert result["metrics"]["c_model"] > 0.90
        assert result["metrics"]["c_ocr"] > 0.90

    def test_low_confidence_poor_ocr(self):
        result = estimate_field_confidence(
            label="AUTHORITY",
            text="M!n1stry 0f H3alth",
            token_probabilities=[0.7, 0.6, 0.5],
            ocr_word_confidences=[0.3, 0.2, 0.4],
            evidence_available=True,
            evidence_span_complete=False,
            consistency_ok=False,
            page_ocr_quality=0.3,
        )
        assert result["confidence"] < 0.7

    def test_no_evidence_reduces_confidence(self):
        with_evidence = estimate_field_confidence(
            label="TITLE",
            text="Notification",
            token_probabilities=[0.9],
            evidence_available=True,
            evidence_span_complete=True,
        )
        without_evidence = estimate_field_confidence(
            label="TITLE",
            text="Notification",
            token_probabilities=[0.9],
            evidence_available=False,
            evidence_span_complete=False,
        )
        assert with_evidence["confidence"] > without_evidence["confidence"]


# ========================================================
# Test: Consistency Checks
# ========================================================

class TestConsistencyChecks:
    """Tests enhanced consistency checking."""

    def test_authority_copy_to_conflict(self):
        text = """MINISTRY OF HOME AFFAIRS
Notification dated 1st January, 2026.

Body text here.

Copy to:
1. Ministry of Finance
2. Ministry of Health"""

        result = check_field_consistency("AUTHORITY", "Ministry of Finance", text)
        assert not result["consistent"]
        assert any("AUTHORITY_SOURCE_CONFLICT" in w for w in result["warnings"])

    def test_valid_authority(self):
        result = check_field_consistency("AUTHORITY", "Ministry of Health and Family Welfare", "")
        assert result["consistent"]

    def test_date_consistency(self):
        extractions = [
            {"label": "DATE", "value": "1st January, 2026", "status": "HIGH_CONFIDENCE",
             "date_type": "START_DATE", "evidence_text": "1st January"},
            {"label": "DATE", "value": "15th December, 2025", "status": "HIGH_CONFIDENCE",
             "date_type": "DEADLINE", "evidence_text": "15th December"},
        ]
        result = run_consistency_checks(extractions)
        # Start date (Jan 2026) after deadline (Dec 2025) should flag
        assert any("DATE_INCONSISTENCY" in w for w in result["warnings"])


# ========================================================
# Test: Evidence with Bounding Boxes
# ========================================================

class TestEvidenceWithBbox:
    """Tests evidence extraction with bounding box propagation."""

    def test_evidence_with_bbox(self):
        text = "MINISTRY OF HOME AFFAIRS notification dated 1st January, 2026."
        words = [
            {"text": "MINISTRY", "bbox": [10, 10, 100, 30], "confidence": 0.95},
            {"text": "OF", "bbox": [110, 10, 130, 30], "confidence": 0.98},
            {"text": "HOME", "bbox": [140, 10, 190, 30], "confidence": 0.96},
            {"text": "AFFAIRS", "bbox": [200, 10, 270, 30], "confidence": 0.94},
        ]
        result = extract_evidence(
            original_text=text,
            label="AUTHORITY",
            start_char=0,
            end_char=len("MINISTRY OF HOME AFFAIRS"),
            page_number=1,
            confidence=0.94,
            page_words=words,
        )
        assert result["value"] == "MINISTRY OF HOME AFFAIRS"
        assert result["bbox"] is not None
        assert len(result["bbox"]) == 4
        assert result["page"] == 1

    def test_invalid_span_returns_low_confidence(self):
        result = extract_evidence(
            original_text="short",
            label="TITLE",
            start_char=10,
            end_char=20,
            page_number=1,
            confidence=0.9,
        )
        assert result["value"] is None
        assert result["status"] == "LOW_CONFIDENCE"


# ========================================================
# DEMO 1: High-Quality Notification
# ========================================================

class TestDemo1HighQuality:
    """
    DEMO 1: High-quality notification.
    Expected: Authority → HIGH CONFIDENCE, Date → HIGH CONFIDENCE, Title → HIGH CONFIDENCE
    """

    def test_digital_pdf_full_pipeline(self, tmp_path):
        """End-to-end test on a digital (high-quality) PDF."""
        pdf_path = str(tmp_path / "high_quality.pdf")
        create_digital_pdf(pdf_path)

        # Step 1: Parse PDF
        doc = PDFParser.parse_pdf(pdf_path)
        assert doc["document_type_summary"] == "TEXT"

        # Step 2: OCR (should skip for TEXT pages)
        engine = OCREngine()
        doc = engine.process_document_ocr(doc, pdf_path)
        assert doc["full_text"].strip() != ""

        # Step 3: Clean
        doc = TextCleaner.clean_document_object(doc)
        assert "original_text" in doc

        # Step 4: Date extraction
        dates = DateClassifier.extract_and_classify_dates(doc["full_text"])
        assert len(dates) >= 1, "Should find at least one date"

        # Step 5: OCR quality should be high for digital PDF
        quality = doc.get("ocr_quality", {})
        assert quality.get("ocr_confidence", 0) >= 0.8 or quality.get("quality_label") in ("GOOD", None)


# ========================================================
# DEMO 2: Low-Quality / Scanned Notification
# ========================================================

class TestDemo2ScannedPDF:
    """
    DEMO 2: Low-quality/scanned notification.
    Expected: Some fields → HIGH CONFIDENCE, Some → LOW CONFIDENCE, Missing → NOT FOUND
    """

    def test_scanned_pdf_type_detection(self, tmp_path):
        """Scanned PDF should be detected as needing OCR."""
        pdf_path = str(tmp_path / "scanned.pdf")
        create_scanned_pdf(pdf_path)

        doc = PDFParser.parse_pdf(pdf_path)
        # At least one page should need OCR
        needs_ocr_pages = [p for p in doc["pages"] if p["needs_ocr"]]
        assert len(needs_ocr_pages) > 0, "Scanned PDF should have pages needing OCR"

    def test_scanned_pdf_ocr_produces_text(self, tmp_path):
        """OCR on scanned PDF should produce non-empty text."""
        pdf_path = str(tmp_path / "scanned.pdf")
        create_scanned_pdf(pdf_path)

        doc = PDFParser.parse_pdf(pdf_path)
        engine = OCREngine(lang="en", dpi=250)
        doc = engine.process_document_ocr(doc, pdf_path)

        full_text = doc.get("full_text", "")
        # OCR should extract some text (even if imperfect)
        assert len(full_text) > 50, f"OCR should extract text from scanned PDF, got {len(full_text)} chars"

    def test_scanned_pdf_has_ocr_quality(self, tmp_path):
        """Scanned PDF should have OCR quality metrics."""
        pdf_path = str(tmp_path / "scanned.pdf")
        create_scanned_pdf(pdf_path)

        doc = PDFParser.parse_pdf(pdf_path)
        engine = OCREngine(lang="en", dpi=250)
        doc = engine.process_document_ocr(doc, pdf_path)

        quality = doc.get("ocr_quality", {})
        assert "ocr_confidence" in quality
        assert "quality_label" in quality

    def test_scanned_pdf_words_have_bboxes(self, tmp_path):
        """OCR words from scanned PDF should have bounding boxes."""
        pdf_path = str(tmp_path / "scanned.pdf")
        create_scanned_pdf(pdf_path)

        doc = PDFParser.parse_pdf(pdf_path)
        engine = OCREngine(lang="en", dpi=250)
        doc = engine.process_document_ocr(doc, pdf_path)

        all_words = doc.get("all_words", [])
        if all_words:  # Only check if OCR produced words
            for word in all_words[:5]:
                assert "bbox" in word
                assert len(word["bbox"]) == 4
                assert "confidence" in word

    def test_scanned_pdf_dual_text_preservation(self, tmp_path):
        """Scanned PDF should preserve original OCR text after cleaning."""
        pdf_path = str(tmp_path / "scanned.pdf")
        create_scanned_pdf(pdf_path)

        doc = PDFParser.parse_pdf(pdf_path)
        engine = OCREngine(lang="en", dpi=250)
        doc = engine.process_document_ocr(doc, pdf_path)
        doc = TextCleaner.clean_document_object(doc)

        for page in doc["pages"]:
            assert "original_ocr_text" in page, "Original OCR text must be preserved"

        assert "original_text" in doc, "Document-level original text must exist"


# ========================================================
# Test: Abstention Decision Display
# ========================================================

class TestAbstentionDecisionDisplay:
    """
    Tests that the system correctly shows:
    - Prediction
    - Confidence
    - Threshold
    - Evidence
    - Decision (ACCEPT / ABSTAIN)
    """

    def test_accept_decision(self):
        """High confidence should lead to ACCEPT."""
        result = estimate_field_confidence(
            label="DATE",
            text="19th August, 2026",
            token_probabilities=[0.95, 0.93, 0.97],
            evidence_available=True,
            evidence_span_complete=True,
            consistency_ok=True,
        )
        # With all signals strong, confidence should exceed typical threshold (0.85)
        assert result["confidence"] > 0.80

    def test_abstain_decision(self):
        """Low confidence should lead to ABSTAIN."""
        result = estimate_field_confidence(
            label="ELIGIBILITY",
            text="all persons",
            token_probabilities=[0.4, 0.3],
            evidence_available=False,
            evidence_span_complete=False,
            consistency_ok=False,
        )
        # With all signals weak, confidence should be below threshold
        assert result["confidence"] < 0.6
