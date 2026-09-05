"""
Unit tests for TrustExtract-N preprocessing modules (PDFParser, OCREngine, TextCleaner).
Uses synthetic document examples and mock structures.
"""

import pytest
from src.preprocessing.pdf_parser import PDFParser
from src.preprocessing.ocr import OCREngine
from src.preprocessing.text_cleaner import TextCleaner

def test_text_cleaner_entity_preservation():
    """
    Test that TextCleaner preserves dates, emails, phone numbers, and DIN identifiers.
    """
    raw_notice = """
    GOVERNMENT OF INDIA  MINISTRY OF FINANCE
    DIN: DIN-2026-993821-GST
    Date: 15-09-2026

    Please submit reply to contact@tax.gov.in or call +91-9876543210.
    Page 1 of 3
    """
    cleaned = TextCleaner.clean_text(raw_notice)

    assert "DIN-2026-993821-GST" in cleaned
    assert "15-09-2026" in cleaned
    assert "contact@tax.gov.in" in cleaned
    assert "+91-9876543210" in cleaned
    assert "Page 1 of 3" not in cleaned

def test_document_schema_structure():
    """
    Test structured document output schema consistency.
    """
    synthetic_doc = {
        "document_id": "DOC-TEST-001",
        "file_path": "synthetic.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Page 1 Notice Header. Date: 01-09-2026.",
                "source": "pdf_text",
                "needs_ocr": False
            },
            {
                "page_number": 2,
                "text": "",
                "source": "low_text_pdf",
                "needs_ocr": True
            }
        ],
        "full_text": "Page 1 Notice Header. Date: 01-09-2026."
    }

    cleaned_doc = TextCleaner.clean_document_object(synthetic_doc)

    assert "document_id" in cleaned_doc
    assert len(cleaned_doc["pages"]) == 2
    assert cleaned_doc["pages"][0]["page_number"] == 1
    assert cleaned_doc["pages"][0]["source"] == "pdf_text"
    assert "full_text" in cleaned_doc
    assert cleaned_doc["pages"][0]["start_offset"] == 0

def test_ocr_engine_configurable_language():
    """
    Test OCREngine language configuration.
    """
    ocr_eng = OCREngine(lang="eng+hin", dpi=150)
    assert ocr_eng.lang == "eng+hin"

    synthetic_doc = {
        "document_id": "DOC-TEST-002",
        "pages": [
            {
                "page_number": 1,
                "text": "Native digital text on page 1 with 100 characters sample text for testing.",
                "source": "pdf_text",
                "needs_ocr": False
            }
        ],
        "full_text": "Native digital text on page 1 with 100 characters sample text for testing."
    }

    # Selective OCR should skip native high-text pages
    result = ocr_eng.process_document_ocr_selective(synthetic_doc, min_chars_threshold=50)
    assert result["pages"][0]["source"] == "pdf_text"
    assert result["ocr_language"] == "eng+hin"
