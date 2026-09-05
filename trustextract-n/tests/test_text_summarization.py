"""
Test Suite for Text-Input Notice Summarization Pipeline — TrustExtract-N
========================================================================
Tests raw text notice processing, extractive summarization, length modes,
date classification, zero-hallucination bounds, Hindi/multilingual support,
and noisy text resilience.
"""

import pytest
from src.inference.pipeline import TrustExtractPipeline
from src.summarization.extractive import summarize_extractive


@pytest.fixture
def pipeline():
    return TrustExtractPipeline()


def test_short_notice_summarization(pipeline):
    text = "TEZPUR UNIVERSITY NOTIFICATION. Admission closed on 10 August 2026."
    res = pipeline.process_raw_text(text, summary_length="SHORT")
    assert res["summary"]["text"] != ""
    assert res["document"]["document_type_summary"] == "TEXT_INPUT"
    assert res["document"]["ocr_confidence"] == 1.0


def test_long_notice_summarization(pipeline):
    text = """
    GOVERNMENT OF INDIA
    MINISTRY OF HEALTH AND FAMILY WELFARE
    NOTIFICATION
    New Delhi, the 19th August, 2026

    G.S.R. 745(E).— The following draft rules further to amend the Drugs Rules, 1945,
    which the Central Government proposes to make, in exercise of powers conferred by section 12.

    Objections and suggestions may be addressed to the Under Secretary,
    Ministry of Health and Family Welfare, New Delhi or emailed at drugsdiv-mohfw@gov.in.

    Last date for submission: 19th September, 2026.

    Eligible persons: All registered drug manufacturers holding valid licenses.
    Required documents: Copy of drug manufacturing license, Registration certificate.
    """
    res = pipeline.process_raw_text(text, summary_length="STANDARD")
    assert "MINISTRY OF HEALTH AND FAMILY WELFARE" in (res["authority"]["value"] or "")
    assert len(res["dates"]) >= 2
    assert res["summary"]["text"] != ""


def test_notice_with_multiple_dates(pipeline):
    text = """
    MINISTRY OF FINANCE NOTIFICATION
    Dated: 05th August, 2026.
    Effective from: 01st September, 2026.
    Last date for objections: 20th September, 2026.
    """
    res = pipeline.process_raw_text(text)
    dates = res["dates"]
    types = [d["type"] for d in dates]
    assert len(dates) >= 2
    assert "ISSUE_DATE" in types or "NOTIFICATION_DATE" in types or "UNKNOWN_DATE" in types
    assert any(t in ("DEADLINE", "LAST_DATE", "END_DATE") for t in types)


def test_notice_with_missing_fields_zero_hallucination(pipeline):
    """
    HALLUCINATION TEST:
    Input contains ONLY a purpose sentence.
    Expected: No fake dates, eligibility, documents, or contact information.
    """
    text = "This notice informs all concerned about the upcoming examination."
    res = pipeline.process_raw_text(text)
    
    # Missing fields MUST be NOT_FOUND / None — zero hallucination guarantee
    assert res["authority"]["value"] is None
    assert res["eligibility"]["value"] is None
    assert res["required_documents"]["value"] is None
    assert res["contact"]["value"] is None
    assert res["notice_number"]["value"] is None

    # No fake last date
    deadline_dates = [d for d in res["dates"] if d.get("type") in ("DEADLINE", "LAST_DATE")]
    assert len(deadline_dates) == 0


def test_hindi_notice_summarization(pipeline):
    text = """
    भारत सरकार
    स्वास्थ्य एवं परिवार कल्याण मंत्रालय
    अधिसूचना
    दिनांक: 15 अगस्त 2026
    अंतिम तिथि: 15 सितम्बर 2026
    सम्पर्क: health-info@gov.in
    """
    res = pipeline.process_raw_text(text)
    assert res["summary"]["text"] != ""
    assert res["document"]["document_type_summary"] == "TEXT_INPUT"


def test_bilingual_notice_summarization(pipeline):
    text = """
    GOVERNMENT OF INDIA / भारत सरकार
    MINISTRY OF CORPORATE AFFAIRS / कॉर्पोरेट कार्य मंत्रालय
    NOTIFICATION / अधिसूचना
    Last date: 30th September 2026 / अंतिम तिथि: 30 सितम्बर 2026
    """
    res = pipeline.process_raw_text(text)
    assert res["summary"]["text"] != ""
    assert len(res["dates"]) >= 1


def test_noisy_ocr_text_summarization(pipeline):
    text = "GOV'T OF IND1A M1NISTRY OF FINANC3 NOT1FICAT1ON Dated 12/08/2026 Last Date: 31/08/2026"
    res = pipeline.process_raw_text(text)
    assert res["summary"]["text"] != ""


def test_summary_length_modes():
    text = """
    Line 1: GOVERNMENT OF INDIA NOTIFICATION.
    Line 2: Ministry of Health issues draft guidelines for 2026.
    Line 3: Public comments are invited from all citizens and experts.
    Line 4: Submissions must be sent to contact@health.gov.in.
    Line 5: Last date for receipt of comments is 30th October 2026.
    Line 6: Late submissions will not be entertained.
    Line 7: Sd/- Under Secretary to the Government of India.
    """
    exts = [{"label": "TITLE", "value": "NOTIFICATION", "status": "HIGH_CONFIDENCE"}]
    
    short_res = summarize_extractive(text, exts, summary_length="SHORT")
    std_res = summarize_extractive(text, exts, summary_length="STANDARD")
    det_res = summarize_extractive(text, exts, summary_length="DETAILED")
    
    assert len(short_res["sentences"]) <= 2
    assert len(std_res["sentences"]) <= 5
    assert len(det_res["sentences"]) <= 8
