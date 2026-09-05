import pytest
from src.inference.evidence import extract_evidence

def test_extract_evidence_exact_match():
    doc_text = "The Ministry of Corporate Affairs issued a notice today. It concerns the new policy."
    
    # "Ministry of Corporate Affairs" starts at index 4, ends at 33
    result = extract_evidence(
        original_text=doc_text,
        label="AUTHORITY",
        start_char=4,
        end_char=33,
        page_number=1,
        confidence=0.98
    )
    
    assert result["field"] == "AUTHORITY"
    assert result["value"] == "Ministry of Corporate Affairs"
    assert result["evidence_text"] == "Ministry of Corporate Affairs"
    # Sentence boundary extraction should grab the full first sentence
    assert result["evidence_sentence"] == "The Ministry of Corporate Affairs issued a notice today."
    assert result["character_start"] == 4
    assert result["character_end"] == 33
    assert result["page"] == 1
    assert result["confidence"] == 0.98
    assert result["status"] == "VALID_SPAN"


def test_invalid_span_returns_low_confidence():
    doc_text = "Short text."
    
    # Out of bounds
    result = extract_evidence(
        original_text=doc_text,
        label="DATE",
        start_char=20,
        end_char=25,
        page_number=1,
        confidence=0.90 # Model was confident but span is invalid
    )
    
    assert result["status"] == "LOW_CONFIDENCE"
    assert result["confidence"] == 0.0 # Force to 0.0
    assert result["value"] is None
    assert result["evidence_text"] is None
    assert result["evidence_sentence"] is None


def test_inverted_span_returns_low_confidence():
    doc_text = "Hello world."
    
    # Inverted start and end
    result = extract_evidence(
        original_text=doc_text,
        label="TITLE",
        start_char=5,
        end_char=2,
        page_number=2,
        confidence=0.85
    )
    
    assert result["status"] == "LOW_CONFIDENCE"
    assert result["confidence"] == 0.0
    assert result["value"] is None


def test_whitespace_span_returns_low_confidence():
    doc_text = "Hello   world."
    
    # Span points exactly to the three spaces between Hello and world
    result = extract_evidence(
        original_text=doc_text,
        label="CONTACT",
        start_char=5,
        end_char=8,
        page_number=1,
        confidence=0.75
    )
    
    assert result["status"] == "LOW_CONFIDENCE"
    assert result["confidence"] == 0.0
    assert result["value"] is None


def test_multiline_evidence_sentence():
    doc_text = "Line one.\nThis is the target span in line two.\nLine three."
    
    # "target span"
    start = doc_text.find("target span")
    end = start + len("target span")
    
    result = extract_evidence(
        original_text=doc_text,
        label="DOCUMENT",
        start_char=start,
        end_char=end,
        page_number=1,
        confidence=0.99
    )
    
    assert result["evidence_text"] == "target span"
    assert result["evidence_sentence"] == "This is the target span in line two."
