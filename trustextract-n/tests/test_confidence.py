import pytest
from src.inference.confidence import estimate_field_confidence

def test_estimate_field_confidence_standard():
    result = estimate_field_confidence(
        label="DATE",
        text="30 September 2026",
        token_probabilities=[0.95, 0.88, 0.90],
        evidence_available=True
    )
    
    assert result["label"] == "DATE"
    assert result["text"] == "30 September 2026"
    assert result["confidence_raw"] == 0.91
    assert result["evidence_available"] is True
    
    metrics = result["metrics"]
    assert metrics["mean_span_probability"] == 0.91
    assert metrics["min_token_probability"] == 0.88
    assert metrics["span_length"] == 3

def test_estimate_field_confidence_empty_tokens():
    result = estimate_field_confidence(
        label="AUTHORITY",
        text="Unknown",
        token_probabilities=[],
        evidence_available=False
    )
    
    assert result["confidence_raw"] == 0.0
    assert result["metrics"]["span_length"] == 0
    assert result["metrics"]["mean_span_probability"] == 0.0

def test_estimate_field_confidence_single_token():
    result = estimate_field_confidence(
        label="DOCUMENT",
        text="Order",
        token_probabilities=[0.99],
        evidence_available=False
    )
    
    assert result["confidence_raw"] == 0.99
    assert result["metrics"]["mean_span_probability"] == 0.99
    assert result["metrics"]["min_token_probability"] == 0.99
    assert result["metrics"]["span_length"] == 1
