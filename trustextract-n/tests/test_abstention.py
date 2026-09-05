import pytest
from src.inference.abstention import (
    apply_abstention, 
    evaluate_selective_metrics, 
    tune_threshold
)

def test_confident_correct_answer():
    """Test 1: Confident correct answer (should ACCEPT)"""
    extraction = {
        "label": "DATE",
        "text": "12 Jan 2024",
        "confidence": 0.95,
        "evidence_text": "Dated 12 Jan 2024",
        "page_number": 1
    }
    
    result = apply_abstention(extraction, threshold=0.90)
    
    assert result["status"] == "HIGH_CONFIDENCE"
    assert result["value"] == "12 Jan 2024"
    assert result["evidence"] == "Dated 12 Jan 2024"
    assert result["page_number"] == 1


def test_uncertain_answer():
    """Test 2: Uncertain answer (should ABSTAIN)"""
    extraction = {
        "label": "AUTHORITY",
        "text": "Department of Guesswork",
        "confidence": 0.60,
        "evidence_text": "Signed by Department of Guesswork",
        "page_number": 2
    }
    
    result = apply_abstention(extraction, threshold=0.90)
    
    assert result["status"] == "LOW_CONFIDENCE"
    assert result["value"] is None
    assert result["evidence"] is None  # Must not present evidence as reliable
    assert result["page_number"] is None


def test_missing_field():
    """Test 3: Missing field (FAR metric)"""
    # Simulate an empty extraction with low confidence
    extraction = {
        "label": "CONTACT",
        "text": None,
        "confidence": 0.10,
        "evidence_text": None,
        "page_number": None
    }
    
    result = apply_abstention(extraction, threshold=0.90)
    
    assert result["status"] == "LOW_CONFIDENCE"
    assert result["value"] is None
    assert result["evidence"] is None
    
    # Check FAR calculation on missing fields
    predictions = [
        {"confidence": 0.95, "is_correct": False, "is_missing_in_ground_truth": True}, # False positive
        {"confidence": 0.20, "is_correct": True,  "is_missing_in_ground_truth": True}  # True negative (abstained)
    ]
    
    metrics = evaluate_selective_metrics(predictions, threshold=0.90)
    # Total missing in GT = 2. Number accepted = 1. FAR = 1/2 = 0.5
    assert metrics["false_answer_rate_missing"] == 0.5


def test_incorrect_high_probability_candidate():
    """Test 4: Incorrect high-probability candidate (lowers selective precision)"""
    predictions = [
        {"confidence": 0.98, "is_correct": True, "is_missing_in_ground_truth": False},
        {"confidence": 0.95, "is_correct": False, "is_missing_in_ground_truth": False} # Incorrect but confident
    ]
    
    # At threshold 0.90, both are accepted. Precision = 1/2 = 0.5
    metrics = evaluate_selective_metrics(predictions, threshold=0.90)
    assert metrics["selective_precision"] == 0.5
    assert metrics["coverage"] == 1.0


def test_tune_threshold():
    """Test threshold selection logic based on target precision"""
    predictions = [
        {"confidence": 0.99, "is_correct": True, "is_missing_in_ground_truth": False},
        {"confidence": 0.95, "is_correct": True, "is_missing_in_ground_truth": False},
        {"confidence": 0.85, "is_correct": False, "is_missing_in_ground_truth": False}, # Confident error
        {"confidence": 0.60, "is_correct": True, "is_missing_in_ground_truth": False},
        {"confidence": 0.40, "is_correct": False, "is_missing_in_ground_truth": True},
    ]
    
    # If we want 1.0 precision, we must set threshold > 0.85 to exclude the confident error
    t_strict = tune_threshold(predictions, target_precision=1.0)
    assert t_strict > 0.85
    assert t_strict <= 0.95 # Should pick lowest possible threshold to maximize coverage
    
    # If we only want 0.66 precision, we can set threshold down to 0.85 (2 correct, 1 wrong accepted)
    t_loose = tune_threshold(predictions, target_precision=0.66)
    assert t_loose <= 0.85
