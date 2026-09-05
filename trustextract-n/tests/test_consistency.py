import pytest
from src.inference.consistency import run_consistency_checks

def test_perfect_consistency():
    extractions = [
        {"label": "AUTHORITY", "value": "Ministry of Finance", "status": "HIGH_CONFIDENCE"},
        {"label": "DATE", "value": "12 Jan 2024", "status": "HIGH_CONFIDENCE"},
        {"label": "CONTACT", "value": "contact@finance.gov.in", "status": "HIGH_CONFIDENCE"}
    ]
    
    result = run_consistency_checks(extractions)
    
    assert result["consistency_score"] == 1.0
    assert len(result["warnings"]) == 0
    assert result["is_highly_inconsistent"] is False


def test_date_plausibility_and_conflicts():
    extractions = [
        {"label": "DATE", "value": "12 Jan 1850", "status": "HIGH_CONFIDENCE"}, # Implausible
        {"label": "DATE", "value": "15 March 2024", "status": "HIGH_CONFIDENCE"}, # Conflicting with the first one
        {"label": "DATE", "value": "Not a date", "status": "HIGH_CONFIDENCE"} # Unparseable
    ]
    
    result = run_consistency_checks(extractions)
    
    # 1.0 - 0.2 (implausible) - 0.1 (conflict) - 0.15 (unparseable) = 0.55
    assert result["consistency_score"] == 0.55
    assert len(result["warnings"]) == 3
    assert result["is_highly_inconsistent"] is True
    
    warning_texts = " ".join(result["warnings"])
    assert "implausible" in warning_texts
    assert "conflicting" in warning_texts
    assert "parsed" in warning_texts


def test_contact_format_validation():
    extractions = [
        {"label": "CONTACT", "value": "Just a name", "status": "HIGH_CONFIDENCE"}, # Invalid
        {"label": "CONTACT", "value": "+91-9876543210", "status": "HIGH_CONFIDENCE"} # Valid phone
    ]
    
    result = run_consistency_checks(extractions)
    
    assert result["consistency_score"] == 0.85 # -0.15 for the bad contact
    assert len(result["warnings"]) == 1
    assert "email or phone" in result["warnings"][0]


def test_empty_short_entity():
    extractions = [
        {"label": "DOCUMENT", "value": "A", "status": "HIGH_CONFIDENCE"} # Too short
    ]
    
    result = run_consistency_checks(extractions)
    
    assert result["consistency_score"] == 0.90 # -0.10
    assert len(result["warnings"]) == 1
    assert "unusually short" in result["warnings"][0]


def test_authority_plausibility():
    extractions = [
        {"label": "AUTHORITY", "value": "Some Random Group", "status": "HIGH_CONFIDENCE"} # No keywords
    ]
    
    result = run_consistency_checks(extractions)
    
    assert result["consistency_score"] == 0.90 # -0.10
    assert len(result["warnings"]) == 1
    assert "recognized keywords" in result["warnings"][0]


def test_ignores_abstained_fields():
    extractions = [
        {"label": "DATE", "value": "12 Jan 1850", "status": "LOW_CONFIDENCE"}, # Implausible but abstained
        {"label": "AUTHORITY", "value": None, "status": "LOW_CONFIDENCE"} # Null
    ]
    
    result = run_consistency_checks(extractions)
    
    # Should not penalize abstained fields
    assert result["consistency_score"] == 1.0
    assert len(result["warnings"]) == 0
