"""
confidence.py – Multi-Signal Field-Level Confidence for TrustExtract-N
=======================================================================
Core innovation: combines multiple signals for trustworthy confidence estimation.

C_final = α·C_model + β·C_ocr + γ·C_evidence + δ·C_consistency

Signals:
  C_model     — model prediction probability (mean span softmax)
  C_ocr       — OCR confidence for the tokens comprising this span
  C_evidence  — evidence availability and span completeness
  C_consistency — consistency check result for this field

Initial weights (tunable on validation data):
  α = 0.45, β = 0.25, γ = 0.20, δ = 0.10
"""

from typing import List, Dict, Any, Optional
import numpy as np


# Default weights — should be tuned on validation data
DEFAULT_WEIGHTS = {
    "model": 0.45,
    "ocr": 0.25,
    "evidence": 0.20,
    "consistency": 0.10,
}

# Field-specific weight overrides
# (some fields rely more on OCR accuracy, others more on model)
FIELD_WEIGHTS = {
    "title": {"model": 0.50, "ocr": 0.20, "evidence": 0.20, "consistency": 0.10},
    "authority": {"model": 0.45, "ocr": 0.20, "evidence": 0.25, "consistency": 0.10},
    "date": {"model": 0.40, "ocr": 0.30, "evidence": 0.20, "consistency": 0.10},
    "contact": {"model": 0.35, "ocr": 0.30, "evidence": 0.25, "consistency": 0.10},
    "eligibility": {"model": 0.50, "ocr": 0.20, "evidence": 0.20, "consistency": 0.10},
    "audience": {"model": 0.50, "ocr": 0.20, "evidence": 0.20, "consistency": 0.10},
    "document": {"model": 0.50, "ocr": 0.20, "evidence": 0.20, "consistency": 0.10},
}


def estimate_field_confidence(
    label: str,
    text: str,
    token_probabilities: List[float],
    ocr_word_confidences: Optional[List[float]] = None,
    evidence_available: bool = False,
    evidence_span_complete: bool = False,
    consistency_ok: bool = True,
    page_ocr_quality: float = 1.0,
) -> Dict[str, Any]:
    """
    Multi-signal field-level confidence estimation.

    Args:
        label: The entity label (e.g., "DATE", "AUTHORITY").
        text: The extracted entity text string.
        token_probabilities: List of softmax probabilities for entity span tokens.
        ocr_word_confidences: OCR confidence scores for words in this span.
        evidence_available: Whether grounding evidence was found.
        evidence_span_complete: Whether the span covers the full entity.
        consistency_ok: Whether consistency checks passed for this field.
        page_ocr_quality: Overall OCR quality score for the page (0-1).

    Returns:
        Dict containing the extraction with multi-signal confidence metrics.
    """
    # Get field-specific weights
    field_key = label.lower()
    weights = FIELD_WEIGHTS.get(field_key, DEFAULT_WEIGHTS)

    # 1. Model confidence (C_model)
    if token_probabilities:
        c_model = float(np.mean(token_probabilities))
        min_prob = float(np.min(token_probabilities))
    else:
        c_model = 0.0
        min_prob = 0.0

    # 2. OCR confidence (C_ocr)
    if ocr_word_confidences and len(ocr_word_confidences) > 0:
        c_ocr = float(np.mean(ocr_word_confidences))
    else:
        # Use page-level OCR quality as fallback
        c_ocr = page_ocr_quality

    # 3. Evidence confidence (C_evidence)
    c_evidence = 0.0
    if evidence_available:
        c_evidence = 0.8
        if evidence_span_complete:
            c_evidence = 1.0
    # If no evidence, c_evidence = 0.0 → reduces overall confidence

    # 4. Consistency confidence (C_consistency)
    c_consistency = 1.0 if consistency_ok else 0.4

    # Compute weighted final confidence
    c_final = (
        weights["model"] * c_model
        + weights["ocr"] * c_ocr
        + weights["evidence"] * c_evidence
        + weights["consistency"] * c_consistency
    )
    c_final = max(0.0, min(1.0, c_final))

    # Span length
    span_length = len(token_probabilities) if token_probabilities else 0

    return {
        "label": label,
        "text": text,
        "confidence": round(c_final, 4),
        "confidence_raw": round(c_model, 4),
        "metrics": {
            "c_model": round(c_model, 4),
            "c_ocr": round(c_ocr, 4),
            "c_evidence": round(c_evidence, 4),
            "c_consistency": round(c_consistency, 4),
            "min_token_probability": round(min_prob, 4),
            "mean_span_probability": round(c_model, 4),
            "span_length": span_length,
        },
        "weights_used": weights,
        "evidence_available": evidence_available,
    }


def get_ocr_confidence_for_span(
    start_char: int,
    end_char: int,
    page_words: List[Dict[str, Any]],
    page_start_offset: int = 0,
) -> List[float]:
    """
    Retrieves OCR confidence scores for words that overlap with the given span.

    Args:
        start_char: Start character offset in page text.
        end_char: End character offset in page text.
        page_words: List of OCR word dicts with 'text', 'confidence', 'bbox'.
        page_start_offset: Character offset of this page in the full document.

    Returns:
        List of OCR confidence scores for overlapping words.
    """
    if not page_words:
        return []

    confidences = []
    current_pos = page_start_offset

    for word in page_words:
        word_text = word.get("text", "")
        word_len = len(word_text)
        word_end = current_pos + word_len

        # Check overlap between word position and target span
        if current_pos < end_char and word_end > start_char:
            conf = word.get("confidence", 0.0)
            confidences.append(conf)

        current_pos = word_end + 1  # +1 for space

    return confidences
