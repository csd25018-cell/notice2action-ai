"""
evidence.py – Evidence Extraction with Bounding Boxes for TrustExtract-N
=========================================================================
Retrieves exact matching substrings from the original document text.
Guarantees zero generation/paraphrasing for maximum verifiability.
Now includes bounding box coordinates and original OCR text.
"""

from typing import Dict, Any, Optional, List
import re


def extract_evidence(
    original_text: str,
    label: str,
    start_char: int,
    end_char: int,
    page_number: Optional[int],
    confidence: float,
    bbox: Optional[List[float]] = None,
    original_ocr_text: Optional[str] = None,
    page_words: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Extracts the exact text span and its surrounding sentence context directly
    from the original document text. Includes bounding box when available.

    Args:
        original_text: The full string of the document (normalized).
        label: The entity field name (e.g., 'DATE').
        start_char: The predicted start index.
        end_char: The predicted end index.
        page_number: The page on which this span occurs.
        confidence: The calibrated or raw confidence score.
        bbox: Optional bounding box [x1, y1, x2, y2] from OCR.
        original_ocr_text: The original (pre-cleaning) OCR text for evidence display.
        page_words: OCR word-level data for this page, for bbox lookup.

    Returns:
        Dict adhering to the exact evidence schema with bbox.
    """
    text_len = len(original_text)

    # Validate the span indices
    is_valid_span = (
        isinstance(start_char, int) and
        isinstance(end_char, int) and
        0 <= start_char < end_char <= text_len
    )

    if not is_valid_span:
        # Invalid span: do not fabricate evidence, force LOW_CONFIDENCE
        return {
            "field": label,
            "value": None,
            "confidence": 0.0,
            "status": "LOW_CONFIDENCE",
            "page": None,
            "page_number": None,
            "bbox": None,
            "character_start": None,
            "character_end": None,
            "evidence_sentence": None,
            "evidence_text": None,
            "original_evidence": None,
        }

    # Extract exact substring
    evidence_text = original_text[start_char:end_char]

    if not evidence_text.strip():
        return {
            "field": label,
            "value": None,
            "confidence": 0.0,
            "status": "LOW_CONFIDENCE",
            "page": None,
            "page_number": None,
            "bbox": None,
            "character_start": None,
            "character_end": None,
            "evidence_sentence": None,
            "evidence_text": None,
            "original_evidence": None,
        }

    # Try to find bounding box from OCR word data if not provided
    evidence_bbox = bbox
    if evidence_bbox is None and page_words:
        evidence_bbox = _find_bbox_for_span(evidence_text, page_words)

    # Get original (pre-cleaning) evidence text if available
    original_evidence = None
    if original_ocr_text:
        # Try to find the evidence in original OCR text
        original_evidence = _find_in_original(evidence_text, original_ocr_text)

    # Extract sentence context by expanding outwards to boundaries (. \n ? !)
    prefix = original_text[:start_char]
    match_start = list(re.finditer(r'[\.\n\?\!]\s+', prefix))
    if match_start:
        sent_start = match_start[-1].end()
    else:
        sent_start = 0

    suffix = original_text[end_char:]
    match_end = re.search(r'[\.\n\?\!]', suffix)
    if match_end:
        sent_end = end_char + match_end.start() + 1
    else:
        sent_end = text_len

    evidence_sentence = original_text[sent_start:sent_end].strip()

    return {
        "field": label,
        "value": evidence_text,
        "confidence": confidence,
        "status": "VALID_SPAN",
        "page": page_number,
        "page_number": page_number,
        "bbox": evidence_bbox,
        "character_start": start_char,
        "character_end": end_char,
        "evidence_sentence": evidence_sentence,
        "evidence_text": evidence_text,
        "original_evidence": original_evidence or evidence_text,
    }


def _find_bbox_for_span(
    evidence_text: str,
    page_words: List[Dict[str, Any]],
) -> Optional[List[float]]:
    """
    Finds the bounding box that covers all OCR words matching the evidence span.
    """
    if not page_words or not evidence_text:
        return None

    evidence_lower = evidence_text.lower().strip()
    matching_bboxes = []

    # Simple: find words whose text appears in the evidence
    for word in page_words:
        word_text = word.get("text", "").strip().lower()
        if word_text and word_text in evidence_lower:
            bbox = word.get("bbox")
            if bbox and len(bbox) == 4:
                matching_bboxes.append(bbox)

    if not matching_bboxes:
        return None

    # Compute union bounding box
    x1 = min(b[0] for b in matching_bboxes)
    y1 = min(b[1] for b in matching_bboxes)
    x2 = max(b[2] for b in matching_bboxes)
    y2 = max(b[3] for b in matching_bboxes)

    return [x1, y1, x2, y2]


def _find_in_original(evidence_text: str, original_text: str) -> Optional[str]:
    """
    Attempts to find the evidence text in the original (pre-cleaned) OCR text.
    Uses fuzzy matching if exact match fails.
    """
    if not evidence_text or not original_text:
        return None

    # Exact match
    idx = original_text.find(evidence_text)
    if idx >= 0:
        return evidence_text

    # Case-insensitive match
    idx = original_text.lower().find(evidence_text.lower())
    if idx >= 0:
        return original_text[idx:idx + len(evidence_text)]

    # Whitespace-normalized match
    normalized_target = " ".join(evidence_text.split())
    normalized_original = " ".join(original_text.split())
    idx = normalized_original.lower().find(normalized_target.lower())
    if idx >= 0:
        # Map back to original
        return normalized_target

    return None
