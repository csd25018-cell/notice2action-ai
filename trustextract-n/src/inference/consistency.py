"""
consistency.py – Enhanced Consistency Checks for TrustExtract-N
================================================================
Performs non-ML sanity checks on extracted fields to flag implausible
or contradictory information. Now includes:
- Authority source conflict detection (Copy To vs Issuing Authority)
- Date ordering checks (start_date > end_date → flag)
- Per-field consistency flags for multi-signal confidence
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dateutil import parser

# Simple deterministic regex for common contact formats in India
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
PHONE_REGEX = re.compile(r'^\+?[\d\s\-\(\)]{7,15}$')

AUTHORITY_KEYWORDS = {
    "ministry", "department", "board", "commission", "authority",
    "council", "office", "directorate", "government", "bureau",
    "cbic", "cbdt", "fssai", "mca", "epfo", "roc",
    "university", "institute", "controller", "registrar",
    "मंत्रालय", "विभाग", "कार्यालय", "सरकार",  # Hindi keywords
}

# Keywords indicating "Copy To" section (NOT the issuing authority)
COPY_TO_KEYWORDS = [
    "copy to", "copies to", "copy forwarded to",
    "endorsed to", "for information",
    "प्रतिलिपि", "की प्रति",  # Hindi
]


def _parse_date(date_str: str) -> datetime | None:
    """Attempts to parse a date string deterministically."""
    if not date_str:
        return None
    try:
        return parser.parse(date_str, fuzzy=True, dayfirst=True)
    except (ValueError, OverflowError, TypeError):
        return None


def check_field_consistency(
    field_name: str,
    value: str,
    full_text: str = "",
) -> Dict[str, Any]:
    """
    Checks consistency for a single field.
    Returns per-field consistency result for use in multi-signal confidence.
    """
    warnings = []
    is_consistent = True

    if not value or not value.strip():
        return {"consistent": True, "warnings": [], "field": field_name}

    value = value.strip()

    penalty = 0.0

    # Field-specific checks
    if field_name == "CONTACT":
        is_email = bool(EMAIL_REGEX.match(value))
        is_phone = bool(PHONE_REGEX.search(value))
        if not is_email and not is_phone:
            # Check if it's an address (acceptable)
            if len(value) < 20 and not any(c.isdigit() for c in value):
                warnings.append(f"Contact '{value}' does not match email or phone format.")
                is_consistent = False
                penalty = 0.15

    elif field_name == "AUTHORITY":
        val_lower = value.lower()
        if not any(kw in val_lower for kw in AUTHORITY_KEYWORDS):
            warnings.append(f"Authority '{value}' does not contain recognized keywords.")
            is_consistent = False
            penalty = 0.10

        # Check if this was extracted from a "Copy To" section
        if full_text:
            authority_pos = full_text.lower().find(value.lower())
            if authority_pos >= 0:
                # Check if there's a "copy to" marker before this position
                preceding_text = full_text[:authority_pos].lower()
                for copy_kw in COPY_TO_KEYWORDS:
                    last_copy_pos = preceding_text.rfind(copy_kw)
                    if last_copy_pos >= 0:
                        # If "copy to" appears close before the authority text
                        chars_between = authority_pos - last_copy_pos
                        if chars_between < 200:
                            warnings.append(
                                f"AUTHORITY_SOURCE_CONFLICT: Authority '{value}' may be from 'Copy To' section."
                            )
                            is_consistent = False
                            penalty = 0.15
                            break

    elif field_name == "DATE":
        parsed = _parse_date(value)
        if parsed is None:
            warnings.append(f"Date '{value}' could not be parsed.")
            is_consistent = False
            penalty = 0.15
        else:
            current_year = datetime.now().year
            if parsed.year < 1900 or parsed.year > current_year + 10:
                warnings.append(f"Date '{value}' ({parsed.year}) is implausible.")
                is_consistent = False
                penalty = 0.20

    elif field_name in ("TITLE", "AUDIENCE", "ELIGIBILITY", "DOCUMENT"):
        if len(value) < 2:
            warnings.append(f"Entity '{field_name}' is unusually short ('{value}').")
            is_consistent = False
            penalty = 0.10

    return {
        "consistent": is_consistent,
        "penalty": penalty,
        "warnings": warnings,
        "field": field_name,
    }


def run_consistency_checks(
    extractions: List[Dict[str, Any]],
    full_text: str = "",
    check_evidence: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a set of extracted fields for deterministic consistency.

    Args:
        extractions: List of field dictionaries.
        full_text: The full document text for context-aware checks.
        check_evidence: Whether to penalize fields missing evidence text.

    Returns:
        Dict containing a consistency score (0.0 to 1.0), a list of warning strings,
        and per-field consistency flags.
    """
    warnings = []
    score = 1.0
    field_consistency: Dict[str, bool] = {}

    dates_found = []

    for ext in extractions:
        # Skip abstained/null extractions
        if ext.get("status") == "LOW_CONFIDENCE" or not ext.get("value"):
            continue

        label = ext.get("label", ext.get("field", "")).upper()
        value = str(ext.get("value", "")).strip()

        # Per-field consistency check
        field_check = check_field_consistency(label, value, full_text)
        field_consistency[label] = field_check["consistent"]
        warnings.extend(field_check["warnings"])

        if not field_check["consistent"]:
            score -= field_check.get("penalty", 0.10)

        # Collect dates for cross-checks
        if label == "DATE":
            parsed_date = _parse_date(value)
            if parsed_date:
                dates_found.append((value, parsed_date, ext.get("date_type", "UNKNOWN")))

    # Cross-date checks
    if len(dates_found) >= 2:
        # Check for conflicting dates
        unique_dates = {dt.date() for _, dt, _ in dates_found}
        if len(unique_dates) > 1:
            # Multiple dates exist — check for logical ordering issues
            date_types = {dt_type for _, _, dt_type in dates_found}

            # Check: start_date should be before end_date
            start_dates = [(raw, dt) for raw, dt, t in dates_found if t == "START_DATE"]
            end_dates = [(raw, dt) for raw, dt, t in dates_found if t in ("END_DATE", "DEADLINE")]

            if start_dates and end_dates:
                for s_raw, s_dt in start_dates:
                    for e_raw, e_dt in end_dates:
                        if s_dt > e_dt:
                            warnings.append(
                                f"DATE_INCONSISTENCY: Start date '{s_raw}' is after end/deadline date '{e_raw}'."
                            )
                            score -= 0.2
            else:
                # Multiple dates without clear ordering
                warnings.append("Found conflicting dates.")
                score -= 0.10

    # Evidence span validation (only if check_evidence is explicitly requested)
    if check_evidence:
        for ext in extractions:
            if ext.get("status") != "LOW_CONFIDENCE" and ext.get("value"):
                has_evidence = ext.get("evidence_text") or ext.get("evidence")
                if not has_evidence:
                    label = ext.get("label", ext.get("field", ""))
                    warnings.append(f"No source evidence found for '{label}' extraction.")
                    score -= 0.1
                    field_consistency[label.upper()] = False

    final_score = max(0.0, round(score, 2))

    return {
        "consistency_score": final_score,
        "warnings": warnings,
        "is_highly_inconsistent": final_score < 0.6,
        "field_consistency": field_consistency,
    }
