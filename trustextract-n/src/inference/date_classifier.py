"""
Date Classifier for TrustExtract-N
====================================
Detects ALL date candidates from text and classifies each date
using contextual keyword analysis.

Classifications:
  ISSUE_DATE      — "dated", "date of issue"
  EFFECTIVE_DATE  — "effective from", "w.e.f."
  START_DATE      — "from", "beginning", "commencement"
  END_DATE        — "up to", "until", "ending"
  DEADLINE        — "last date", "before", "within", "closing"
  EVENT_DATE      — "on the date of", "scheduled"
  REFERENCE_DATE  — "as on", "as of"
  UNKNOWN_DATE    — no clear contextual signal

Does NOT infer dates that are not explicitly present.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dateutil import parser as dateutil_parser
from datetime import datetime


# Context window: how many characters around a date to examine
CONTEXT_WINDOW = 120

# Date classification patterns — keywords that appear BEFORE or NEAR a date
DATE_CONTEXT_PATTERNS = {
    "ISSUE_DATE": [
        r"dated?\s*[:\-]?\s*$",
        r"date\s+of\s+issue",
        r"date\s+of\s+notification",
        r"date\s+of\s+order",
        r"date\s+of\s+circular",
        r"new\s+delhi\s*,?\s*(?:the\s+)?$",
        r"issued?\s+(?:on|dated?)\s*[:\-]?\s*$",
        r"published\s+(?:on|dated?)\s*[:\-]?\s*$",
        r"तिथि\s*[:\-]?\s*$",  # Hindi: "date"
        r"दिनांक\s*[:\-]?\s*$",  # Hindi: "date"
    ],
    "EFFECTIVE_DATE": [
        r"effective\s+(?:from|date)",
        r"w\.?\s*e\.?\s*f\.?\s*[:\-]?\s*$",
        r"with\s+effect\s+from",
        r"shall\s+come\s+into\s+(?:force|effect)",
        r"enforced?\s+(?:from|on)",
        r"applicable\s+(?:from|on|w\.e\.f)",
        r"प्रभावी\s+(?:तिथि|दिनांक)",  # Hindi
    ],
    "START_DATE": [
        r"(?:start|commence|begin|open)\s*(?:ing|s)?\s*(?:from|on|date)?\s*[:\-]?\s*$",
        r"from\s*[:\-]?\s*$",
        r"admission\s+(?:start|open|begin)",
        r"registration\s+(?:start|open|begin)",
        r"apply\s+from",
        r"आरम्भ\s+(?:तिथि|दिनांक)",  # Hindi
    ],
    "END_DATE": [
        r"(?:end|expir|clos|conclud)\s*(?:ing|e|es|ed)?\s*(?:on|date)?\s*[:\-]?\s*$",
        r"up\s*to\s*[:\-]?\s*$",
        r"until\s*[:\-]?\s*$",
        r"valid\s+(?:till|until|up\s+to)",
        r"समाप्ति\s+(?:तिथि|दिनांक)",  # Hindi
    ],
    "DEADLINE": [
        r"last\s+date",
        r"deadline\s*[:\-]?\s*$",
        r"before\s*[:\-]?\s*$",
        r"within\s+\d+\s+days\s+(?:of|from)",
        r"(?:submit|file|furnish|send)\s+(?:by|before|on\s+or\s+before)",
        r"closed?\s+(?:on|from|after)",
        r"(?:application|reply|submission)\s+(?:last|closing)\s+date",
        r"अंतिम\s+(?:तिथि|दिनांक)",  # Hindi: "last date"
    ],
    "EVENT_DATE": [
        r"(?:meeting|hearing|exam|interview|inspection)\s+(?:on|date|scheduled)",
        r"scheduled\s+(?:on|for)\s*[:\-]?\s*$",
        r"(?:held|conducted|convened)\s+on",
    ],
    "REFERENCE_DATE": [
        r"as\s+(?:on|of)\s*[:\-]?\s*$",
        r"refer(?:red|ring|ence)\s+(?:to\s+)?(?:dated?|letter)",
        r"vide\s+(?:order|letter|notification)\s+(?:no\.?\s+)?.*?dated?",
        r"mentioned\s+(?:above|below)\s+dated?",
    ],
}

# Regex patterns to detect date strings in text
DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b',
    # DD Month YYYY (e.g., "5th August, 2026", "19 August 2026")
    r'\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(\d{4})\b',
    # Month DD, YYYY
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b',
    # Hindi months (common transliterations)
    r'\b(\d{1,2})\s+(जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s*,?\s*(\d{4})\b',
]


class DateClassifier:
    """
    Detects all dates in document text and classifies each by its semantic role.
    """

    @classmethod
    def extract_and_classify_dates(
        cls,
        text: str,
        ner_date_spans: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts all date candidates and classifies each.

        Args:
            text: Full document text.
            ner_date_spans: Optional list of NER-detected date spans with
                           'start', 'end', 'text', 'score' keys.

        Returns:
            List of classified date objects.
        """
        dates_found: List[Dict[str, Any]] = []
        seen_positions = set()

        # 1. Extract dates from regex patterns
        for pattern in DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = match.start()
                end = match.end()
                date_text = match.group(0)

                # Avoid duplicates at same position
                pos_key = (start, end)
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)

                # Try to parse into datetime
                parsed_date = cls._parse_date_string(date_text)

                # Classify based on context
                date_type, type_confidence = cls._classify_date_context(text, start, end)

                dates_found.append({
                    "value": date_text,
                    "parsed": parsed_date.isoformat() if parsed_date else None,
                    "type": date_type,
                    "type_confidence": round(type_confidence, 4),
                    "start_char": start,
                    "end_char": end,
                    "source": "regex",
                })

        # 2. Add NER-detected dates that weren't found by regex
        if ner_date_spans:
            for span in ner_date_spans:
                start = span.get("start", 0)
                end = span.get("end", 0)
                pos_key = (start, end)

                # Check if already found by regex (allow some overlap tolerance)
                already_found = False
                for existing_pos in seen_positions:
                    if abs(existing_pos[0] - start) < 5 and abs(existing_pos[1] - end) < 5:
                        already_found = True
                        break

                if already_found:
                    continue

                seen_positions.add(pos_key)
                date_text = span.get("text", text[start:end])
                parsed_date = cls._parse_date_string(date_text)
                date_type, type_confidence = cls._classify_date_context(text, start, end)

                dates_found.append({
                    "value": date_text,
                    "parsed": parsed_date.isoformat() if parsed_date else None,
                    "type": date_type,
                    "type_confidence": round(type_confidence, 4),
                    "start_char": start,
                    "end_char": end,
                    "source": "ner",
                    "ner_score": span.get("score", 0.0),
                })

        # Sort by position in document
        dates_found.sort(key=lambda d: d.get("start_char", 0))

        return dates_found

    @classmethod
    def _classify_date_context(
        cls, text: str, start: int, end: int
    ) -> Tuple[str, float]:
        """
        Classifies a date based on surrounding text context with proximity weighting.
        Returns (date_type, confidence).
        """
        # Extract context window
        ctx_start = max(0, start - CONTEXT_WINDOW)
        ctx_end = min(len(text), end + CONTEXT_WINDOW)

        prefix = text[ctx_start:start].lower()
        suffix = text[end:ctx_end].lower()
        context = prefix + " " + suffix

        best_type = "UNKNOWN_DATE"
        best_score = 0.0

        for date_type, patterns in DATE_CONTEXT_PATTERNS.items():
            for pattern in patterns:
                try:
                    # Check prefix (text before the date)
                    match_prefix = re.search(pattern, prefix, re.IGNORECASE)
                    if match_prefix:
                        # Proximity bonus: higher score if match is closer to the date (end of prefix)
                        dist_to_date = len(prefix) - match_prefix.end()
                        # Base score 0.85, minus small penalty for distance (max penalty ~0.20)
                        score = 0.85 - min(0.20, dist_to_date * 0.005)
                        if score > best_score:
                            best_type = date_type
                            best_score = score

                    # Check wider context (suffix / general context)
                    match_ctx = re.search(pattern, context, re.IGNORECASE)
                    if match_ctx and not match_prefix:
                        score = 0.70
                        if score > best_score:
                            best_type = date_type
                            best_score = score
                except re.error:
                    continue

        # Special heuristic: if the date appears very early in the document
        # (first 500 chars) and near "New Delhi" or header area, likely ISSUE_DATE
        if best_type == "UNKNOWN_DATE" and start < 500:
            header_text = text[:start].lower()
            if any(kw in header_text for kw in ["new delhi", "नई दिल्ली", "notification", "अधिसूचना"]):
                best_type = "ISSUE_DATE"
                best_score = 0.65

        confidence = round(min(0.99, max(0.30, best_score)), 4)
        return best_type, confidence

    @staticmethod
    def _parse_date_string(date_str: str) -> Optional[datetime]:
        """
        Attempts to parse a date string into datetime.
        """
        if not date_str:
            return None

        # Hindi month mapping
        hindi_months = {
            "जनवरी": "January", "फरवरी": "February", "मार्च": "March",
            "अप्रैल": "April", "मई": "May", "जून": "June",
            "जुलाई": "July", "अगस्त": "August", "सितंबर": "September",
            "अक्टूबर": "October", "नवंबर": "November", "दिसंबर": "December",
        }
        parsed_str = date_str
        for hindi, eng in hindi_months.items():
            parsed_str = parsed_str.replace(hindi, eng)

        try:
            return dateutil_parser.parse(parsed_str, fuzzy=True, dayfirst=True)
        except (ValueError, OverflowError, TypeError):
            pass

        return None
