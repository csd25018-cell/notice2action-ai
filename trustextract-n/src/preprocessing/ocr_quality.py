"""
OCR Quality Scorer for TrustExtract-N
======================================
Computes OCR quality metrics per page and document level.
Used to:
1. Determine if OCR output is trustworthy
2. Feed into multi-signal confidence calculation
3. Alert users about low-quality scans
"""

import re
from typing import Dict, Any, List, Optional

try:
    from langdetect import detect as detect_language, DetectorFactory
    DetectorFactory.seed = 42  # Deterministic language detection
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# Devanagari Unicode range: 0x0900 - 0x097F
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')
# Common ASCII printable (letters, digits, common punctuation)
VALID_CHARS = re.compile(r'[a-zA-Z0-9\s.,;:!?@#$%&*()\-/\'\"\u0900-\u097F]')


class OCRQualityScorer:
    """
    Evaluates OCR output quality using multiple signals.
    Returns a composite quality score between 0.0 (garbage) and 1.0 (perfect).
    """

    @classmethod
    def score_page(
        cls,
        words: List[Dict[str, Any]],
        page_width: int = 0,
        page_height: int = 0,
    ) -> Dict[str, Any]:
        """
        Scores OCR quality for a single page.

        Args:
            words: List of word dicts with 'text', 'confidence', 'bbox' keys.
            page_width: Width of the page in pixels.
            page_height: Height of the page in pixels.

        Returns:
            Dict with individual metrics and composite score.
        """
        if not words:
            return {
                "avg_confidence": 0.0,
                "token_validity_ratio": 0.0,
                "text_density": 0.0,
                "abnormal_char_ratio": 0.0,
                "language_detected": "unknown",
                "language_confidence": 0.0,
                "word_count": 0,
                "ocr_quality_score": 0.0,
                "quality_label": "NO_TEXT",
            }

        # 1. Average OCR confidence
        confidences = [w.get("confidence", 0.0) for w in words]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # 2. Token validity ratio — how many words look like real words
        valid_tokens = 0
        total_tokens = len(words)
        for w in words:
            text = w.get("text", "").strip()
            if len(text) >= 2 and cls._is_valid_token(text):
                valid_tokens += 1
        token_validity_ratio = valid_tokens / total_tokens if total_tokens > 0 else 0.0

        # 3. Text density — characters per page area
        full_text = " ".join(w.get("text", "") for w in words)
        total_chars = len(full_text.replace(" ", ""))
        if page_width > 0 and page_height > 0:
            page_area = page_width * page_height
            # Normalize: typical A4 at 250 DPI ≈ 2075×2925 pixels ≈ 6M pixels
            # A well-filled page has ~3000-5000 chars → density ~0.0005-0.0008
            text_density = min(1.0, (total_chars / page_area) * 1500)
        else:
            # Fallback: estimate based on character count
            text_density = min(1.0, total_chars / 3000)

        # 4. Abnormal character ratio
        abnormal_chars = 0
        for ch in full_text:
            if ch != ' ' and not VALID_CHARS.match(ch):
                abnormal_chars += 1
        abnormal_char_ratio = abnormal_chars / max(1, total_chars)

        # 5. Language detection
        lang_detected = "unknown"
        lang_confidence = 0.0
        if full_text.strip() and LANGDETECT_AVAILABLE:
            try:
                lang_detected = detect_language(full_text)
                # langdetect doesn't provide confidence directly,
                # but we can estimate from detection stability
                lang_confidence = 0.8  # Default moderate confidence
                if lang_detected in ("en", "hi"):
                    lang_confidence = 0.9  # Expected languages
            except Exception:
                lang_detected = "unknown"
                lang_confidence = 0.0
        elif full_text.strip():
            # Simple heuristic without langdetect
            devanagari_count = len(DEVANAGARI_RANGE.findall(full_text))
            ascii_alpha = sum(1 for c in full_text if c.isascii() and c.isalpha())
            if devanagari_count > ascii_alpha:
                lang_detected = "hi"
                lang_confidence = 0.7
            elif ascii_alpha > 0:
                lang_detected = "en"
                lang_confidence = 0.7

        # 6. Composite score
        # Weighted combination of signals
        composite = (
            0.40 * avg_confidence
            + 0.25 * token_validity_ratio
            + 0.15 * text_density
            + 0.10 * (1.0 - abnormal_char_ratio)
            + 0.10 * lang_confidence
        )
        composite = max(0.0, min(1.0, composite))

        # Quality label
        if composite >= 0.80:
            quality_label = "GOOD"
        elif composite >= 0.60:
            quality_label = "MODERATE"
        elif composite >= 0.40:
            quality_label = "POOR"
        else:
            quality_label = "VERY_POOR"

        return {
            "avg_confidence": round(avg_confidence, 4),
            "token_validity_ratio": round(token_validity_ratio, 4),
            "text_density": round(text_density, 4),
            "abnormal_char_ratio": round(abnormal_char_ratio, 4),
            "language_detected": lang_detected,
            "language_confidence": round(lang_confidence, 4),
            "word_count": total_tokens,
            "ocr_quality_score": round(composite, 4),
            "quality_label": quality_label,
        }

    @classmethod
    def score_document(
        cls,
        page_scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregates per-page OCR quality into document-level score.
        """
        if not page_scores:
            return {
                "ocr_confidence": 0.0,
                "language": "unknown",
                "quality_label": "NO_TEXT",
                "page_count": 0,
                "pages_good": 0,
                "pages_poor": 0,
            }

        # Weighted average by word count
        total_words = sum(ps.get("word_count", 0) for ps in page_scores)
        if total_words > 0:
            weighted_score = sum(
                ps.get("ocr_quality_score", 0.0) * ps.get("word_count", 0)
                for ps in page_scores
            ) / total_words
        else:
            weighted_score = 0.0

        # Majority language
        lang_counts: Dict[str, int] = {}
        for ps in page_scores:
            lang = ps.get("language_detected", "unknown")
            wc = ps.get("word_count", 1)
            lang_counts[lang] = lang_counts.get(lang, 0) + wc
        primary_language = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"

        # Check if bilingual
        languages = {ps.get("language_detected", "unknown") for ps in page_scores if ps.get("word_count", 0) > 10}
        languages.discard("unknown")
        is_bilingual = len(languages) > 1

        pages_good = sum(1 for ps in page_scores if ps.get("quality_label") in ("GOOD", "MODERATE"))
        pages_poor = sum(1 for ps in page_scores if ps.get("quality_label") in ("POOR", "VERY_POOR"))

        if weighted_score >= 0.80:
            quality_label = "GOOD"
        elif weighted_score >= 0.60:
            quality_label = "MODERATE"
        elif weighted_score >= 0.40:
            quality_label = "POOR"
        else:
            quality_label = "VERY_POOR"

        return {
            "ocr_confidence": round(weighted_score, 4),
            "language": primary_language,
            "is_bilingual": is_bilingual,
            "languages_detected": sorted(languages) if languages else [primary_language],
            "quality_label": quality_label,
            "page_count": len(page_scores),
            "pages_good": pages_good,
            "pages_poor": pages_poor,
        }

    @staticmethod
    def _is_valid_token(text: str) -> bool:
        """
        Heuristic check if a token looks like a real word vs OCR garbage.
        """
        if not text:
            return False

        # All-digits are valid (dates, numbers)
        if text.isdigit():
            return True

        # Devanagari tokens are valid
        if DEVANAGARI_RANGE.search(text):
            return True

        # ASCII alpha tokens of reasonable length
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count >= max(1, len(text) * 0.5):
            return True

        # Common patterns (dates, reference numbers)
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', text):
            return True

        return False
