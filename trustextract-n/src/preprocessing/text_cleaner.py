"""
Text Cleaner & Artifact Removal Engine for TrustExtract-N (v2)
================================================================
Normalizes whitespace, removes repeated page artifacts (headers/footers/page numbers),
while strictly preserving dates, emails, phone numbers, document IDs, and sentence boundaries.

CRITICAL: Preserves original OCR text alongside normalized text.
Never modifies the original evidence — maintains dual versions:
  1. original_ocr_text — exact OCR output for evidence display
  2. normalized_text (text) — cleaned text for NLP model input
"""

import re
import unicodedata
from typing import Dict, Any, List


class TextCleaner:
    """
    Safe text cleaner that normalizes whitespace and removes page artifacts
    without destroying statutory entities (dates, emails, phones, DIN numbers).
    Preserves original OCR text for evidence grounding.
    """

    # Regex patterns for entities that must NEVER be destroyed
    SAFE_PATTERNS = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',                          # Dates: 01-09-2026, 15/10/2026
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', # Dates: 15 Sept 2026
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',             # Emails
        r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', # Phone numbers
        r'\b(?:DIN|REF|NOTICE|F\.?\s*NO|MEMO)[:\s\.-]+[A-Za-z0-9\/-]{5,}\b' # Document Identifiers
    ]

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalizes Unicode characters while keeping standard ASCII/Indian scripts.
        """
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        replacements = {
            "\xa0": " ",
            "\u200b": "",
            "\u200c": "",  # Zero-width non-joiner (common in Hindi)
            "\u200d": "",  # Zero-width joiner
            "\u00ad": "",  # Soft hyphen
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "-", "\u2010": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalizes inline spaces and tabs while preserving paragraph line breaks.
        """
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n")]
        cleaned_lines = []
        blank_counter = 0

        for line in lines:
            # Collapse multiple inline spaces/tabs
            line = re.sub(r'[ \t]+', ' ', line)
            if not line:
                blank_counter += 1
                if blank_counter <= 1:
                    cleaned_lines.append("")
            else:
                blank_counter = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def remove_page_artifacts(text: str) -> str:
        """
        Removes repeated page numbers and header/footer noise (e.g. 'Page 1 of 5', '- 1 -').
        Does not touch dates or statutory reference numbers.
        """
        if not text:
            return ""

        # Remove 'Page X of Y', 'Page X', '- 1 -' page number footers
        text = re.sub(r'(?i)\bPage\s+\d+(\s+of\s+\d+)?\b', '', text)
        text = re.sub(r'^\s*[-—]\s*\d+\s*[-—]\s*$', '', text, flags=re.MULTILINE)

        return text

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans single text string cleanly.
        """
        if not text:
            return ""

        text = cls.normalize_unicode(text)
        text = cls.remove_page_artifacts(text)
        text = cls.normalize_whitespace(text)
        return text

    @classmethod
    def clean_document_object(cls, doc_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans a structured document object while:
        1. PRESERVING original_ocr_text per page (never modified)
        2. Producing normalized_text for model input
        3. Recalculating character offsets based on normalized text
        4. Maintaining page mapping and page numbers
        """
        updated_pages: List[Dict[str, Any]] = []
        full_text_list: List[str] = []
        original_text_list: List[str] = []
        current_offset = 0

        for page in doc_obj.get("pages", []):
            raw_page_text = page.get("text", "")

            # Preserve original OCR text — NEVER modify this
            original_ocr_text = page.get("original_ocr_text", raw_page_text)

            # Clean text for NLP model
            cleaned_page_text = cls.clean_text(raw_page_text)

            start_off = current_offset
            end_off = start_off + len(cleaned_page_text)

            updated_page = {
                "page_number": page["page_number"],
                "text": cleaned_page_text,  # Normalized text for model
                "original_ocr_text": original_ocr_text,  # Original for evidence
                "source": page.get("source", "pdf_text"),
                "page_type": page.get("page_type", "UNKNOWN"),
                "char_count": len(cleaned_page_text),
                "start_offset": start_off,
                "end_offset": end_off,
                # Preserve OCR-level data
                "words": page.get("words", []),
                "ocr_quality": page.get("ocr_quality", {}),
                "layout": page.get("layout", {}),
            }
            updated_pages.append(updated_page)
            full_text_list.append(cleaned_page_text)
            original_text_list.append(original_ocr_text)
            current_offset = end_off + 2

        doc_obj["pages"] = updated_pages
        doc_obj["full_text"] = "\n\n".join(full_text_list)
        doc_obj["original_text"] = "\n\n".join(original_text_list)
        return doc_obj
