"""
PDF Parser Module for TrustExtract-N (v2)
==========================================
Extracts native text page-by-page using PyMuPDF with intelligent page type detection.
Classifies each page as TEXT, SCANNED, or MIXED to determine OCR requirements.
Conforms to the standardized document output schema.
"""

import os
import uuid
from typing import Dict, Any, List, Optional

try:
    import pymupdf  # PyMuPDF
except ImportError:
    import fitz as pymupdf


class PDFParser:
    """
    PDF parser with intelligent page type detection and structured schema output.
    Determines whether each page is machine-readable text, scanned image, or mixed.
    """

    # Minimum chars for a page to be considered having meaningful text
    MIN_TEXT_THRESHOLD = 100

    # If text extraction yields characters but many are garbage, flag as scanned
    GARBAGE_CHAR_RATIO_THRESHOLD = 0.25

    @classmethod
    def parse_pdf(cls, pdf_path: str, min_chars_threshold: int = 100) -> Dict[str, Any]:
        """
        Parses a PDF file page by page using PyMuPDF with page type detection.

        Returns structured document dictionary:
        {
          "document_id": "...",
          "file_path": "...",
          "pages": [
            {
              "page_number": 1,
              "text": "...",
              "source": "pdf_text" | "low_text_pdf",
              "page_type": "TEXT" | "SCANNED" | "MIXED",
              "char_count": 120,
              "start_offset": 0,
              "end_offset": 120,
              "needs_ocr": bool,
              "has_images": bool,
              "has_fonts": bool,
              "image_count": int,
            }
          ],
          "full_text": "...",
          "document_type_summary": "TEXT" | "SCANNED" | "MIXED"
        }
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = pymupdf.open(pdf_path)
        document_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"

        pages_data: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        current_offset = 0
        page_types_found = set()

        for page_index in range(len(doc)):
            page_num = page_index + 1
            page = doc.load_page(page_index)

            # Extract text
            raw_text = page.get_text("text").strip()
            char_count = len(raw_text)

            # Detect page characteristics
            page_analysis = cls._analyze_page(page, raw_text, min_chars_threshold)
            page_type = page_analysis["page_type"]
            page_types_found.add(page_type)

            source_type = "pdf_text" if page_type == "TEXT" else "low_text_pdf"
            needs_ocr = page_type in ("SCANNED", "MIXED")

            start_offset = current_offset
            end_offset = start_offset + len(raw_text)

            pages_data.append({
                "page_number": page_num,
                "text": raw_text,
                "source": source_type,
                "page_type": page_type,
                "char_count": char_count,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "needs_ocr": needs_ocr,
                "has_images": page_analysis["has_images"],
                "has_fonts": page_analysis["has_fonts"],
                "image_count": page_analysis["image_count"],
                "text_quality": page_analysis["text_quality"],
            })

            full_text_parts.append(raw_text)
            current_offset = end_offset + 2  # Accounting for "\n\n" join separator

        doc.close()
        full_text = "\n\n".join(full_text_parts)

        # Document-level type summary
        if page_types_found == {"TEXT"}:
            doc_type_summary = "TEXT"
        elif page_types_found == {"SCANNED"}:
            doc_type_summary = "SCANNED"
        else:
            doc_type_summary = "MIXED"

        return {
            "document_id": document_id,
            "file_path": pdf_path,
            "pages": pages_data,
            "full_text": full_text,
            "document_type_summary": doc_type_summary,
        }

    @classmethod
    def _analyze_page(
        cls, page, raw_text: str, min_chars_threshold: int
    ) -> Dict[str, Any]:
        """
        Analyzes a single PDF page to determine its type.

        Checks:
        1. Does the page have embedded fonts? (indicates machine-readable text)
        2. Does the page have embedded images? (indicates scanned content)
        3. Is the extracted text meaningful or garbage?
        4. What is the text-to-image ratio?

        Returns analysis dict with page_type classification.
        """
        char_count = len(raw_text)

        # Check for embedded images
        image_list = page.get_images(full=True)
        has_images = len(image_list) > 0
        image_count = len(image_list)

        # Check for embedded fonts
        fonts = page.get_fonts()
        has_fonts = len(fonts) > 0

        # Check text quality — is the extracted text actually readable?
        text_quality = cls._assess_text_quality(raw_text) if raw_text else 0.0

        # Classification logic
        if char_count >= min_chars_threshold and text_quality >= 0.5 and has_fonts:
            # Good amount of quality text with fonts → machine-readable
            if has_images and image_count > 0:
                # Check if images are large (full-page scans) or small (logos, signatures)
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height

                large_images = 0
                for img in image_list:
                    try:
                        xref = img[0]
                        img_rect = page.get_image_rects(xref)
                        if img_rect:
                            for r in img_rect:
                                img_area = r.width * r.height
                                if img_area > page_area * 0.5:
                                    large_images += 1
                    except Exception:
                        continue

                if large_images > 0:
                    page_type = "MIXED"
                else:
                    page_type = "TEXT"  # Small images (logos) don't make it scanned
            else:
                page_type = "TEXT"

        elif char_count < min_chars_threshold and has_images:
            # Low text + images → likely scanned
            page_type = "SCANNED"

        elif char_count >= min_chars_threshold and text_quality < 0.5:
            # Has text but it's garbage → scanned PDF with bad text layer
            page_type = "SCANNED"

        elif char_count < min_chars_threshold and not has_images:
            # Very little text, no images — could be blank or nearly empty
            page_type = "SCANNED"  # Treat as needing OCR attempt

        else:
            page_type = "MIXED"

        return {
            "page_type": page_type,
            "has_images": has_images,
            "has_fonts": has_fonts,
            "image_count": image_count,
            "text_quality": round(text_quality, 4),
        }

    @staticmethod
    def _assess_text_quality(text: str) -> float:
        """
        Heuristic assessment of extracted text quality.
        Returns 0.0 (garbage) to 1.0 (clean readable text).

        Checks:
        - Ratio of alphanumeric + common punctuation to total characters
        - Average word length
        - Presence of normal word patterns
        """
        if not text or len(text) < 10:
            return 0.0

        total_chars = len(text)
        printable_chars = sum(
            1 for c in text
            if c.isalnum() or c.isspace() or c in '.,;:!?@#$%&*()-/\'"'
            or '\u0900' <= c <= '\u097F'  # Devanagari
        )

        printable_ratio = printable_chars / total_chars

        # Check average word length
        words = text.split()
        if not words:
            return 0.0

        avg_word_len = sum(len(w) for w in words) / len(words)
        word_len_score = 1.0 if 2 <= avg_word_len <= 15 else 0.5

        # Check for repeated garbage characters
        import re
        garbage_patterns = len(re.findall(r'[^\x00-\x7F\u0900-\u097F]{5,}', text))
        garbage_penalty = min(1.0, garbage_patterns * 0.15)

        quality = (0.6 * printable_ratio + 0.3 * word_len_score) * (1.0 - garbage_penalty)
        return max(0.0, min(1.0, quality))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        doc_obj = PDFParser.parse_pdf(sys.argv[1])
        print(f"Document ID: {doc_obj['document_id']}")
        print(f"Total Pages: {len(doc_obj['pages'])}")
        print(f"Document Type: {doc_obj['document_type_summary']}")
        for p in doc_obj["pages"]:
            print(f"  Page {p['page_number']}: type={p['page_type']}, chars={p['char_count']}, "
                  f"images={p['image_count']}, needs_ocr={p['needs_ocr']}, quality={p['text_quality']}")
