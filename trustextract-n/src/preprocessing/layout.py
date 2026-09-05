"""
Layout Reconstruction for TrustExtract-N
==========================================
Reconstructs document layout from OCR word-level outputs.
Groups words into lines, lines into paragraphs, and identifies
structural elements (headers, footers, signature areas).
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class LayoutReconstructor:
    """
    Reconstructs document structure from OCR word data.
    Preserves bounding box coordinates throughout.
    """

    def __init__(
        self,
        line_y_tolerance: float = 15.0,
        paragraph_gap_factor: float = 2.0,
        header_zone_pct: float = 0.12,
        footer_zone_pct: float = 0.10,
    ):
        """
        Args:
            line_y_tolerance: Max vertical pixel distance for words on same line.
            paragraph_gap_factor: Gap between lines > factor * avg_line_height → new paragraph.
            header_zone_pct: Top percentage of page considered header zone.
            footer_zone_pct: Bottom percentage of page considered footer zone.
        """
        self.line_y_tolerance = line_y_tolerance
        self.paragraph_gap_factor = paragraph_gap_factor
        self.header_zone_pct = header_zone_pct
        self.footer_zone_pct = footer_zone_pct

    def reconstruct(
        self,
        words: List[Dict[str, Any]],
        page_width: int,
        page_height: int,
        page_number: int = 1,
    ) -> Dict[str, Any]:
        """
        Reconstructs layout from OCR word data.

        Args:
            words: List of word dicts with 'text', 'bbox' [x1,y1,x2,y2], 'confidence'.
            page_width: Page width in pixels.
            page_height: Page height in pixels.
            page_number: Page number in the document.

        Returns:
            Structured layout dict with lines, paragraphs, and zones.
        """
        if not words:
            return {
                "page_number": page_number,
                "page_size": [page_width, page_height],
                "lines": [],
                "paragraphs": [],
                "header_text": "",
                "footer_text": "",
                "body_text": "",
                "full_text": "",
                "zones": {},
            }

        # 1. Group words into lines by Y-coordinate proximity
        lines = self._group_words_into_lines(words)

        # 2. Sort lines top to bottom
        lines.sort(key=lambda ln: ln["bbox"][1])

        # 3. Classify zones: header, body, footer, signature
        header_y = page_height * self.header_zone_pct
        footer_y = page_height * (1.0 - self.footer_zone_pct)

        header_lines = []
        body_lines = []
        footer_lines = []

        for line in lines:
            line_center_y = (line["bbox"][1] + line["bbox"][3]) / 2
            if line_center_y < header_y:
                line["zone"] = "header"
                header_lines.append(line)
            elif line_center_y > footer_y:
                line["zone"] = "footer"
                footer_lines.append(line)
            else:
                line["zone"] = "body"
                body_lines.append(line)

        # 4. Group body lines into paragraphs
        paragraphs = self._group_lines_into_paragraphs(body_lines)

        # 5. Detect signature area (bottom body lines with specific patterns)
        signature_zone = self._detect_signature_zone(footer_lines + body_lines[-3:] if body_lines else footer_lines)

        # 6. Build text representations
        header_text = "\n".join(ln["text"] for ln in header_lines)
        footer_text = "\n".join(ln["text"] for ln in footer_lines)
        body_text = "\n\n".join(p["text"] for p in paragraphs)
        full_text = "\n\n".join(filter(None, [header_text, body_text, footer_text]))

        return {
            "page_number": page_number,
            "page_size": [page_width, page_height],
            "lines": lines,
            "paragraphs": paragraphs,
            "header_text": header_text,
            "footer_text": footer_text,
            "body_text": body_text,
            "full_text": full_text,
            "zones": {
                "header": {"line_count": len(header_lines), "text": header_text},
                "body": {"line_count": len(body_lines), "paragraph_count": len(paragraphs)},
                "footer": {"line_count": len(footer_lines), "text": footer_text},
                "signature": signature_zone,
            },
        }

    def _group_words_into_lines(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups words into lines based on Y-coordinate proximity.
        Words within `line_y_tolerance` pixels vertically are on the same line.
        """
        if not words:
            return []

        # Sort by y-center, then x
        sorted_words = sorted(words, key=lambda w: (
            (w["bbox"][1] + w["bbox"][3]) / 2,
            w["bbox"][0]
        ))

        lines: List[List[Dict]] = []
        current_line: List[Dict] = [sorted_words[0]]
        current_y = (sorted_words[0]["bbox"][1] + sorted_words[0]["bbox"][3]) / 2

        for word in sorted_words[1:]:
            word_y = (word["bbox"][1] + word["bbox"][3]) / 2
            if abs(word_y - current_y) <= self.line_y_tolerance:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
                current_y = word_y

        if current_line:
            lines.append(current_line)

        # Build line objects
        line_objects = []
        for line_words in lines:
            # Sort words left to right within line
            line_words.sort(key=lambda w: w["bbox"][0])

            # Compute line bounding box
            x1 = min(w["bbox"][0] for w in line_words)
            y1 = min(w["bbox"][1] for w in line_words)
            x2 = max(w["bbox"][2] for w in line_words)
            y2 = max(w["bbox"][3] for w in line_words)

            line_text = " ".join(w["text"] for w in line_words)
            avg_conf = sum(w.get("confidence", 0.0) for w in line_words) / len(line_words)

            line_objects.append({
                "text": line_text,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(avg_conf, 4),
                "words": line_words,
                "word_count": len(line_words),
            })

        return line_objects

    def _group_lines_into_paragraphs(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups lines into paragraphs based on vertical spacing.
        """
        if not lines:
            return []

        if len(lines) == 1:
            return [{
                "text": lines[0]["text"],
                "bbox": lines[0]["bbox"],
                "line_count": 1,
                "lines": lines,
            }]

        # Calculate average line height
        line_heights = []
        for line in lines:
            h = line["bbox"][3] - line["bbox"][1]
            if h > 0:
                line_heights.append(h)
        avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 20

        # Group by gaps
        paragraphs: List[List[Dict]] = []
        current_para: List[Dict] = [lines[0]]

        for i in range(1, len(lines)):
            prev_line = lines[i - 1]
            curr_line = lines[i]
            gap = curr_line["bbox"][1] - prev_line["bbox"][3]

            if gap > avg_line_height * self.paragraph_gap_factor:
                paragraphs.append(current_para)
                current_para = [curr_line]
            else:
                current_para.append(curr_line)

        if current_para:
            paragraphs.append(current_para)

        # Build paragraph objects
        para_objects = []
        for para_lines in paragraphs:
            x1 = min(ln["bbox"][0] for ln in para_lines)
            y1 = min(ln["bbox"][1] for ln in para_lines)
            x2 = max(ln["bbox"][2] for ln in para_lines)
            y2 = max(ln["bbox"][3] for ln in para_lines)

            para_text = "\n".join(ln["text"] for ln in para_lines)

            para_objects.append({
                "text": para_text,
                "bbox": [x1, y1, x2, y2],
                "line_count": len(para_lines),
                "lines": para_lines,
            })

        return para_objects

    @staticmethod
    def _detect_signature_zone(candidate_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detects signature area from bottom lines using keyword patterns.
        """
        signature_keywords = [
            "signature", "signed", "authorized", "signatory",
            "director", "commissioner", "officer", "secretary",
            "sd/-", "(sd/-)", "for and on behalf",
            "name:", "designation:", "seal",
        ]

        signature_text_parts = []
        for line in candidate_lines:
            text_lower = line.get("text", "").lower()
            if any(kw in text_lower for kw in signature_keywords):
                signature_text_parts.append(line.get("text", ""))

        return {
            "detected": len(signature_text_parts) > 0,
            "text": "\n".join(signature_text_parts),
        }
