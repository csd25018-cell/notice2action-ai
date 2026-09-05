"""
extractive.py – Extractive Summarization for TrustExtract-N
===========================================================
Creates a high-information summary of a notice by scoring and selecting 
the original sentences that contain the most extracted entities and key terms.

Guarantees zero generation/paraphrasing by returning exact source strings.
Supports length modes: SHORT (1-2 sentences), STANDARD (3-5 sentences), DETAILED (5-8 sentences).
"""

import re
from typing import List, Dict, Any, Tuple, Optional

# Keywords that boost sentence importance
NOTICE_KEYWORDS = {
    # English keywords
    "notification", "circular", "show cause", "public notice", "advisory", "directive",
    "deadline", "last date", "closing date", "effective from", "w.e.f.", "dated",
    "eligible", "eligibility", "qualification", "appl", "submit", "submission",
    "commence", "directed", "instructed", "admission", "fee", "penalty", "required",
    "documents", "enclosure", "contact", "email", "phone", "website",
    # Hindi keywords
    "अधिसूचना", "परिपत्र", "अंतिम तिथि", "पात्रता", "आवेदन", "जमा", "आवश्यक", "दस्तावेज़",
    "सम्पर्क", "कार्यालय", "मंत्रालय", "विभाग", "निर्देश"
}

BOILERPLATE_PATTERNS = [
    r'^copy\s+to\b', r'^copy\s+forwarded\s+to\b', r'^प्रतिलिपि', r'^sd/-',
    r'^page\s+\d+', r'^\d+$', r'^[a-z0-9\/-]{10,}$'
]


def split_sentences_with_offsets(text: str) -> List[Tuple[int, int, str]]:
    """
    Splits text into sentences while perfectly preserving character offsets.
    Uses basic punctuation and newline boundaries.
    """
    sentences = []
    for match in re.finditer(r'[^.!?\n]+[.!?\n]*', text):
        start, end = match.span()
        span_text = text[start:end].strip()
        if span_text:
            sentences.append((start, end, span_text))
    return sentences


def summarize_extractive(
    document_text: str,
    extractions: List[Dict[str, Any]],
    min_sentences: int = 2,
    max_sentences: int = 4,
    summary_length: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates an extractive summary by selecting the highest-information 
    sentences from the original text.
    
    Args:
        document_text: The full source text.
        extractions: List of accepted entity extraction dictionaries.
        min_sentences: Minimum number of sentences to return.
        max_sentences: Maximum number of sentences to return.
        summary_length: Optional length string ("SHORT", "STANDARD", "DETAILED").
        
    Returns:
        Dict containing the full summary string and metadata per sentence.
    """
    if not document_text.strip():
        return {"summary_text": "", "sentences": [], "confidence": 0.0}

    # Map summary_length if provided
    if summary_length:
        mode = summary_length.upper().split()[0]
        if mode == "SHORT":
            min_sentences, max_sentences = 1, 2
        elif mode == "STANDARD":
            min_sentences, max_sentences = 3, 5
        elif mode == "DETAILED":
            min_sentences, max_sentences = 5, 8

    sentences = split_sentences_with_offsets(document_text)
    
    if len(sentences) <= min_sentences:
        selected = sentences
        return {
            "summary_text": " ".join([s[2] for s in selected]),
            "sentences": [
                {
                    "sentence_index": i,
                    "text": s[2],
                    "page_number": 1,
                    "entity_score": 1.0,
                    "start_char": s[0],
                    "end_char": s[1]
                } 
                for i, s in enumerate(selected)
            ],
            "confidence": 0.95
        }
        
    sent_scores = [0.0] * len(sentences)
    sent_pages = [None] * len(sentences)
    num_sentences = len(sentences)
    
    # 1. Score sentences based on entity overlap
    has_entity_matches = False
    for ext in extractions:
        if ext.get("status") == "LOW_CONFIDENCE" or not ext.get("value"):
            continue
            
        start_char = ext.get("character_start")
        end_char = ext.get("character_end")
        page = ext.get("page")
        
        if start_char is None or end_char is None:
            val = ext.get("value", "")
            if val and len(val) > 2:
                for i, (s, e, text) in enumerate(sentences):
                    if val.lower() in text.lower():
                        sent_scores[i] += 2.0
                        has_entity_matches = True
                        if page is not None and sent_pages[i] is None:
                            sent_pages[i] = page
        else:
            for i, (s, e, text) in enumerate(sentences):
                overlap_start = max(s, start_char)
                overlap_end = min(e, end_char)
                if overlap_start < overlap_end:
                    sent_scores[i] += 3.0
                    has_entity_matches = True
                    if page is not None and sent_pages[i] is None:
                        sent_pages[i] = page

    # 2. Position & Keyword scoring + Boilerplate Penalties
    for i, (s, e, text) in enumerate(sentences):
        text_lower = text.lower()

        # Keyword boost
        for kw in NOTICE_KEYWORDS:
            if kw in text_lower:
                sent_scores[i] += 1.0

        # Position boost (first 3 sentences and last sentence get bonus)
        if i < 3:
            sent_scores[i] += 1.5 * (3 - i)
        elif i >= num_sentences - 2:
            sent_scores[i] += 1.0

        # Penalize boilerplate / signatures / copy-to lines
        for bp in BOILERPLATE_PATTERNS:
            if re.search(bp, text_lower):
                sent_scores[i] -= 5.0
                break

    # 3. Rank sentences by score
    scored_sentences = [
        (sent_scores[i], -i, i, sentences[i][2], sent_pages[i] or 1, sentences[i][0], sentences[i][1]) 
        for i in range(len(sentences))
    ]
    
    scored_sentences.sort(reverse=True)
    
    # Determine target selection size: use max_sentences when entity matches exist, otherwise min_sentences
    target_k = max_sentences if has_entity_matches else min_sentences

    selected = scored_sentences[:target_k]
    
    if len(selected) < min_sentences:
        selected_text_set = {s[3] for s in selected}
        for s in scored_sentences:
            if len(selected) >= min_sentences:
                break
            if s[3] not in selected_text_set:
                selected.append(s)
                selected_text_set.add(s[3])
                
    # Sort selected sentences chronologically by index
    selected.sort(key=lambda x: x[2])
    
    summary_sentences = []
    for score, _neg_i, idx, text, page, start_off, end_off in selected:
        summary_sentences.append({
            "sentence_index": idx,
            "text": text,
            "page_number": page,
            "entity_score": round(max(0.0, score), 2),
            "start_char": start_off,
            "end_char": end_off
        })
        
    full_summary_text = " ".join([s["text"] for s in summary_sentences])
    
    # Calculate confidence based on coverage
    top_scores = [s[0] for s in selected]
    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    summary_conf = min(0.98, max(0.60, round(0.60 + min(0.38, avg_score * 0.05), 2)))

    return {
        "summary_text": full_summary_text,
        "sentences": summary_sentences,
        "confidence": summary_conf
    }
