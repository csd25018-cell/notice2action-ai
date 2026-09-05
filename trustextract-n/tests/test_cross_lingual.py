import pytest
from transformers import AutoTokenizer

def test_hindi_offset_mapping():
    """
    Tests that MuRIL tokenizer correctly aligns Devanagari character offsets.
    Crucial for cross-lingual zero-shot extraction evidence grounding.
    """
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    
    # "Ministry of Finance" in Hindi
    text = "वित्त मंत्रालय"
    
    inputs = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = inputs["offset_mapping"]
    
    # Verify that the offsets completely cover the string bounds without exceeding
    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(text)
    
    # Reconstruct text from offsets to ensure character slices match
    reconstructed = "".join([text[s:e] for s, e in offsets])
    assert reconstructed.replace(" ", "") == text.replace(" ", "")

def test_cross_lingual_token_alignment():
    """
    Tests that the tokenizer creates comparable token boundaries for 
    English and Hindi dates, ensuring zero-shot capabilities.
    """
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    
    date_en = "15 August 2026"
    date_hi = "15 अगस्त 2026"
    
    tokens_en = tokenizer.tokenize(date_en)
    tokens_hi = tokenizer.tokenize(date_hi)
    
    # Both should identify the numbers as distinct tokens identically
    assert "15" in tokens_en or "##15" in tokens_en or "15" in [t.replace("##", "") for t in tokens_en]
    assert "15" in tokens_hi or "##15" in tokens_hi or "15" in [t.replace("##", "") for t in tokens_hi]
    # MuRIL splits 2026 into '202' and '##6'
    assert "202" in tokens_en and "##6" in tokens_en
    assert "202" in tokens_hi and "##6" in tokens_hi
