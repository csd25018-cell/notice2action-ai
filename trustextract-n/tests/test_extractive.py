import pytest
from src.summarization.extractive import summarize_extractive, split_sentences_with_offsets

def test_split_sentences():
    text = "Hello world. This is a test! How are you?\nI am fine."
    sentences = split_sentences_with_offsets(text)
    
    assert len(sentences) == 4
    assert sentences[0][2] == "Hello world."
    assert sentences[1][2] == "This is a test!"
    assert sentences[2][2] == "How are you?"
    assert sentences[3][2] == "I am fine."


def test_extractive_summarization_basic():
    doc_text = (
        "This is a boring first sentence. "
        "The Ministry of Finance issued a new policy today. "
        "It is irrelevant what the weather is. "
        "The deadline for compliance is 30 September 2026. "
        "This is the end of the document."
    )
    
    extractions = [
        {"value": "Ministry of Finance", "character_start": 37, "character_end": 56, "page": 1},
        {"value": "30 September 2026", "character_start": 129, "character_end": 146, "page": 1}
    ]
    
    result = summarize_extractive(
        document_text=doc_text,
        extractions=extractions,
        min_sentences=2,
        max_sentences=2
    )
    
    # It should pick exactly the 2nd and 4th sentences because they have score 1
    assert len(result["sentences"]) == 2
    
    summary = result["summary_text"]
    assert "Ministry of Finance issued a new policy today" in summary
    assert "deadline for compliance is 30 September 2026" in summary
    assert "boring first sentence" not in summary
    assert "weather" not in summary
    
    # Chronological ordering check
    assert result["sentences"][0]["sentence_index"] == 1
    assert result["sentences"][1]["sentence_index"] == 3


def test_fallback_to_first_sentences():
    # If no entities are found, it should grab the first few sentences
    doc_text = "Sentence one. Sentence two. Sentence three. Sentence four."
    
    result = summarize_extractive(
        document_text=doc_text,
        extractions=[], # No extractions
        min_sentences=2,
        max_sentences=4
    )
    
    assert len(result["sentences"]) == 2
    assert "Sentence one." in result["summary_text"]
    assert "Sentence two." in result["summary_text"]
    assert "Sentence three." not in result["summary_text"]


def test_ignores_low_confidence_extractions():
    doc_text = "Sentence one contains an entity. Sentence two contains a bad entity."
    
    extractions = [
        {"value": "entity", "character_start": 25, "character_end": 31, "status": "HIGH_CONFIDENCE"},
        {"value": "bad entity", "character_start": 57, "character_end": 67, "status": "LOW_CONFIDENCE"}
    ]
    
    result = summarize_extractive(
        document_text=doc_text,
        extractions=extractions,
        min_sentences=1,
        max_sentences=1
    )
    
    assert len(result["sentences"]) == 1
    assert "Sentence one" in result["summary_text"]
    assert "Sentence two" not in result["summary_text"]
