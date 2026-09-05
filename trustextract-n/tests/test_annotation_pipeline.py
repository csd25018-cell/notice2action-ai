"""
Unit tests for:
  - src/annotation/validate_labels.py
  - src/annotation/convert_labels.py

Run:
    PYTHONPATH=. pytest tests/test_annotation_pipeline.py -v
"""

import json
import os
import tempfile

import pytest

from src.annotation.validate_labels import (
    VALID_LABELS,
    validate_document,
    validate_file,
)
from src.annotation.convert_labels import (
    LABEL_LIST,
    LABEL2ID,
    _sort_and_validate_spans,
    _char_to_bio,
    _compute_word_ids,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_doc(doc_id="DOC-001", text="Ministry of Finance notification 2024",
             entities=None):
    if entities is None:
        entities = []
    return {"document_id": doc_id, "text": text, "entities": entities}


def entity(label, start, end, text=None):
    e = {"label": label, "start": start, "end": end}
    if text is not None:
        e["text"] = text
    return e


# ──────────────────────────────────────────────────────────────────────────────
# validate_labels tests
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateDocument:

    def test_valid_document_passes(self):
        text = "Ministry of Finance"
        doc = make_doc(text=text, entities=[entity("AUTHORITY", 0, 19, text)])
        result = validate_document(doc, line_number=1)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_top_level_field(self):
        doc = {"document_id": "D1", "entities": []}  # missing 'text'
        result = validate_document(doc, line_number=1)
        error_types = [e.error_type for e in result.errors]
        assert "MISSING_FIELD" in error_types

    def test_invalid_offset_end_before_start(self):
        text = "hello world"
        doc = make_doc(text=text, entities=[entity("DATE", 5, 2)])
        result = validate_document(doc, line_number=1)
        error_types = [e.error_type for e in result.errors]
        assert "EMPTY_SPAN" in error_types

    def test_offset_out_of_bounds(self):
        text = "short"
        doc = make_doc(text=text, entities=[entity("TITLE", 0, 100)])
        result = validate_document(doc, line_number=1)
        error_types = [e.error_type for e in result.errors]
        assert "INVALID_OFFSET" in error_types

    def test_unknown_label_rejected(self):
        text = "some notice text"
        doc = make_doc(text=text, entities=[entity("INVALID_LABEL", 0, 4)])
        result = validate_document(doc, line_number=1)
        error_types = [e.error_type for e in result.errors]
        assert "UNKNOWN_LABEL" in error_types

    def test_overlapping_spans_detected(self):
        text = "Government of India notice 2024"
        # spans [0:10] and [5:15] overlap
        doc = make_doc(text=text, entities=[
            entity("AUTHORITY", 0, 10),
            entity("TITLE", 5, 15),
        ])
        result = validate_document(doc, line_number=1)
        error_types = [e.error_type for e in result.errors]
        assert "OVERLAPPING" in error_types

    def test_empty_whitespace_span_rejected(self):
        text = "hello   world"
        doc = make_doc(text=text, entities=[entity("DATE", 5, 8)])  # "   "
        result = validate_document(doc, line_number=1)
        error_types = [e.error_type for e in result.errors]
        assert "EMPTY_SPAN" in error_types

    def test_text_mismatch_detected(self):
        text = "Notification 2024"
        doc = make_doc(text=text, entities=[
            entity("TITLE", 0, 12, text="WRONG_TEXT")
        ])
        result = validate_document(doc, line_number=1)
        error_types = [e.error_type for e in result.errors]
        assert "TEXT_MISMATCH" in error_types

    def test_no_entities_produces_warning(self):
        doc = make_doc(text="Some notice", entities=[])
        result = validate_document(doc, line_number=1)
        assert result.is_valid          # no errors
        assert len(result.warnings) > 0  # but a warning

    def test_adjacent_spans_not_overlapping(self):
        text = "Ministry of Finance 2024"
        doc = make_doc(text=text, entities=[
            entity("AUTHORITY", 0, 19),   # "Ministry of Finance"
            entity("DATE", 20, 24),       # "2024"
        ])
        result = validate_document(doc, line_number=1)
        assert result.is_valid

    def test_all_seven_labels_accepted(self):
        text = "A " * 20  # 40 chars total
        entities_list = []
        offset = 0
        for label in sorted(VALID_LABELS):
            entities_list.append(entity(label, offset, offset + 1))
            offset += 2
        doc = make_doc(text=text, entities=entities_list)
        result = validate_document(doc, line_number=1)
        # Only unknown-label errors would matter; here there should be none
        unknown = [e for e in result.errors if e.error_type == "UNKNOWN_LABEL"]
        assert len(unknown) == 0


class TestValidateFile:

    def test_missing_file_returns_parse_error(self):
        result = validate_file("/nonexistent/path.jsonl")
        assert len(result.parse_errors) == 1

    def test_valid_jsonl_file(self, tmp_path):
        text = "Ministry of Finance issued notification"
        record = make_doc(text=text, entities=[
            entity("AUTHORITY", 0, 19, "Ministry of Finance"),
            entity("DOCUMENT", 27, 39, "notification"),
        ])
        f = tmp_path / "ann.jsonl"
        f.write_text(json.dumps(record) + "\n", encoding="utf-8")
        result = validate_file(str(f))
        assert result.is_clean
        assert result.total_documents == 1

    def test_mixed_valid_and_invalid(self, tmp_path):
        valid_rec = make_doc("D1", "Finance notice", [entity("TITLE", 0, 7, "Finance")])
        invalid_rec = make_doc("D2", "Short", [entity("DATE", 0, 100)])  # OOB
        f = tmp_path / "mixed.jsonl"
        f.write_text(
            json.dumps(valid_rec) + "\n" + json.dumps(invalid_rec) + "\n",
            encoding="utf-8"
        )
        result = validate_file(str(f))
        assert result.total_documents == 2
        assert result.valid_documents == 1
        assert result.invalid_documents == 1


# ──────────────────────────────────────────────────────────────────────────────
# convert_labels tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLabelSchema:

    def test_label_list_starts_with_O(self):
        assert LABEL_LIST[0] == "O"

    def test_all_entity_types_have_B_and_I(self):
        for ent in ["TITLE", "AUTHORITY", "AUDIENCE", "ELIGIBILITY",
                    "DOCUMENT", "DATE", "CONTACT"]:
            assert f"B-{ent}" in LABEL_LIST
            assert f"I-{ent}" in LABEL_LIST

    def test_label2id_consistent(self):
        for i, lbl in enumerate(LABEL_LIST):
            assert LABEL2ID[lbl] == i


class TestSortAndValidateSpans:

    def test_valid_span_returned(self):
        text = "hello world"
        spans, errors = _sort_and_validate_spans(
            [{"label": "TITLE", "start": 0, "end": 5}],
            len(text), "D1", 1
        )
        assert len(errors) == 0
        assert spans == [(0, 5, "TITLE")]

    def test_overlapping_span_rejected(self):
        text = "hello world foo"
        _, errors = _sort_and_validate_spans(
            [
                {"label": "TITLE",  "start": 0, "end": 8},
                {"label": "AUTHORITY", "start": 5, "end": 11},
            ],
            len(text), "D1", 1
        )
        assert any(e.error_type == "OVERLAPPING" for e in errors)

    def test_out_of_bounds_rejected(self):
        _, errors = _sort_and_validate_spans(
            [{"label": "DATE", "start": 0, "end": 999}],
            10, "D1", 1
        )
        assert any(e.error_type == "INVALID_OFFSET" for e in errors)

    def test_spans_sorted_by_start(self):
        text = "aaa bbb ccc ddd"
        spans, errors = _sort_and_validate_spans(
            [
                {"label": "CONTACT",  "start": 8, "end": 11},  # "ccc"
                {"label": "DATE",     "start": 0, "end": 3},   # "aaa"
                {"label": "DOCUMENT", "start": 12, "end": 15}, # "ddd"
            ],
            len(text), "D1", 1
        )
        assert len(errors) == 0
        assert spans[0][0] == 0    # sorted by start
        assert spans[1][0] == 8
        assert spans[2][0] == 12


class TestCharToBio:

    def _make_mock_offsets(self, words):
        """Build fake offset_mapping and word_ids for space-separated words."""
        offsets = [(0, 0)]   # CLS
        wids = [None]
        pos = 0
        for wid, w in enumerate(words):
            offsets.append((pos, pos + len(w)))
            wids.append(wid)
            pos += len(w) + 1  # +1 for space
        offsets.append((0, 0))  # SEP
        wids.append(None)
        return offsets, wids

    def test_single_span_bio(self):
        # text: "Ministry of Finance notification"
        # words: Ministry(0-8) of(9-11) Finance(12-19) notification(20-32)
        #
        # BIO per-word rule:
        #   The FIRST subword of EACH word that starts inside the span gets B-.
        #   Only continuation subwords of the *same* word get I-.
        #   Since our mock tokenizer gives each word a unique word_id,
        #   every word boundary resets to B-.
        words = ["Ministry", "of", "Finance", "notification"]
        text = " ".join(words)
        offsets, wids = self._make_mock_offsets(words)
        seq_len = len(offsets)

        # Annotate "Ministry of Finance" as AUTHORITY: [0, 19)
        char_spans = [(0, 19, "AUTHORITY")]

        bio = _char_to_bio(char_spans, offsets, wids, seq_len)
        # CLS (special token) → O
        assert bio[0] == "O"
        # "Ministry" — first subword of first word in span → B-AUTHORITY
        assert bio[1] == "B-AUTHORITY"
        # "of" — first subword of a *new* word inside span → B-AUTHORITY
        assert bio[2] == "B-AUTHORITY"
        # "Finance" — first subword of a *new* word inside span → B-AUTHORITY
        assert bio[3] == "B-AUTHORITY"
        # "notification" starts at char 20, outside [0:19] → O
        assert bio[4] == "O"
        # SEP (special token) → O
        assert bio[5] == "O"

    def test_outside_tokens_are_O(self):
        words = ["hello", "world"]
        offsets, wids = self._make_mock_offsets(words)
        seq_len = len(offsets)
        bio = _char_to_bio([], offsets, wids, seq_len)
        assert all(lbl == "O" for lbl in bio)


class TestComputeWordIds:

    def test_simple_words(self):
        text = "hello world"
        offsets = [(0, 5), (6, 11)]
        wids = _compute_word_ids(offsets, text)
        # Both should be non-None and the second word > first
        assert wids[0] is not None
        assert wids[1] is not None
        assert wids[0] != wids[1]

    def test_special_token_offset_zero_zero(self):
        text = "hello"
        offsets = [(0, 0), (0, 5), (0, 0)]  # CLS, hello, SEP
        wids = _compute_word_ids(offsets, text)
        assert wids[0] is None
        assert wids[2] is None
        assert wids[1] is not None
