"""
convert_labels.py  –  Character-Span → BIO Token Labels
=========================================================
Converts NoticeIE-Gold character-span annotations to BIO (Begin-Inside-Outside)
token-level labels aligned to a HuggingFace tokenizer's subword offsets.

Tokenizer choice:
  google/muril-base-cased  — Multilingual Universal Representations for Indian
  Languages; covers 17 Indian languages + English and is purpose-built for the
  government notice domain this project targets.  Alternatively configurable via
  --model.

BIO scheme:
  O           → outside all entities
  B-LABEL     → first token of an entity span
  I-LABEL     → continuation token inside an entity span

Subword alignment rules:
  - Only the FIRST subword of a word that falls inside a span gets B-/I-.
  - Continuation subwords (word_ids == previous) receive I-LABEL.
  - Special tokens ([CLS], [SEP], padding) receive label -100 (ignored in loss).
  - Long documents are split into overlapping windows (stride configurable).
  - Overlapping / invalid spans are REJECTED — the validator must pass first.

Inputs:
  data/annotations/*.jsonl   (any file matching the glob)

Outputs:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl

Each output line:
{
  "document_id": "...",
  "chunk_index": 0,
  "tokens": ["▁Min", "istry", "▁of", ...],
  "input_ids": [...],
  "attention_mask": [...],
  "labels": ["O", "B-TITLE", "I-TITLE", "O", ...],
  "label_ids": [-100, 0, 4, 1, ...]    # -100 for special tokens
}

Usage:
    PYTHONPATH=. python -m src.annotation.convert_labels \\
        --input  data/annotations/noticegold_annotations.jsonl \\
        --outdir data/processed \\
        --model  google/muril-base-cased \\
        --max-length 512 \\
        --stride 128 \\
        --val-ratio 0.15 \\
        --test-ratio 0.15 \\
        --seed 42
"""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Label schema
# ──────────────────────────────────────────────────────────────────────────────

ENTITY_LABELS = ["TITLE", "AUTHORITY", "AUDIENCE", "ELIGIBILITY",
                 "DOCUMENT", "DATE", "CONTACT"]

# Build ordered label list: O first, then B-/I- for each entity type.
# The index in this list is the integer label_id used during training.
LABEL_LIST: List[str] = ["O"]
for _ent in ENTITY_LABELS:
    LABEL_LIST.append(f"B-{_ent}")
    LABEL_LIST.append(f"I-{_ent}")

LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}

# Tokens that should be ignored during loss computation (special tokens)
IGNORE_LABEL_ID = -100
OUTSIDE_LABEL = "O"

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConvertedChunk:
    """
    A single tokenized window of a document ready for model training.
    Long documents produce multiple chunks (sliding window).
    """
    document_id: str
    chunk_index: int
    tokens: List[str]
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[str]              # human-readable BIO tag per token
    label_ids: List[int]           # integer IDs (-100 for special tokens)
    doc_label_id: int              # integer ID for document type

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionError:
    document_id: str
    line_number: int
    entity_index: int
    error_type: str
    message: str


@dataclass
class ConversionResult:
    model_name: str
    input_files: List[str]
    chunks: List[ConvertedChunk] = field(default_factory=list)
    errors: List[ConversionError] = field(default_factory=list)
    skipped_documents: int = 0

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Span → BIO alignment
# ──────────────────────────────────────────────────────────────────────────────

def _sort_and_validate_spans(
    entities: List[Dict[str, Any]],
    text_len: int,
    document_id: str,
    line_number: int,
) -> Tuple[List[Tuple[int, int, str]], List[ConversionError]]:
    """
    Sort spans by start position, rejects invalid / overlapping ones.
    Returns (valid_sorted_spans, errors).
    Each span is (start, end, label).
    """
    valid: List[Tuple[int, int, str]] = []
    errors: List[ConversionError] = []

    for i, ent in enumerate(entities):
        label = ent.get("label", "")
        start = ent.get("start")
        end   = ent.get("end")

        # Type / bounds
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(ConversionError(
                document_id=document_id, line_number=line_number,
                entity_index=i, error_type="INVALID_OFFSET",
                message=f"start/end must be int; got {type(start).__name__}/{type(end).__name__}"
            ))
            continue
        if end <= start:
            errors.append(ConversionError(
                document_id=document_id, line_number=line_number,
                entity_index=i, error_type="EMPTY_SPAN",
                message=f"end ({end}) <= start ({start})"
            ))
            continue
        if start < 0 or end > text_len:
            errors.append(ConversionError(
                document_id=document_id, line_number=line_number,
                entity_index=i, error_type="INVALID_OFFSET",
                message=f"[{start}:{end}] out of bounds (text_len={text_len})"
            ))
            continue
        if f"B-{label}" not in LABEL2ID:
            errors.append(ConversionError(
                document_id=document_id, line_number=line_number,
                entity_index=i, error_type="UNKNOWN_LABEL",
                message=f"Unknown label '{label}'"
            ))
            continue

        valid.append((start, end, label))

    # Sort and detect overlaps
    valid.sort(key=lambda x: x[0])
    clean: List[Tuple[int, int, str]] = []
    prev_end = -1
    for (s, e, lbl) in valid:
        if s < prev_end:
            errors.append(ConversionError(
                document_id=document_id, line_number=line_number,
                entity_index=-1, error_type="OVERLAPPING",
                message=f"Span [{s}:{e}] ('{lbl}') overlaps previous span ending at {prev_end}. Rejected."
            ))
            continue
        clean.append((s, e, lbl))
        prev_end = e

    return clean, errors


def _char_to_bio(
    char_spans: List[Tuple[int, int, str]],
    offset_mapping: List[Tuple[int, int]],
    word_ids: List[Optional[int]],
    sequence_length: int,
) -> List[str]:
    """
    Map character-level entity spans to per-token BIO labels.

    Algorithm:
      For each token position t:
        - If word_id is None → special token → 'O' (will be masked to -100 later)
        - Find which entity span (if any) contains token_char_start
        - If this token is the first subword of a new word entering a span → B-LABEL
        - If it's a continuation subword of a word already inside the span → I-LABEL
        - Otherwise → O

    Handles:
      - Subword tokens (word_ids tracks word boundaries)
      - Punctuation attached to words
      - Tokens that span a boundary (attributed to whichever span claims the start)
    """
    bio_labels = [OUTSIDE_LABEL] * sequence_length

    # Build a lookup: for each char position, which label is it inside?
    # We'll use interval scanning instead of a char-array to avoid O(n*m) cost.
    span_idx = 0  # index into char_spans (they're sorted)

    for t in range(sequence_length):
        wid = word_ids[t]
        if wid is None:
            # Special token: keep O (will be masked to -100 later)
            continue

        tok_char_start, tok_char_end = offset_mapping[t]
        if tok_char_start == tok_char_end:
            # Zero-length token (e.g., padding artefact)
            continue

        # Advance span_idx past spans that end before this token starts
        while span_idx < len(char_spans) and char_spans[span_idx][1] <= tok_char_start:
            span_idx += 1

        if span_idx >= len(char_spans):
            # No more spans; rest is O
            break

        s, e, lbl = char_spans[span_idx]

        # Check if this token overlaps the current span
        if tok_char_start >= s and tok_char_start < e:
            # Determine B vs I:
            # It's B if this is the first subword of a word (i.e., previous token
            # had a different word_id or was a special token) AND the word starts
            # at or after the span start.
            prev_wid = word_ids[t - 1] if t > 0 else None
            if prev_wid != wid:
                # First subword of this word
                bio_labels[t] = f"B-{lbl}"
            else:
                # Continuation subword within the same word
                bio_labels[t] = f"I-{lbl}"

    return bio_labels


# ──────────────────────────────────────────────────────────────────────────────
# Chunking (sliding window for long documents)
# ──────────────────────────────────────────────────────────────────────────────

def _convert_document(
    obj: Dict[str, Any],
    line_number: int,
    tokenizer,
    max_length: int,
    stride: int,
) -> Tuple[List[ConvertedChunk], List[ConversionError]]:
    """
    Convert one annotation document into one or more ConvertedChunk objects.
    Uses tokenizer(return_offsets_mapping=True) for precise char alignment.
    Sliding window with `stride` overlap for documents longer than max_length tokens.
    """
    document_id = obj.get("document_id", f"<unknown@line{line_number}>")
    text: str = obj.get("text", "")
    entities: List[Dict] = obj.get("entities", [])
    
    # Auto-align strings to spans for Active Learning format
    if not entities:
        for key, label in [("document_title", "TITLE"), ("issuing_authority", "AUTHORITY"), ("issue_date", "DATE")]:
            val = obj.get(key)
            if val and isinstance(val, str) and val in text:
                start = text.find(val)
                entities.append({"label": label, "start": start, "end": start + len(val)})
                
    doc_type = obj.get("document_type", "Unknown")
    doc_types = ["Circular", "Notification", "Ordinance", "Draft Rule"]
    doc_label_id = doc_types.index(doc_type) if doc_type in doc_types else 0

    # Validate and sort spans first
    char_spans, span_errors = _sort_and_validate_spans(
        entities, len(text), document_id, line_number
    )

    # Tokenize with offset mapping (no truncation yet; we window manually)
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,  # we'll handle [CLS]/[SEP] per window below
        truncation=False,
        return_attention_mask=False,
    )

    all_input_ids: List[int]          = encoding["input_ids"]
    all_offsets:   List[Tuple[int,int]] = encoding["offset_mapping"]

    # Compute word_ids manually from offsets + tokenizer
    # (word_ids() is available on fast tokenizers; we replicate it for safety)
    all_word_ids = _compute_word_ids(all_offsets, text)

    # Split into windows of (max_length - 2) tokens, stride apart
    window_size = max_length - 2  # reserve 2 for [CLS] / [SEP]
    chunks: List[ConvertedChunk] = []
    n_tokens = len(all_input_ids)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    start = 0
    chunk_index = 0
    while start < n_tokens or chunk_index == 0:
        end = min(start + window_size, n_tokens)

        window_ids     = all_input_ids[start:end]
        window_offsets = all_offsets[start:end]
        window_wids    = all_word_ids[start:end]

        # Add special tokens
        full_ids     = [cls_id] + window_ids + [sep_id]
        full_offsets = [(0, 0)] + window_offsets + [(0, 0)]
        full_wids    = [None] + window_wids + [None]
        seq_len      = len(full_ids)

        # Compute per-token BIO labels
        bio_labels = _char_to_bio(char_spans, full_offsets, full_wids, seq_len)

        # Convert BIO to IDs, masking special tokens
        label_ids: List[int] = []
        for t, lbl in enumerate(bio_labels):
            if full_wids[t] is None:
                label_ids.append(IGNORE_LABEL_ID)
            else:
                label_ids.append(LABEL2ID[lbl])

        # Decode tokens for readability
        tokens = tokenizer.convert_ids_to_tokens(full_ids)

        chunks.append(ConvertedChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            tokens=tokens,
            input_ids=full_ids,
            attention_mask=[1] * seq_len,
            labels=bio_labels,
            label_ids=label_ids,
            doc_label_id=doc_label_id,
        ))

        chunk_index += 1
        if end >= n_tokens:
            break
        start = end - stride   # overlapping stride

    return chunks, span_errors


def _compute_word_ids(
    offsets: List[Tuple[int, int]],
    text: str,
) -> List[Optional[int]]:
    """
    Assigns a word_id to each token based on whitespace boundaries in the
    original text. Used when word_ids() is not available on the tokenizer.
    Tokens with (0,0) offset (special tokens not added here) → None.
    """
    word_ids: List[Optional[int]] = []
    # Build a char→word_id map
    char_word = {}
    wid = 0
    in_word = False
    for ci, ch in enumerate(text):
        if ch.strip():
            if not in_word:
                wid += 1
                in_word = True
            char_word[ci] = wid
        else:
            in_word = False

    for (cs, ce) in offsets:
        if cs == ce == 0:
            word_ids.append(None)
        else:
            word_ids.append(char_word.get(cs))

    return word_ids


# ──────────────────────────────────────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_annotations(input_paths: List[str]) -> List[Tuple[int, Dict[str, Any], str]]:
    """
    Load all annotation objects from one or more JSONL files.
    Returns list of (line_number, obj, source_file).
    """
    records: List[Tuple[int, Dict, str]] = []
    for fpath in input_paths:
        p = Path(fpath)
        if not p.exists():
            print(f"  WARNING: Input file not found: {fpath}", file=sys.stderr)
            continue
        with open(p, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append((lineno, obj, str(p)))
                except json.JSONDecodeError as exc:
                    print(f"  ERROR: {fpath}:{lineno}: JSON parse error: {exc}", file=sys.stderr)
    return records


def _split(
    chunks: List[ConvertedChunk],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    """
    Stratified split by document_id to avoid chunk leakage across splits.
    """
    # Group chunks by document_id
    from collections import defaultdict
    doc_chunks: Dict[str, List] = defaultdict(list)
    for c in chunks:
        doc_chunks[c.document_id].append(c)

    doc_ids = list(doc_chunks.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_test = max(1, round(n * test_ratio))
    n_val  = max(1, round(n * val_ratio))
    n_train = n - n_val - n_test

    train_ids = set(doc_ids[:n_train])
    val_ids   = set(doc_ids[n_train:n_train + n_val])
    test_ids  = set(doc_ids[n_train + n_val:])

    train = [c for did in train_ids for c in doc_chunks[did]]
    val   = [c for did in val_ids   for c in doc_chunks[did]]
    test  = [c for did in test_ids  for c in doc_chunks[did]]

    return train, val, test


def _write_split(chunks: List[ConvertedChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Wrote {len(chunks):>5} chunks → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion pipeline
# ──────────────────────────────────────────────────────────────────────────────

def convert(
    input_paths: List[str],
    outdir: str,
    model_name: str,
    max_length: int,
    stride: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> ConversionResult:
    """
    Full conversion pipeline.
    Validates spans, tokenizes with offset mapping, writes train/val/test splits.
    """
    from transformers import AutoTokenizer

    result = ConversionResult(model_name=model_name, input_files=input_paths)

    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    records = load_annotations(input_paths)
    print(f"  Loaded {len(records)} annotation records from {len(input_paths)} file(s)\n")

    for lineno, obj, src_file in records:
        doc_id = obj.get("document_id", f"<unknown@line{lineno}>")

        # Skip documents with no entities (informational only)
        entities = obj.get("entities", [])
        has_string_labels = any(obj.get(k) for k in ["document_title", "issuing_authority", "issue_date"])
        
        if not entities and not has_string_labels:
            print(f"  ⚠  Skipping {doc_id} (no entities)")
            result.skipped_documents += 1
            continue

        chunks, errors = _convert_document(obj, lineno, tokenizer, max_length, stride)

        if errors:
            # Report errors but do NOT silently drop the whole document.
            # Drop only the specific invalid entities (already handled inside).
            for err in errors:
                print(
                    f"  ❌ {doc_id} entity#{err.entity_index}: "
                    f"[{err.error_type}] {err.message}",
                    file=sys.stderr
                )
            result.errors.extend(errors)

        result.chunks.extend(chunks)
        print(f"  ✅ {doc_id:40s} → {len(chunks)} chunk(s)")

    # Split
    train, val, test = _split(result.chunks, val_ratio, test_ratio, seed)

    out = Path(outdir)
    _write_split(train, out / "train.jsonl")
    _write_split(val,   out / "val.jsonl")
    _write_split(test,  out / "test.jsonl")

    # Write label map
    label_map_path = out / "label_map.json"
    with open(label_map_path, "w") as lmf:
        json.dump({
            "model": model_name,
            "labels": LABEL_LIST,
            "label2id": LABEL2ID,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            "ignore_label_id": IGNORE_LABEL_ID,
        }, lmf, indent=2)
    print(f"\n  Label map → {label_map_path}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert NoticeIE-Gold character-span annotations to BIO token labels."
    )
    p.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="Input JSONL annotation file(s).",
    )
    p.add_argument(
        "--outdir", "-o", default="data/processed",
        help="Output directory for train/val/test.jsonl. Default: data/processed",
    )
    p.add_argument(
        "--model", "-m", default="google/muril-base-cased",
        help="HuggingFace model name for tokenizer. Default: google/muril-base-cased",
    )
    p.add_argument(
        "--max-length", type=int, default=512,
        help="Max token window size including special tokens. Default: 512",
    )
    p.add_argument(
        "--stride", type=int, default=128,
        help="Overlap stride between consecutive windows. Default: 128",
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of documents for validation split. Default: 0.15",
    )
    p.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Fraction of documents for test split. Default: 0.15",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits. Default: 42",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    result = convert(
        input_paths=args.input,
        outdir=args.outdir,
        model_name=args.model,
        max_length=args.max_length,
        stride=args.stride,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Conversion complete")
    print(f"  Total chunks   : {len(result.chunks)}")
    print(f"  Skipped docs   : {result.skipped_documents}")
    print(f"  Span errors    : {len(result.errors)}")
    print(f"  Label list ({len(LABEL_LIST)}): {LABEL_LIST}")
    print(sep)

    return 0 if result.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
