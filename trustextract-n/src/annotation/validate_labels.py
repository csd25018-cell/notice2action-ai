"""
validate_labels.py  –  NoticeIE-Gold Annotation Validator
==========================================================
Validates a JSONL annotation file before BIO conversion.

Reports (never silently modifies):
  - invalid character offsets (out of bounds, end <= start)
  - empty spans (start == end or span text is blank)
  - overlapping entity spans within a document
  - unknown labels (not in the seven canonical labels)
  - documents with no entities (informational warning)
  - missing required fields (document_id, text, entities)
  - span text mismatch (annotated "text" differs from doc_text[start:end])

Usage:
    PYTHONPATH=. python -m src.annotation.validate_labels \
        --input data/annotations/noticegold_annotations.jsonl

Or import programmatically:
    from src.annotation.validate_labels import validate_file, ValidationResult
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# The seven canonical entity labels for NoticeIE-Gold.
VALID_LABELS = frozenset(["TITLE", "AUTHORITY", "AUDIENCE", "ELIGIBILITY",
                           "DOCUMENT", "DATE", "CONTACT"])

REQUIRED_TOP_FIELDS = ("document_id", "text", "entities")
REQUIRED_ENTITY_FIELDS = ("label", "start", "end")


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityError:
    """Describes a single entity-level validation error."""
    entity_index: int
    error_type: str   # one of: INVALID_OFFSET, EMPTY_SPAN, UNKNOWN_LABEL,
                      #         TEXT_MISMATCH, OVERLAPPING, MISSING_FIELD
    message: str
    entity: Optional[Dict[str, Any]] = None


@dataclass
class DocumentResult:
    """Validation result for a single document."""
    document_id: str
    line_number: int
    errors: List[EntityError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    entity_count: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


@dataclass
class ValidationResult:
    """Aggregate validation result for a full JSONL file."""
    input_path: str
    document_results: List[DocumentResult] = field(default_factory=list)
    parse_errors: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def total_documents(self) -> int:
        return len(self.document_results)

    @property
    def valid_documents(self) -> int:
        return sum(1 for r in self.document_results if r.is_valid)

    @property
    def invalid_documents(self) -> int:
        return self.total_documents - self.valid_documents

    @property
    def total_errors(self) -> int:
        return sum(len(r.errors) for r in self.document_results)

    @property
    def total_warnings(self) -> int:
        return sum(len(r.warnings) for r in self.document_results)

    @property
    def is_clean(self) -> bool:
        """True only if there are zero errors (warnings allowed)."""
        return self.total_errors == 0 and len(self.parse_errors) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Core validation logic
# ──────────────────────────────────────────────────────────────────────────────

def validate_document(obj: Dict[str, Any], line_number: int) -> DocumentResult:
    """
    Validate a single annotation object.
    Checks for:
      1. Missing top-level fields
      2. Per-entity field presence, label validity, offset sanity, span emptiness,
         and text consistency
      3. Cross-entity overlaps (sorted interval intersection)
    """
    doc_id = obj.get("document_id", f"<unknown@line{line_number}>")
    result = DocumentResult(document_id=doc_id, line_number=line_number)

    # ── 1. Required top-level fields ─────────────────────────────────────────
    for fld in REQUIRED_TOP_FIELDS:
        if fld not in obj:
            result.errors.append(EntityError(
                entity_index=-1,
                error_type="MISSING_FIELD",
                message=f"Top-level field '{fld}' is missing."
            ))

    # If the document doesn't have text or entities at all, stop here
    if "text" not in obj or "entities" not in obj:
        return result

    doc_text: str = obj["text"]
    entities: list = obj["entities"]
    text_len = len(doc_text)
    result.entity_count = len(entities)

    # ── 2. Warn on empty entity list ─────────────────────────────────────────
    if not entities:
        result.warnings.append("Document has no annotated entities.")

    # ── 3. Per-entity checks ─────────────────────────────────────────────────
    valid_spans: List[Tuple[int, int, int]] = []  # (start, end, index)

    for i, ent in enumerate(entities):
        ent_errors: List[EntityError] = []

        # 3a. Required entity fields
        for efld in REQUIRED_ENTITY_FIELDS:
            if efld not in ent:
                ent_errors.append(EntityError(
                    entity_index=i,
                    error_type="MISSING_FIELD",
                    message=f"Entity {i}: missing field '{efld}'.",
                    entity=ent,
                ))

        if ent_errors:
            result.errors.extend(ent_errors)
            continue  # Can't do further checks without all fields

        label = ent["label"]
        start = ent["start"]
        end   = ent["end"]

        # 3b. Unknown label
        if label not in VALID_LABELS:
            result.errors.append(EntityError(
                entity_index=i,
                error_type="UNKNOWN_LABEL",
                message=f"Entity {i}: unknown label '{label}'. "
                        f"Valid labels: {sorted(VALID_LABELS)}.",
                entity=ent,
            ))

        # 3c. Type checks on offsets
        if not isinstance(start, int) or not isinstance(end, int):
            result.errors.append(EntityError(
                entity_index=i,
                error_type="INVALID_OFFSET",
                message=f"Entity {i}: start/end must be integers, "
                        f"got start={type(start).__name__}, end={type(end).__name__}.",
                entity=ent,
            ))
            continue

        # 3d. Empty span
        if end <= start:
            result.errors.append(EntityError(
                entity_index=i,
                error_type="EMPTY_SPAN",
                message=f"Entity {i}: end ({end}) must be strictly greater "
                        f"than start ({start}).",
                entity=ent,
            ))
            continue

        # 3e. Out-of-bounds offsets
        if start < 0 or end > text_len:
            result.errors.append(EntityError(
                entity_index=i,
                error_type="INVALID_OFFSET",
                message=f"Entity {i}: offsets [{start}:{end}] out of bounds "
                        f"(document length = {text_len}).",
                entity=ent,
            ))
            continue

        # 3f. Empty span text
        span_text = doc_text[start:end]
        if not span_text.strip():
            result.errors.append(EntityError(
                entity_index=i,
                error_type="EMPTY_SPAN",
                message=f"Entity {i}: span [{start}:{end}] resolves to "
                        f"blank/whitespace-only text.",
                entity=ent,
            ))
            continue

        # 3g. Annotated 'text' field must match doc_text[start:end] if present
        annotated_text = ent.get("text")
        if annotated_text is not None and annotated_text != span_text:
            result.errors.append(EntityError(
                entity_index=i,
                error_type="TEXT_MISMATCH",
                message=(
                    f"Entity {i}: annotated text field does not match the "
                    f"character span.\n"
                    f"  annotated : {repr(annotated_text[:80])}\n"
                    f"  span text : {repr(span_text[:80])}"
                ),
                entity=ent,
            ))

        valid_spans.append((start, end, i))

    # ── 4. Cross-entity overlap detection ────────────────────────────────────
    # Sort by start; check consecutive pairs for intersection
    valid_spans_sorted = sorted(valid_spans, key=lambda x: x[0])
    for idx in range(len(valid_spans_sorted) - 1):
        s1, e1, i1 = valid_spans_sorted[idx]
        s2, e2, i2 = valid_spans_sorted[idx + 1]
        if s2 < e1:   # overlap: s2 is inside [s1, e1)
            result.errors.append(EntityError(
                entity_index=i1,
                error_type="OVERLAPPING",
                message=(
                    f"Entities {i1} and {i2} overlap: "
                    f"[{s1}:{e1}] overlaps [{s2}:{e2}]."
                ),
            ))

    return result


def validate_file(input_path: str) -> ValidationResult:
    """
    Validate every annotation record in a JSONL file.
    Returns a ValidationResult object (never raises; reports all errors).
    """
    result = ValidationResult(input_path=input_path)
    path = Path(input_path)

    if not path.exists():
        result.parse_errors.append((0, f"File not found: {input_path}"))
        return result

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                result.parse_errors.append((lineno, f"JSON parse error: {exc}"))
                continue
            doc_result = validate_document(obj, lineno)
            result.document_results.append(doc_result)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_error_type(etype: str) -> str:
    icons = {
        "INVALID_OFFSET":  "🔴 INVALID_OFFSET",
        "EMPTY_SPAN":      "🟠 EMPTY_SPAN",
        "UNKNOWN_LABEL":   "🔴 UNKNOWN_LABEL",
        "TEXT_MISMATCH":   "🟡 TEXT_MISMATCH",
        "OVERLAPPING":     "🔴 OVERLAPPING",
        "MISSING_FIELD":   "🔴 MISSING_FIELD",
    }
    return icons.get(etype, etype)


def print_report(result: ValidationResult, verbose: bool = False) -> None:
    """Pretty-print a full validation report to stdout."""
    sep = "─" * 70

    print(f"\n{sep}")
    print(f"  NoticeIE-Gold Annotation Validator")
    print(f"  File: {result.input_path}")
    print(sep)
    print(f"  Documents   : {result.total_documents}")
    print(f"  Valid        : {result.valid_documents}")
    print(f"  Invalid      : {result.invalid_documents}")
    print(f"  Total errors : {result.total_errors}")
    print(f"  Total warnings: {result.total_warnings}")
    print(f"  JSON parse errors: {len(result.parse_errors)}")
    print(sep)

    # JSON parse errors
    for lineno, msg in result.parse_errors:
        print(f"\n  ⛔ Line {lineno}: {msg}")

    # Per-document errors
    for doc_res in result.document_results:
        if not doc_res.errors and not (verbose and doc_res.warnings):
            continue

        print(f"\n  {'✅' if doc_res.is_valid else '❌'} "
              f"[line {doc_res.line_number}] {doc_res.document_id}  "
              f"({doc_res.entity_count} entities)")

        for warn in doc_res.warnings:
            print(f"      ⚠️  {warn}")

        for err in doc_res.errors:
            print(f"      {_fmt_error_type(err.error_type)}")
            for line in err.message.splitlines():
                print(f"        {line}")

    # Summary
    print(f"\n{sep}")
    if result.is_clean:
        print("  ✅  All annotations are valid. Ready for BIO conversion.")
    else:
        print(f"  ❌  {result.total_errors} error(s) found. "
              "Fix before running convert_labels.py.")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate NoticeIE-Gold JSONL annotation files."
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        nargs="+",
        help="One or more JSONL annotation files to validate.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Also show documents that have warnings but no errors.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 even if only warnings are present.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    overall_clean = True

    for input_path in args.input:
        result = validate_file(input_path)
        print_report(result, verbose=args.verbose)

        if not result.is_clean:
            overall_clean = False
        if args.strict and result.total_warnings > 0:
            overall_clean = False

    return 0 if overall_clean else 1


if __name__ == "__main__":
    sys.exit(main())
