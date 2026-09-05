"""
Corpus Curator for NoticeIE-Gold Annotation Tool

Selects a representative, balanced set of documents for annotation:
  - 10 Notifications
  - 10 Circulars
  - 5  Orders
  - 5  Guidelines
  - 10 Schemes
  - 10 Public / Citizen Notices

Sources:
  - KanoonGPT/indian-legal-documents (streaming, no full download)
  - shrijayan/gov_myscheme (for Schemes)

Output: data/annotations/annotation_corpus.jsonl

Run:
    PYTHONPATH=. python -m src.annotation.corpus_curator
"""

import json
import os
import sys
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

# ──────────────────────────────────────────────────────────────────────────────
# TARGET QUOTAS PER DOCUMENT TYPE
# Maps a canonical label → (quota, list of matching keywords in document_type)
# ──────────────────────────────────────────────────────────────────────────────
QUOTAS: Dict[str, Dict] = {
    "Notification": {
        "quota": 10,
        "keywords": ["notification", "gazette notification", "official notification"],
    },
    "Circular": {
        "quota": 10,
        "keywords": ["circular", "office circular", "trade circular", "master circular"],
    },
    "Order": {
        "quota": 5,
        "keywords": ["order", "standing order", "executive order", "administrative order"],
    },
    "Guidelines": {
        "quota": 5,
        "keywords": ["guidelines", "guideline", "advisory"],
    },
    "Scheme": {
        "quota": 10,
        "keywords": ["scheme", "scheme document", "scheme guidelines"],
    },
    "Public Notice": {
        "quota": 10,
        "keywords": [
            "public notice", "citizen notice", "notice",
            "public advisory", "notice to public"
        ],
    },
}

# Minimum and maximum text lengths for a record to be annotation-worthy
MIN_TEXT_LEN = 200
MAX_TEXT_LEN = 8000

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "annotations", "annotation_corpus.jsonl")


def canonical_type(raw_type: str) -> str:
    """
    Maps a raw document_type string to a canonical quota key.
    Returns None if no quota key matches.
    """
    raw_lower = raw_type.strip().lower()
    for canonical, cfg in QUOTAS.items():
        for kw in cfg["keywords"]:
            if kw in raw_lower:
                return canonical
    return None


def make_doc_id(text: str, title: str) -> str:
    """Stable 8-char ID from content hash."""
    h = hashlib.md5((title + text[:200]).encode()).hexdigest()
    return f"NOTGOLD-{h[:8].upper()}"


def fetch_from_kanoongpt(buckets: Dict[str, List], filled: Dict[str, int]) -> None:
    """
    Streams KanoonGPT/indian-legal-documents and fills quota buckets.
    Stops early once all quotas are met to avoid downloading the full dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed.")
        sys.exit(1)

    all_filled = lambda: all(
        filled[k] >= QUOTAS[k]["quota"] for k in QUOTAS
    )
    if all_filled():
        return

    print("Streaming KanoonGPT/indian-legal-documents …")
    ds = load_dataset(
        "KanoonGPT/indian-legal-documents",
        split="train",
        streaming=True
    )

    checked = 0
    for record in ds:
        if all_filled():
            break

        raw_type = record.get("document_type", "") or ""
        canon = canonical_type(raw_type)
        if canon is None:
            checked += 1
            continue

        if filled[canon] >= QUOTAS[canon]["quota"]:
            checked += 1
            continue

        text = record.get("text", "") or ""
        if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
            checked += 1
            continue

        title = record.get("document_title", "") or ""
        doc_id = record.get("doc_id") or make_doc_id(text, title)

        buckets[canon].append({
            "document_id": doc_id,
            "canonical_type": canon,
            "document_type": raw_type,
            "document_title": title,
            "document_jurisdiction": record.get("document_jurisdiction", ""),
            "issuing_authority": record.get("issuing_authority", ""),
            "issue_date": record.get("issue_date", ""),
            "text": text,
            "source": "KanoonGPT/indian-legal-documents",
        })
        filled[canon] += 1

        if checked % 500 == 0:
            print(f"  Scanned {checked:,} records … "
                  + " | ".join(f"{k}:{filled[k]}/{QUOTAS[k]['quota']}" for k in QUOTAS))
        checked += 1

    print(f"KanoonGPT scan complete. Scanned {checked:,} records.")


def fetch_from_myscheme(buckets: Dict[str, List], filled: Dict[str, int]) -> None:
    """
    Streams shrijayan/gov_myscheme to fill the Scheme quota.
    """
    if filled["Scheme"] >= QUOTAS["Scheme"]["quota"]:
        return

    try:
        from datasets import load_dataset
    except ImportError:
        return

    print("Streaming shrijayan/gov_myscheme …")
    try:
        ds = load_dataset("shrijayan/gov_myscheme", split="train", streaming=True)
    except Exception as e:
        print(f"  Could not load myscheme: {e}")
        return

    for record in ds:
        if filled["Scheme"] >= QUOTAS["Scheme"]["quota"]:
            break

        # Inspect available text fields
        text_candidates = [
            record.get("scheme_description", ""),
            record.get("description", ""),
            record.get("text", ""),
            record.get("content", ""),
        ]
        text = next((t for t in text_candidates if t and len(t) >= MIN_TEXT_LEN), None)
        if text is None:
            continue
        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN]

        title = (
            record.get("scheme_name")
            or record.get("title")
            or record.get("name")
            or "Scheme Document"
        )
        doc_id = make_doc_id(text, title)

        buckets["Scheme"].append({
            "document_id": doc_id,
            "canonical_type": "Scheme",
            "document_type": "Scheme Document",
            "document_title": title,
            "document_jurisdiction": record.get("state", "India"),
            "issuing_authority": record.get("ministry", record.get("department", "")),
            "issue_date": "",
            "text": text,
            "source": "shrijayan/gov_myscheme",
        })
        filled["Scheme"] += 1

    print(f"MyScheme: collected {filled['Scheme']} scheme records.")


def curate() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    buckets: Dict[str, List] = defaultdict(list)
    filled: Dict[str, int] = {k: 0 for k in QUOTAS}

    # 1. Fill from KanoonGPT (covers Notifications, Circulars, Orders, Guidelines, Public Notice)
    fetch_from_kanoongpt(buckets, filled)

    # 2. Fill Scheme quota from MyScheme if not already satisfied
    fetch_from_myscheme(buckets, filled)

    # 3. Write curated corpus
    total = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for canon, docs in buckets.items():
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total += 1

    # 4. Print summary
    print("\n── Annotation Corpus Summary ─────────────────────────")
    for canon in QUOTAS:
        quota = QUOTAS[canon]["quota"]
        got   = filled[canon]
        bar   = "█" * got + "░" * (quota - got)
        print(f"  {canon:<18} {bar}  {got}/{quota}")
    print(f"\n  Total: {total} documents")
    print(f"  Output: {OUTPUT_PATH}")
    print("─────────────────────────────────────────────────────")

    # 5. Write manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "total_documents": total,
        "quotas": {
            k: {"target": QUOTAS[k]["quota"], "collected": filled[k]}
            for k in QUOTAS
        },
        "output_path": OUTPUT_PATH,
    }
    manifest_path = OUTPUT_PATH.replace(".jsonl", "_manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    curate()
