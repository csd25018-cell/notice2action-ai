"""
NoticeIE-Gold: Manual Annotation Tool for TrustExtract-N
Streamlit app for annotating Indian government notices with entity spans.

Run with:
    streamlit run src/annotation/annotator.py
(from inside the trustextract-n/ directory)

Annotates seven entity types:
  TITLE, AUTHORITY, AUDIENCE, ELIGIBILITY, DOCUMENT, DATE, CONTACT

Output schema (JSONL):
{
  "document_id": "...",
  "text": "...",
  "entities": [
    {"label": "TITLE", "start": 0, "end": 25, "text": "..."}
  ]
}
"""

import os
import sys
import json

import streamlit as st

# Ensure src/ is importable when running from trustextract-n/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.annotation.annotation_store import AnnotationStore, ANNOTATION_LABELS

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Curated corpus built by corpus_curator.py — balanced by document type
CORPUS_PATH = os.path.join(BASE_DIR, "data", "annotations", "annotation_corpus.jsonl")
ANNOTATIONS_PATH = os.path.join(BASE_DIR, "data", "annotations", "noticegold_annotations.jsonl")

LABEL_COLORS = {
    "TITLE":       "#06b6d4",
    "AUTHORITY":   "#8b5cf6",
    "AUDIENCE":    "#10b981",
    "ELIGIBILITY": "#f59e0b",
    "DOCUMENT":    "#f43f5e",
    "DATE":        "#3b82f6",
    "CONTACT":     "#ec4899",
}

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NoticeIE-Gold Annotator",
    page_icon="🏷️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0f1117; color: #f1f5f9; }
.block-container { padding-top: 1rem; }
h1, h2, h3 { color: #e2e8f0; }
.entity-chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 14px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 4px;
    color: #000;
}
.annotation-row {
    background: #1e293b;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 4px solid #334155;
}
.doc-text-box {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.7;
    white-space: pre-wrap;
    max-height: 460px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DOCUMENTS FROM data/processed/ JSONL
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_corpus() -> list[dict]:
    """
    Loads the curated annotation corpus from data/annotations/annotation_corpus.jsonl.
    This file is built by corpus_curator.py with balanced document-type quotas:
      10 Notifications, 10 Circulars, 5 Orders, 5 Guidelines,
      10 Schemes, 10 Public/Citizen Notices.
    Returns an empty list (handled below) if the corpus hasn't been built yet.
    """
    if not os.path.exists(CORPUS_PATH):
        return []

    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                docs.append({
                    "document_id": item.get("document_id", f"DOC-{len(docs)+1:04d}"),
                    "title": item.get("document_title", "Untitled"),
                    "canonical_type": item.get("canonical_type", ""),
                    "document_type": item.get("document_type", ""),
                    "issuing_authority": item.get("issuing_authority", ""),
                    "issue_date": item.get("issue_date", ""),
                    "source": item.get("source", ""),
                    "text": item.get("text", "")
                })
            except json.JSONDecodeError:
                continue
    return docs

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
store = AnnotationStore(ANNOTATIONS_PATH)

if "records" not in st.session_state:
    st.session_state.records = store.load_all()
if "doc_index" not in st.session_state:
    st.session_state.doc_index = 0


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def save_records():
    store.save_all(st.session_state.records)


def highlight_text_with_entities(text: str, entities: list[dict]) -> str:
    """
    Returns HTML string with entity spans highlighted inline.
    """
    if not entities:
        return f'<div class="doc-text-box">{text}</div>'

    sorted_entities = sorted(entities, key=lambda e: e["start"])
    html = ""
    cursor = 0
    for ent in sorted_entities:
        s, e, label = ent["start"], ent["end"], ent["label"]
        if s < cursor:
            continue  # skip overlapping spans
        color = LABEL_COLORS.get(label, "#888")
        html += text[cursor:s]
        html += (
            f'<mark style="background:{color}22; border-bottom: 2px solid {color}; '
            f'border-radius:3px; padding:0 2px;" title="{label}">'
            f'{text[s:e]}'
            f'<sup style="font-size:0.6rem;color:{color};font-weight:700"> [{label}]</sup></mark>'
        )
        cursor = e
    html += text[cursor:]
    return f'<div class="doc-text-box">{html}</div>'


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏷️ NoticeIE-Gold — Annotation Tool")
st.caption("Manually annotate Indian government notices for entity extraction (TrustExtract-N)")

corpus = load_corpus()
if not corpus:
    st.error(
        "**Annotation corpus not found.**\n\n"
        "Build it first by running:\n\n"
        "```bash\n"
        "cd trustextract-n\n"
        "PYTHONPATH=. python -m src.annotation.corpus_curator\n"
        "```\n\n"
        f"Expected file: `{CORPUS_PATH}`"
    )
    st.stop()

total_docs = len(corpus)

# ── Top navigation ──────────────────────────────────────────────────────────
nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
with nav_col1:
    if st.button("⬅ Previous", use_container_width=True):
        st.session_state.doc_index = max(0, st.session_state.doc_index - 1)
        st.rerun()
with nav_col2:
    doc_index_sel = st.slider(
        "Document",
        min_value=1,
        max_value=total_docs,
        value=st.session_state.doc_index + 1,
        label_visibility="collapsed"
    )
    if doc_index_sel - 1 != st.session_state.doc_index:
        st.session_state.doc_index = doc_index_sel - 1
        st.rerun()
with nav_col3:
    if st.button("Next ➡", use_container_width=True):
        st.session_state.doc_index = min(total_docs - 1, st.session_state.doc_index + 1)
        st.rerun()

st.markdown(f"**Document {st.session_state.doc_index + 1} of {total_docs}**")

# ── Current document ────────────────────────────────────────────────────────
doc = corpus[st.session_state.doc_index]
doc_id   = doc["document_id"]
doc_text = doc["text"]

st.markdown(f"#### 📄 {doc.get('title', doc_id)}")
st.caption(
    f"ID: `{doc_id}` "
    f"| **{doc.get('canonical_type', doc.get('document_type', '—'))}** "
    f"| {doc.get('document_type', '')} "
    f"| 🏛 {doc.get('issuing_authority', '—')} "
    f"| 📅 {doc.get('issue_date', '—')} "
    f"| Source: {doc.get('source', '—')}"
)

# ── Annotated version of text ────────────────────────────────────────────────
existing = st.session_state.records.get(doc_id, {})
existing_entities = existing.get("entities", [])

st.markdown("##### Document Text (annotated spans highlighted)")
st.markdown(
    highlight_text_with_entities(doc_text, existing_entities),
    unsafe_allow_html=True
)

# Label legend
st.markdown(
    " ".join(
        f'<span class="entity-chip" style="background-color:{LABEL_COLORS[l]}">{l}</span>'
        for l in ANNOTATION_LABELS
    ),
    unsafe_allow_html=True
)

st.divider()

# ── Annotation Form ─────────────────────────────────────────────────────────
col_form, col_ann = st.columns([1.2, 1])

with col_form:
    st.markdown("#### ➕ Add Annotation")
    st.caption("Enter the character span and select the entity label.")

    annotation_method = st.radio(
        "Span selection method",
        ["By text search (auto-locate)", "By character offsets (manual)"],
        horizontal=True
    )

    if annotation_method == "By text search (auto-locate)":
        search_text = st.text_input("Paste or type the exact span text to annotate:")
        label_sel = st.selectbox("Entity Label", ["(skip — do not annotate)"] + ANNOTATION_LABELS)
        occurrence = st.number_input("Occurrence (1 = first, 2 = second, etc.)", min_value=1, value=1, step=1)

        if search_text and label_sel != "(skip — do not annotate)":
            found_start = -1
            idx = 0
            for _ in range(int(occurrence)):
                idx = doc_text.find(search_text, idx)
                if idx == -1:
                    break
                found_start = idx
                idx += len(search_text)

            if found_start == -1:
                st.warning(f"Text not found (occurrence {int(occurrence)}).")
            else:
                found_end = found_start + len(search_text)
                st.success(f"Located span → chars `[{found_start}:{found_end}]`")
                st.code(doc_text[found_start:found_end], language=None)

                if st.button("✅ Add Annotation", type="primary", use_container_width=True):
                    st.session_state.records = store.add_entity(
                        doc_id, doc_text, label_sel, found_start, found_end, st.session_state.records
                    )
                    save_records()
                    st.rerun()

    else:  # Manual offset entry
        label_sel = st.selectbox("Entity Label", ["(skip — do not annotate)"] + ANNOTATION_LABELS)
        char_start = st.number_input("Start character index", min_value=0, value=0, step=1)
        char_end = st.number_input("End character index", min_value=0, value=10, step=1)

        if char_end > char_start:
            st.code(f"Preview: {doc_text[int(char_start):int(char_end)]}", language=None)

        if label_sel != "(skip — do not annotate)":
            if st.button("✅ Add Annotation", type="primary", use_container_width=True):
                if int(char_end) > int(char_start):
                    st.session_state.records = store.add_entity(
                        doc_id, doc_text, label_sel, int(char_start), int(char_end), st.session_state.records
                    )
                    save_records()
                    st.rerun()
                else:
                    st.error("End index must be greater than start index.")

# ── Existing Annotations Panel ───────────────────────────────────────────────
with col_ann:
    annotated_count = len([r for r in st.session_state.records.values() if r.get("entities")])
    st.markdown(f"#### 📋 Existing Annotations ({len(existing_entities)} spans)")
    st.caption(f"Total annotated documents so far: **{annotated_count}**")

    if not existing_entities:
        st.info("No annotations yet for this document.")
    else:
        for i, ent in enumerate(existing_entities):
            color = LABEL_COLORS.get(ent["label"], "#666")
            with st.container():
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(
                        f'<div class="annotation-row">'
                        f'<span class="entity-chip" style="background-color:{color}">{ent["label"]}</span>'
                        f' <code>[{ent["start"]}:{ent["end"]}]</code><br>'
                        f'<small>"{ent["text"]}"</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_b:
                    if st.button("🗑️", key=f"del_{i}", help="Delete this annotation"):
                        st.session_state.records = store.delete_entity(doc_id, i, st.session_state.records)
                        save_records()
                        st.rerun()

st.divider()

# ── Export / Stats sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Annotation Progress")
    total_annotated = len([r for r in st.session_state.records.values() if r.get("entities")])
    total_entities  = sum(len(r.get("entities", [])) for r in st.session_state.records.values())

    st.metric("Documents Annotated", total_annotated)
    st.metric("Total Entity Spans", total_entities)
    st.metric("Total Documents", total_docs)

    st.markdown("---")
    st.markdown("### 📁 Output File")
    st.code(ANNOTATIONS_PATH, language=None)

    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        st.download_button(
            "⬇️ Download JSONL",
            data=content,
            file_name="noticegold_annotations.jsonl",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 🏷️ Label Reference")
    for label in ANNOTATION_LABELS:
        color = LABEL_COLORS[label]
        st.markdown(
            f'<span class="entity-chip" style="background-color:{color}">{label}</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("NoticeIE-Gold Annotation Tool · TrustExtract-N")
