import streamlit as st
import os
import json
import glob
from pathlib import Path

st.set_page_config(page_title="Active Learning Inbox", page_icon=":material/inbox:", layout="wide")

st.title(":material/inbox: Active Learning Admin Inbox")
st.markdown("Review documents where the model abstained (`LOW_CONFIDENCE`). Correct the fields to automatically add them back to the training dataset for continuous improvement.")

inbox_dir = Path("data/active_learning/inbox")
corpus_file = Path("data/annotations/annotation_corpus.jsonl")

if not inbox_dir.exists():
    inbox_dir.mkdir(parents=True, exist_ok=True)

pending_files = list(inbox_dir.glob("*.json"))

if not pending_files:
    st.success(":material/check_circle: Inbox is empty! All abstentions have been reviewed.")
    st.stop()

st.sidebar.metric("Pending Reviews", len(pending_files))

# Select a document to review
selected_file = st.selectbox("Select Document to Review", pending_files, format_func=lambda x: x.stem)

if selected_file:
    with open(selected_file, "r") as f:
        doc_data = json.load(f)
        
    doc_id = doc_data.get("document_id")
    text = doc_data.get("text", "")
    predictions = doc_data.get("predictions", {})
    
    st.markdown("### Document Text")
    st.text_area("Original Text", text, height=300, disabled=True)
    
    st.markdown("### Correction Form")
    st.info("The model abstained on the following fields. Please extract the exact string from the text above.")
    
    with st.form(key="correction_form"):
        corrected_fields = {}
        for field, data in predictions.items():
            if data.get("status") == "LOW_CONFIDENCE":
                corrected_fields[field] = st.text_input(f"Correct {field.upper()}", value="")
                
        submit = st.form_submit_button("Submit Correction", type="primary")
        
        if submit:
            # 1. Append to corpus
            corpus_entry = {
                "document_id": doc_id,
                "document_title": corrected_fields.get("title", ""),
                "issuing_authority": corrected_fields.get("authority", ""),
                "issue_date": corrected_fields.get("date", ""),
                "text": text,
                "source": "ActiveLearning_HumanCorrection"
            }
            
            with open(corpus_file, "a") as f:
                f.write(json.dumps(corpus_entry) + "\n")
                
            # 2. Delete from inbox
            selected_file.unlink()
            
            st.success(f"Successfully reviewed {doc_id} and added to training corpus!")
            st.rerun()
