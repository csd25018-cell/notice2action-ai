"""
Runner script for 01_dataset_exploration.ipynb logic.
Executes statistical analysis over processed notice corpus and generates dataset_statistics.json.
"""

import os
import json
import pandas as pd

def run_exploration():
    BASE_DIR = os.path.abspath(".")
    KANOONGPT_FILE = os.path.join(BASE_DIR, "data", "processed", "kanoongpt_selected.jsonl")
    MYSCHEME_FILE = os.path.join(BASE_DIR, "data", "processed", "myscheme_selected.jsonl")
    
    # Check fallback paths if executed from trustextract-n folder
    if not os.path.exists(KANOONGPT_FILE):
        KANOONGPT_FILE = os.path.join(BASE_DIR, "trustextract-n", "data", "processed", "kanoongpt_selected.jsonl")
        MYSCHEME_FILE = os.path.join(BASE_DIR, "trustextract-n", "data", "processed", "myscheme_selected.jsonl")

    STATS_OUTPUT_FILE_1 = os.path.join(BASE_DIR, "data", "processed", "dataset_statistics.json")
    STATS_OUTPUT_FILE_2 = os.path.join(BASE_DIR, "trustextract-n", "data", "processed", "dataset_statistics.json")

    records = []
    if os.path.exists(KANOONGPT_FILE):
        with open(KANOONGPT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                item['source_dataset'] = 'KanoonGPT'
                records.append(item)

    if os.path.exists(MYSCHEME_FILE):
        with open(MYSCHEME_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                item['source_dataset'] = 'MyScheme'
                item['document_title'] = item.get('filename')
                records.append(item)

    df = pd.DataFrame(records)
    total_records = len(df)

    if total_records == 0:
        print("No records found to analyze.")
        return

    # Compute metrics
    doc_type_counts = df['document_type'].value_counts(dropna=False).to_dict()
    jurisdiction_counts = df['document_jurisdiction'].value_counts(dropna=False).to_dict() if 'document_jurisdiction' in df else {}
    missing_counts = df.isnull().sum().to_dict()

    df['char_length'] = df['text'].astype(str).str.len()
    df['word_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))

    stats_char = df['char_length'].describe().to_dict()
    stats_word = df['word_length'].describe().to_dict()

    authority_counts = df['issuing_authority'].value_counts(dropna=False).head(10).to_dict() if 'issuing_authority' in df else {}

    exact_text_duplicates = int(df.duplicated(subset=['text']).sum())
    title_duplicates = int(df.duplicated(subset=['document_title']).sum())

    export_stats = {
        "total_documents": int(total_records),
        "document_type_distribution": {str(k): int(v) for k, v in doc_type_counts.items()},
        "jurisdiction_distribution": {str(k): int(v) for k, v in jurisdiction_counts.items()},
        "missing_values": {str(k): int(v) for k, v in missing_counts.items()},
        "character_length_stats": {str(k): float(v) for k, v in stats_char.items()},
        "word_length_stats": {str(k): float(v) for k, v in stats_word.items()},
        "top_issuing_authorities": {str(k): int(v) for k, v in authority_counts.items()},
        "exact_text_duplicates": exact_text_duplicates,
        "title_duplicates": title_duplicates
    }

    for path in [STATS_OUTPUT_FILE_1, STATS_OUTPUT_FILE_2]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_stats, f, indent=2)

    print(f"Dataset exploration successful. Saved stats for {total_records} documents to:")
    print(f" - {STATS_OUTPUT_FILE_1}")
    print(f" - {STATS_OUTPUT_FILE_2}")

if __name__ == "__main__":
    run_exploration()
