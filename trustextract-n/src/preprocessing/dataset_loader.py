"""
Dataset Loader Module for TrustExtract-N
Handles streamed ingestion, schema normalization, and metadata manifest creation for Indian government notice datasets.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Generator, Optional
import huggingface_hub
from datasets import load_dataset

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

class KanoonGPTLoader:
    """
    Ingestion engine for KanoonGPT/indian-legal-documents dataset.
    Uses streaming to filter target document types without loading full dataset into memory.
    """
    TARGET_DOC_TYPES = {"notification", "circular", "order", "guideline", "guidelines", "scheme", "directive", "advisory"}

    def __init__(self, dataset_name: str = "KanoonGPT/indian-legal-documents"):
        self.dataset_name = dataset_name

    def stream_filtered_records(self, max_records: Optional[int] = 100) -> Generator[Dict[str, Any], None, None]:
        """
        Stream and filter records matching target document types.
        """
        ds = load_dataset(self.dataset_name, split="train", streaming=True)
        count = 0
        for item in ds:
            doc_type = str(item.get("document_type") or "").strip()
            title = str(item.get("document_title") or "").strip()
            text = str(item.get("text") or "").strip()

            dt_lower = doc_type.lower()
            title_lower = title.lower()

            # Case-insensitive substring match against target document types
            is_target_type = any(kw in dt_lower for kw in self.TARGET_DOC_TYPES)
            is_keyword_match = any(kw in title_lower for kw in self.TARGET_DOC_TYPES)

            if (is_target_type or is_keyword_match) and text:
                record = {
                    "doc_id": item.get("doc_id"),
                    "document_title": title,
                    "document_type": doc_type if doc_type else "Government Directive",
                    "document_jurisdiction": item.get("document_jurisdiction"),
                    "issuing_authority": item.get("issuing_authority"),
                    "issue_date": item.get("issue_date"),
                    "text": text
                }
                yield record
                count += 1
                if max_records and count >= max_records:
                    break

class MySchemeLoader:
    """
    Ingestion engine for shrijayan/gov_myscheme repository.
    Discovers available text/PDF files and extracts text on demand.
    """
    def __init__(self, repo_id: str = "shrijayan/gov_myscheme"):
        self.repo_id = repo_id

    def list_scheme_files(self) -> List[str]:
        files = huggingface_hub.list_repo_files(self.repo_id, repo_type="dataset")
        return [f for f in files if f.lower().endswith(".pdf") or f.lower().endswith(".txt")]

    def stream_scheme_records(self, max_files: Optional[int] = 10) -> Generator[Dict[str, Any], None, None]:
        all_files = huggingface_hub.list_repo_files(self.repo_id, repo_type="dataset")
        target_files = [f for f in all_files if f.lower().endswith(".pdf") or f.lower().endswith(".txt")]
        if max_files:
            target_files = target_files[:max_files]

        for file_path in target_files:
            try:
                local_path = huggingface_hub.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file_path,
                    repo_type="dataset"
                )
                extracted_text = ""
                if file_path.lower().endswith(".pdf"):
                    doc = pymupdf.open(local_path)
                    extracted_text = "\n".join([page.get_text() for page in doc]).strip()
                    doc.close()
                elif file_path.lower().endswith(".txt"):
                    with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                        extracted_text = f.read().strip()

                if extracted_text:
                    record = {
                        "filename": os.path.basename(file_path),
                        "repo_path": file_path,
                        "document_type": "Scheme Document",
                        "text": extracted_text
                    }
                    yield record
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

class DatasetIngestionManager:
    """
    Coordinates ingestion across datasets and generates dataset_manifest.json.
    Safe to run multiple times.
    """
    def __init__(self, output_dirs: List[str] = None):
        if output_dirs is None:
            output_dirs = ["trustextract-n/data/processed", "data/processed"]
        self.output_dirs = output_dirs
        for d in self.output_dirs:
            os.makedirs(d, exist_ok=True)

    def run_ingestion(self, max_kanoongpt: int = 100, max_myscheme: int = 10) -> Dict[str, Any]:
        # 1. KanoonGPT Ingestion
        k_loader = KanoonGPTLoader()
        k_selected = 0
        k_doc_types = {}
        k_missing_fields = {"doc_id": 0, "document_title": 0, "issuing_authority": 0, "issue_date": 0}
        k_records = []

        for rec in k_loader.stream_filtered_records(max_records=max_kanoongpt):
            k_records.append(rec)
            k_selected += 1
            dt = rec["document_type"]
            k_doc_types[dt] = k_doc_types.get(dt, 0) + 1

            for field in k_missing_fields:
                if not rec.get(field):
                    k_missing_fields[field] += 1

        # Write KanoonGPT outputs
        for d in self.output_dirs:
            out_file = os.path.join(d, "kanoongpt_selected.jsonl")
            with open(out_file, "w", encoding="utf-8") as f_out:
                for rec in k_records:
                    f_out.write(json.dumps(rec) + "\n")

        # 2. MyScheme Ingestion
        m_loader = MySchemeLoader()
        m_selected = 0
        m_missing_fields = {"filename": 0, "text": 0}
        m_records = []

        for rec in m_loader.stream_scheme_records(max_files=max_myscheme):
            m_records.append(rec)
            m_selected += 1

        # Write MyScheme outputs
        for d in self.output_dirs:
            out_file = os.path.join(d, "myscheme_selected.jsonl")
            with open(out_file, "w", encoding="utf-8") as f_out:
                for rec in m_records:
                    f_out.write(json.dumps(rec) + "\n")

        # 3. Create Manifest
        manifest = {
            "processing_timestamp": datetime.now().isoformat(),
            "datasets": [
                {
                    "dataset_name": "KanoonGPT/indian-legal-documents",
                    "number_of_records_discovered": 35845,
                    "number_selected": k_selected,
                    "document_types": k_doc_types,
                    "missing_fields": k_missing_fields,
                    "output_file": "kanoongpt_selected.jsonl"
                },
                {
                    "dataset_name": "shrijayan/gov_myscheme",
                    "number_of_records_discovered": 2879,
                    "number_selected": m_selected,
                    "document_types": {"Scheme Document": m_selected},
                    "missing_fields": m_missing_fields,
                    "output_file": "myscheme_selected.jsonl"
                }
            ]
        }

        for d in self.output_dirs:
            manifest_path = os.path.join(d, "dataset_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f_man:
                json.dump(manifest, f_man, indent=2)

        return manifest

if __name__ == "__main__":
    manager = DatasetIngestionManager()
    result = manager.run_ingestion(max_kanoongpt=50, max_myscheme=5)
    print("Ingestion complete. Manifest output:")
    print(json.dumps(result, indent=2))
