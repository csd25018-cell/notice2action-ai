"""
Dataset Inspector Module for TrustExtract-N
Inspects schemas, repository structures, and feature distributions without loading full datasets into memory.
"""

import sys
import json
from typing import Dict, Any, List

try:
    import huggingface_hub
    from datasets import load_dataset_builder
except ImportError as e:
    print(f"Error: Missing required dependency ({e}).")
    print("Please activate your Python virtual environment first:\n  source .venv/bin/activate")
    sys.exit(1)

class DatasetInspector:
    """
    Utility class for inspecting Hugging Face datasets metadata and structure.
    """

    @staticmethod
    def inspect_kanoongpt() -> Dict[str, Any]:
        """
        Inspects KanoonGPT/indian-legal-documents metadata.
        """
        try:
            builder = load_dataset_builder("KanoonGPT/indian-legal-documents")
            features = list(builder.info.features.keys()) if builder.info.features else []
            splits = {k: v.num_examples for k, v in builder.info.splits.items()} if builder.info.splits else {}
            return {
                "dataset": "KanoonGPT/indian-legal-documents",
                "status": "available",
                "features": features,
                "splits": splits,
                "total_examples": sum(splits.values())
            }
        except Exception as e:
            return {
                "dataset": "KanoonGPT/indian-legal-documents",
                "status": "error",
                "error": str(e)
            }

    @staticmethod
    def inspect_myscheme() -> Dict[str, Any]:
        """
        Inspects shrijayan/gov_myscheme repository file listing.
        """
        try:
            files = huggingface_hub.list_repo_files("shrijayan/gov_myscheme", repo_type="dataset")
            pdf_files = [f for f in files if f.lower().endswith(".pdf")]
            text_files = [f for f in files if f.lower().endswith(".txt")]
            return {
                "dataset": "shrijayan/gov_myscheme",
                "status": "available",
                "total_files": len(files),
                "pdf_count": len(pdf_files),
                "txt_count": len(text_files),
                "sample_files": pdf_files[:10]
            }
        except Exception as e:
            return {
                "dataset": "shrijayan/gov_myscheme",
                "status": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    print("--- KanoonGPT Inspection ---")
    print(json.dumps(DatasetInspector.inspect_kanoongpt(), indent=2))
    print("\n--- MyScheme Inspection ---")
    print(json.dumps(DatasetInspector.inspect_myscheme(), indent=2))
