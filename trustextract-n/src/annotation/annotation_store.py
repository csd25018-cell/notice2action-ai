"""
Annotation Store for NoticeIE-Gold
Handles reading and writing JSONL annotation files with exact character offsets.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

ANNOTATION_LABELS = ["TITLE", "AUTHORITY", "AUDIENCE", "ELIGIBILITY", "DOCUMENT", "DATE", "CONTACT"]

class AnnotationStore:
    """
    Persistent JSONL store for annotation records.
    Each line is one document's full annotation object.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads all annotation records keyed by document_id.
        Returns empty dict if file doesn't exist yet.
        """
        records: Dict[str, Dict[str, Any]] = {}
        if not os.path.exists(self.output_path):
            return records

        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        records[obj["document_id"]] = obj
                    except json.JSONDecodeError:
                        continue
        return records

    def save_all(self, records: Dict[str, Dict[str, Any]]) -> None:
        """
        Writes all annotation records back to JSONL file.
        One document per line.
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            for doc_id, record in records.items():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_annotation(self, document_id: str, records: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return records.get(document_id)

    def add_entity(
        self,
        document_id: str,
        document_text: str,
        label: str,
        start: int,
        end: int,
        records: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Adds a new entity span annotation to a document's record.
        Creates the record if it doesn't exist.
        """
        span_text = document_text[start:end]

        if document_id not in records:
            records[document_id] = {
                "document_id": document_id,
                "text": document_text,
                "entities": [],
                "annotated_at": datetime.now().isoformat()
            }

        records[document_id]["entities"].append({
            "label": label,
            "start": start,
            "end": end,
            "text": span_text
        })
        records[document_id]["last_updated"] = datetime.now().isoformat()
        return records

    def delete_entity(
        self,
        document_id: str,
        entity_index: int,
        records: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Removes an entity by its index in the entities list.
        """
        if document_id in records:
            entities = records[document_id].get("entities", [])
            if 0 <= entity_index < len(entities):
                entities.pop(entity_index)
                records[document_id]["entities"] = entities
                records[document_id]["last_updated"] = datetime.now().isoformat()
        return records
