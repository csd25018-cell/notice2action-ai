"""
evaluate.py – Evaluation Metrics for TrustExtract-N Token Classification
========================================================================
Computes entity-level precision, recall, and F1 score using `seqeval`.
Filters out special tokens (label_id == -100).
"""

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from typing import Dict, Tuple, List, Callable


def get_compute_metrics_fn(id2label: Dict[str, str], ignore_label_id: int = -100) -> Callable:
    """
    Returns a compute_metrics function configured with the given label map.
    This factory pattern allows passing the function directly to HF Trainer.
    """
    def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_preds
        
        # In case the model returns a tuple instead of just logits
        if isinstance(logits, tuple):
            logits = logits[0]
            
        predictions = np.argmax(logits, axis=2)

        # Remove ignored index (special tokens)
        true_labels: List[List[str]] = []
        true_predictions: List[List[str]] = []

        for i in range(labels.shape[0]):
            true_label_seq = []
            true_pred_seq = []
            for j in range(labels.shape[1]):
                if labels[i, j] != ignore_label_id:
                    # Map IDs back to string labels for seqeval
                    true_label_seq.append(id2label[str(labels[i, j])])
                    true_pred_seq.append(id2label[str(predictions[i, j])])
                    
            true_labels.append(true_label_seq)
            true_predictions.append(true_pred_seq)

        # Calculate metrics
        metrics = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
        
        # Include detailed report string if needed for logging
        # We don't return it as a dict key because Trainer expects floats,
        # but you can print it or log it to wandb/tensorboard here.
        # print("\n" + classification_report(true_labels, true_predictions))

        return metrics

    return compute_metrics
