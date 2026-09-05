"""
abstention.py – Risk-Aware Selective Inference for TrustExtract-N
================================================================
Implements risk-aware abstention logic based on calibrated confidence thresholds.
Tunes thresholds on the validation set using a precision-coverage trade-off
rather than arbitrary cutoffs.
"""

from typing import Dict, Any, List, Optional
import numpy as np

def apply_abstention(
    extraction: Dict[str, Any], 
    threshold: float
) -> Dict[str, Any]:
    """
    Applies abstention logic to a single extraction.
    
    Args:
        extraction: Dict containing 'label', 'text', 'confidence', 
                    'evidence_text', and 'page_number'.
        threshold: The calibrated threshold for this entity type.
        
    Returns:
        A dict with the final accepted/abstained result.
    """
    conf = extraction.get("confidence", 0.0)
    label = extraction.get("label", "UNKNOWN")
    
    if conf >= threshold:
        return {
            "status": "HIGH_CONFIDENCE",
            "label": label,
            "value": extraction.get("text"),
            "evidence": extraction.get("evidence_text"),
            "page_number": extraction.get("page_number"),
            "confidence": conf
        }
    else:
        return {
            "status": "LOW_CONFIDENCE",
            "label": label,
            "value": None,
            "evidence": None,
            "page_number": None,
            "confidence": conf
        }


def evaluate_selective_metrics(
    predictions: List[Dict[str, Any]], 
    threshold: float
) -> Dict[str, float]:
    """
    Evaluates selective inference metrics for a set of predictions at a given threshold.
    
    Args:
        predictions: List of dicts with:
            - 'confidence': float
            - 'is_correct': bool
            - 'is_missing_in_ground_truth': bool (True if the document actually lacked this field)
        threshold: The confidence cutoff.
        
    Returns:
        Dict of metrics: coverage, selective_precision, abstention_rate, false_answer_rate.
    """
    total = len(predictions)
    if total == 0:
        return {
            "coverage": 0.0, 
            "selective_precision": 0.0, 
            "abstention_rate": 0.0, 
            "false_answer_rate_missing": 0.0
        }
        
    accepted = 0
    correct_and_accepted = 0
    missing_in_gt = 0
    accepted_but_missing_in_gt = 0
    
    for p in predictions:
        conf = p.get("confidence", 0.0)
        is_missing = p.get("is_missing_in_ground_truth", False)
        
        if is_missing:
            missing_in_gt += 1
            
        if conf >= threshold:
            accepted += 1
            if p.get("is_correct", False):
                correct_and_accepted += 1
            if is_missing:
                accepted_but_missing_in_gt += 1

    coverage = accepted / total
    abstention_rate = 1.0 - coverage
    selective_precision = (correct_and_accepted / accepted) if accepted > 0 else 1.0
    far_missing = (accepted_but_missing_in_gt / missing_in_gt) if missing_in_gt > 0 else 0.0

    return {
        "coverage": coverage,
        "selective_precision": selective_precision,
        "abstention_rate": abstention_rate,
        "false_answer_rate_missing": far_missing
    }


def tune_threshold(
    val_predictions: List[Dict[str, Any]], 
    target_precision: float = 0.95
) -> float:
    """
    Finds the optimal threshold on the validation set that meets the target 
    precision while maximizing coverage.
    
    Args:
        val_predictions: List of predictions (same format as evaluate_selective_metrics).
        target_precision: The minimum selective precision we want to maintain.
        
    Returns:
        float: The chosen threshold.
    """
    if not val_predictions:
        return 0.99  # Safe fallback
        
    best_threshold = 0.99
    best_coverage = -1.0
    
    # Grid search over possible thresholds
    thresholds = np.linspace(0.0, 1.0, 101)
    
    for t in thresholds:
        metrics = evaluate_selective_metrics(val_predictions, t)
        
        # We want to meet or exceed target precision
        if metrics["selective_precision"] >= target_precision:
            # And we want the highest coverage possible among valid thresholds
            if metrics["coverage"] > best_coverage:
                best_coverage = metrics["coverage"]
                best_threshold = t
                
    # If no threshold meets the target precision, return 0.99 as a highly conservative fallback
    if best_coverage == -1.0:
        return 0.99
        
    return float(best_threshold)
