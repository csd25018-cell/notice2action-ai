import numpy as np
from typing import List, Dict, Any, Tuple
import json
import os

class ConformalPredictor:
    """
    Implements Split Conformal Prediction for risk-aware abstention.
    Instead of outputting a single prediction, it outputs a *prediction set* 
    with a mathematical guarantee of marginal coverage.
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha (float): The target error rate. E.g., alpha=0.05 targets 95% coverage.
        """
        self.alpha = alpha
        self.q_hat = None

    def calibrate(self, validation_probabilities: np.ndarray, true_labels: np.ndarray):
        """
        Calibrates the conformal predictor using a validation set.
        
        Args:
            validation_probabilities: shape (N, C) containing calibrated probabilities.
            true_labels: shape (N,) containing the true class indices.
        """
        n = validation_probabilities.shape[0]
        
        # Calculate non-conformity scores: 1 - P(y_true)
        # We want the probability the model assigned to the *true* class
        true_class_probs = validation_probabilities[np.arange(n), true_labels]
        non_conformity_scores = 1.0 - true_class_probs
        
        # Compute the quantile q_hat
        val = np.ceil((n + 1) * (1 - self.alpha)) / n
        val = min(max(val, 0.0), 1.0) # bound between 0 and 1
        
        self.q_hat = np.quantile(non_conformity_scores, val)
        return self.q_hat

    def predict_set(self, probabilities: np.ndarray) -> List[int]:
        """
        Generates a prediction set for a new example.
        
        Args:
            probabilities: shape (C,) array of class probabilities for a single token/span.
            
        Returns:
            List of class indices that fall within the prediction set.
        """
        if self.q_hat is None:
            # Fallback to thresholding if not calibrated
            return [np.argmax(probabilities)] if np.max(probabilities) >= 0.5 else []
            
        # We include all classes where 1 - P(y) <= q_hat
        # which is equivalent to P(y) >= 1 - q_hat
        threshold = 1.0 - self.q_hat
        
        prediction_set = np.where(probabilities >= threshold)[0].tolist()
        return prediction_set

    def save_calibration(self, filepath: str):
        """Saves the conformal quantile to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {"q_hat": float(self.q_hat) if self.q_hat is not None else None, "alpha": self.alpha}
        with open(filepath, "w") as f:
            json.dump(data, f)
            
    def load_calibration(self, filepath: str):
        """Loads the conformal quantile from disk."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                self.q_hat = data.get("q_hat")
                self.alpha = data.get("alpha", 0.05)
