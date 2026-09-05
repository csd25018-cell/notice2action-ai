"""
calibration.py – Temperature Scaling for TrustExtract-N
======================================================
Implements post-hoc temperature scaling on validation logits to calibrate
model probabilities without modifying underlying weights or retraining.

Calculates Expected Calibration Error (ECE) and Brier Score, and generates
a reliability diagram to compare before/after calibration.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from src.training.dataset import TokenClassificationDataset

class TemperatureScaler(nn.Module):
    """
    A thin PyTorch module to learn the temperature scalar parameter T.
    """
    def __init__(self, initial_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Prevent division by zero or negative temp
        return logits / self.temperature.clamp(min=1e-3)


def get_all_logits_and_labels(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    ignore_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs inference to collect all logits and labels from a dataloader.
    Filters out ignored tokens (-100).
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (B, SeqLen, NumClasses)

            # Flatten to (B * SeqLen, NumClasses)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            # Filter ignored indices
            mask = labels != ignore_index
            logits_filtered = logits[mask]
            labels_filtered = labels[mask]

            all_logits.append(logits_filtered.cpu())
            all_labels.append(labels_filtered.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def optimize_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Optimizes the temperature T using LBFGS on the validation set.
    """
    scaler = TemperatureScaler()
    # LBFGS is well-suited for this 1-parameter optimization
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def eval_fn():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return scaler.temperature.item()


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class classification.
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = 0.0
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0.0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def compute_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes multi-class Brier score.
    Brier Score = 1/N * sum_{i} sum_{c} (p_{ic} - y_{ic})^2
    """
    num_classes = probs.size(-1)
    # One-hot encode labels
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    brier_score = torch.mean(torch.sum((probs - labels_one_hot) ** 2, dim=1))
    return brier_score.item()


def plot_reliability_diagram(
    uncal_probs: torch.Tensor, 
    cal_probs: torch.Tensor, 
    labels: torch.Tensor, 
    output_path: str,
    n_bins: int = 10
):
    """
    Plots a reliability diagram comparing uncalibrated vs calibrated probabilities.
    """
    def _get_bin_stats(probs, labels):
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accs = []
        bin_confs = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.any():
                bin_accs.append(accuracies[in_bin].float().mean().item())
                bin_confs.append(confidences[in_bin].mean().item())
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
        return bin_confs, bin_accs

    # Compute stats
    uncal_confs, uncal_accs = _get_bin_stats(uncal_probs, labels)
    cal_confs, cal_accs = _get_bin_stats(cal_probs, labels)

    plt.figure(figsize=(10, 5))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    # Uncalibrated plot
    plt.plot(uncal_confs, uncal_accs, marker='o', label='Uncalibrated', color='red', alpha=0.7)
    
    # Calibrated plot
    plt.plot(cal_confs, cal_accs, marker='s', label='Temperature Scaled', color='blue', alpha=0.7)

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram (NoticeIE-Gold Calibration)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save diagram
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved reliability diagram to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate TrustExtract-N using Temperature Scaling")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ignore_index", type=int, default=-100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Using device: {device}")

    # 1. Load Model & Dataset
    print(f"  Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)

    print(f"  Loading validation dataset from {args.val_data}...")
    val_dataset = TokenClassificationDataset(args.val_data)
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=args.ignore_index)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collator)

    # 2. Extract Logits
    print("  Running inference on validation set...")
    val_logits, val_labels = get_all_logits_and_labels(model, val_loader, device, args.ignore_index)
    
    if val_logits.numel() == 0:
        print("  Error: No valid tokens found for calibration.")
        sys.exit(1)

    # 3. Before Calibration Metrics
    uncal_probs = F.softmax(val_logits, dim=-1)
    uncal_loss = F.cross_entropy(val_logits, val_labels).item()
    uncal_ece = compute_ece(uncal_probs, val_labels)
    uncal_brier = compute_brier_score(uncal_probs, val_labels)

    print("\n── BEFORE CALIBRATION ─────────────────────")
    print(f"  Cross Entropy Loss : {uncal_loss:.4f}")
    print(f"  ECE                : {uncal_ece:.4f}")
    print(f"  Brier Score        : {uncal_brier:.4f}")

    # 4. Optimize Temperature
    print("\n  Optimizing temperature...")
    optimal_T = optimize_temperature(val_logits, val_labels)
    print(f"  Learned Temperature (T): {optimal_T:.4f}")

    # 5. After Calibration Metrics
    cal_logits = val_logits / optimal_T
    cal_probs = F.softmax(cal_logits, dim=-1)
    cal_loss = F.cross_entropy(cal_logits, val_labels).item()
    cal_ece = compute_ece(cal_probs, val_labels)
    cal_brier = compute_brier_score(cal_probs, val_labels)

    print("\n── AFTER CALIBRATION (Temperature Scaled) ──")
    print(f"  Cross Entropy Loss : {cal_loss:.4f}")
    print(f"  ECE                : {cal_ece:.4f}")
    print(f"  Brier Score        : {cal_brier:.4f}")

    # 6. Save Config
    out_dir = Path("models/trustextract")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "calibration.json"
    
    config = {
        "temperature": optimal_T,
        "metrics_before": {
            "loss": uncal_loss,
            "ece": uncal_ece,
            "brier_score": uncal_brier
        },
        "metrics_after": {
            "loss": cal_loss,
            "ece": cal_ece,
            "brier_score": cal_brier
        }
    }
    
    with open(out_file, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\n  Saved calibration config to: {out_file}")

    # 7. Reliability Diagram
    diagram_path = out_dir / "calibration_reliability.png"
    plot_reliability_diagram(uncal_probs, cal_probs, val_labels, str(diagram_path))
    
    print("\n  Calibration complete.\n")


if __name__ == "__main__":
    main()
