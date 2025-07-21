"""
Evaluation metrics for license plate recognition
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def character_accuracy(predictions, targets):
    """Calculate character-level accuracy"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    correct = 0
    total = 0

    for pred, target in zip(predictions, targets):
        total += len(target)
        correct += np.sum(pred == target)

    return correct / total if total > 0 else 0.0


def sequence_accuracy(predictions, targets):
    """Calculate sequence-level accuracy (all characters must be correct)"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    correct = 0
    total = len(targets)

    for pred, target in zip(predictions, targets):
        if np.array_equal(pred, target):
            correct += 1

    return correct / total if total > 0 else 0.0


def calculate_metrics(predictions, targets):
    """Calculate comprehensive metrics"""
    char_acc = character_accuracy(predictions, targets)
    seq_acc = sequence_accuracy(predictions, targets)

    # Flatten for sklearn metrics
    pred_flat = (
        predictions.flatten()
        if hasattr(predictions, "flatten")
        else np.array(predictions).flatten()
    )
    target_flat = (
        targets.flatten()
        if hasattr(targets, "flatten")
        else np.array(targets).flatten()
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average="weighted", zero_division=0
    )

    return {
        "character_accuracy": char_acc,
        "sequence_accuracy": seq_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


class MetricsTracker:
    """Track metrics during training"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, preds, targets, loss=None):
        """Update metrics with batch results"""
        if isinstance(preds, torch.Tensor):
            preds = torch.argmax(preds, dim=-1)
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        self.predictions.extend(preds)
        self.targets.extend(targets)

        if loss is not None:
            self.losses.append(loss)

    def compute(self):
        """Compute final metrics"""
        if not self.predictions:
            return {}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = calculate_metrics(predictions, targets)

        if self.losses:
            metrics["avg_loss"] = np.mean(self.losses)

        return metrics

    def get_summary(self):
        """Get formatted summary string"""
        metrics = self.compute()
        if not metrics:
            return "No metrics available"

        summary = []
        if "avg_loss" in metrics:
            summary.append(f"Loss: {metrics['avg_loss']:.4f}")
        summary.append(f"Char Acc: {metrics['character_accuracy']:.3f}")
        summary.append(f"Seq Acc: {metrics['sequence_accuracy']:.3f}")
        summary.append(f"F1: {metrics['f1_score']:.3f}")

        return " | ".join(summary)
