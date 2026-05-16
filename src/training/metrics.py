"""
DeepGuard — Evaluation Metrics
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class MetricResult:
    """Container for all computed metrics."""
    accuracy: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    loss: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "loss": self.loss,
        }

    def __str__(self) -> str:
        return (
            f"Acc={self.accuracy:.4f} | AUC={self.auc_roc:.4f} | "
            f"F1={self.f1:.4f} | Prec={self.precision:.4f} | "
            f"Rec={self.recall:.4f} | Loss={self.loss:.4f}"
        )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    loss: float = 0.0,
) -> MetricResult:
    """
    Compute all classification metrics for deepfake detection.

    Args:
        labels: Ground truth binary labels (0=real, 1=fake), shape (N,)
        predictions: Predicted binary labels, shape (N,)
        probabilities: Predicted probabilities for fake class, shape (N,)
        threshold: Decision threshold for converting probs to labels.
        loss: Average epoch loss.

    Returns:
        MetricResult with all computed metrics.
    """
    labels = np.array(labels).astype(int)
    predictions = np.array(predictions).astype(int)
    probabilities = np.array(probabilities).astype(float)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc_roc = 0.0

    # AUC-PR
    try:
        auc_pr = average_precision_score(labels, probabilities)
    except ValueError:
        auc_pr = 0.0

    # Specificity (True Negative Rate)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return MetricResult(
        accuracy=accuracy_score(labels, predictions),
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        f1=f1_score(labels, predictions, zero_division=0),
        precision=precision_score(labels, predictions, zero_division=0),
        recall=recall_score(labels, predictions, zero_division=0),
        specificity=specificity,
        loss=loss,
        confusion_matrix=cm,
    )


class MetricTracker:
    """
    Accumulates predictions and labels across batches for epoch-level metrics.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            ...
            tracker.update(labels, preds, probs, loss)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._labels: List[int] = []
        self._predictions: List[int] = []
        self._probabilities: List[float] = []
        self._losses: List[float] = []

    def update(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        loss: float,
    ):
        self._labels.extend(labels.cpu().numpy().tolist())
        self._predictions.extend(predictions.cpu().numpy().tolist())
        self._probabilities.extend(probabilities.cpu().numpy().tolist())
        self._losses.append(loss)

    def compute(self, threshold: float = 0.5) -> MetricResult:
        avg_loss = float(np.mean(self._losses)) if self._losses else 0.0
        return compute_metrics(
            labels=np.array(self._labels),
            predictions=np.array(self._predictions),
            probabilities=np.array(self._probabilities),
            threshold=threshold,
            loss=avg_loss,
        )


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = ["Real", "Fake"]):
    """
    Plot confusion matrix using matplotlib and seaborn.
    
    Args:
        cm: Confusion matrix array of shape (2, 2)
        class_names: List of class names
        
    Returns:
        matplotlib Figure object
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 14, "weight": "bold"}, ax=ax
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    
    plt.tight_layout()
    return fig
