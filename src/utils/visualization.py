"""
DeepGuard — Visualization Utilities
"""
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Headless backend for web servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# --- Style defaults ---
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#0f3460",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "grid.color": "#0f3460",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

PALETTE = {
    "real": "#00b4d8",
    "fake": "#ef233c",
    "accent": "#7b2d8b",
    "success": "#2ecc71",
    "bg": "#1a1a2e",
}


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "ROC Curve — DeepGuard",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=PALETTE["fake"], lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["fake"])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.legend(loc="lower right", facecolor="#16213e")
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pr_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Precision-Recall Curve — DeepGuard",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 7),
) -> plt.Figure:
    """Plot Precision-Recall curve with Average Precision score."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    ap = average_precision_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color=PALETTE["accent"], lw=2,
            label=f"PR (AP = {ap:.4f})")
    
    # Baseline is the ratio of positive examples
    baseline = np.sum(labels) / len(labels) if len(labels) > 0 else 0.5
    ax.plot([0, 1], [baseline, baseline], "w--", lw=1, alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.fill_between(recall, precision, alpha=0.1, color=PALETTE["accent"])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.legend(loc="upper right", facecolor="#16213e")
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig



def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["REAL", "FAKE"],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 5),
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar=True,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_frame_probabilities(
    frame_probs: List[float],
    threshold: float = 0.5,
    title: str = "Per-Frame Deepfake Probability",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot per-frame probability timeline for video analysis."""
    frames = list(range(len(frame_probs)))

    fig, ax = plt.subplots(figsize=figsize)
    colors = [PALETTE["fake"] if p >= threshold else PALETTE["real"] for p in frame_probs]

    ax.bar(frames, frame_probs, color=colors, alpha=0.8, width=0.8)
    ax.axhline(y=threshold, color="white", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold}")
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlim(-0.5, len(frames) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frame Index", fontsize=12)
    ax.set_ylabel("P(FAKE)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.legend(facecolor="#16213e")
    ax.grid(True, axis="y")

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["fake"], label="FAKE frame"),
        Patch(facecolor=PALETTE["real"], label="REAL frame"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="#16213e")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    val_aucs: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training loss and validation AUC curves."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, color=PALETTE["real"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, color=PALETTE["fake"], label="Val Loss",
             linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend(facecolor="#16213e")
    ax1.grid(True)

    # AUC plot
    ax2.plot(epochs, val_aucs, color=PALETTE["accent"], linewidth=2)
    ax2.fill_between(epochs, val_aucs, alpha=0.2, color=PALETTE["accent"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation AUC")
    ax2.set_title("Validation AUC-ROC", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(True)

    plt.suptitle("DeepGuard Training Progress", fontsize=15, fontweight="bold", color="white")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
