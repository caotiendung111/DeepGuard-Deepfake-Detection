"""
DeepGuard — Custom Loss Functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Addresses class imbalance by down-weighting easy examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Weighting factor for the positive class (0-1). Default: 0.25
        gamma: Focusing parameter. Higher = more focus on hard examples. Default: 2.0
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output, shape (B,) or (B, 1)
            targets: Binary labels, shape (B,) with values {0, 1}
        """
        logits = logits.squeeze(1)
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with per-class weights.
    Useful when dataset has imbalanced real/fake ratio.

    Args:
        pos_weight: Weight for the positive (fake) class.
                    Set > 1 if fake samples are underrepresented.
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1)
        targets = targets.float()
        weight = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


class LabelSmoothingBCE(nn.Module):
    """
    BCE with label smoothing to reduce overconfidence.

    Args:
        smoothing: Label smoothing factor (0.0 to 0.2). Default: 0.1
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1)
        targets = targets.float()
        # Smooth labels: 0 → smoothing/2, 1 → 1 - smoothing/2
        targets = targets * (1 - self.smoothing) + self.smoothing / 2
        return F.binary_cross_entropy_with_logits(logits, targets)


def build_loss(loss_type: str = "focal", **kwargs) -> nn.Module:
    """Factory function for loss functions."""
    loss_map = {
        "bce": nn.BCEWithLogitsLoss,
        "weighted_bce": WeightedBCELoss,
        "focal": FocalLoss,
        "label_smoothing": LabelSmoothingBCE,
    }
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss: {loss_type}. Choose from {list(loss_map.keys())}")
    return loss_map[loss_type](**kwargs)
