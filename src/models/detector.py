"""
DeepGuard — Abstract Base Detector
All backbone models should inherit from this class.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple


class DeepfakeDetector(nn.Module, ABC):
    """
    Abstract base class for all DeepGuard detection models.

    Subclasses must implement `build_model()`.
    """

    def __init__(
        self,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.pretrained = pretrained

        # To be populated by subclasses
        self.backbone: Optional[nn.Module] = None
        self.classifier: Optional[nn.Module] = None

        self.build_model()

    @abstractmethod
    def build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits of shape (B, 1) for binary classification.
        """
        features = self.backbone(x)
        return self.classifier(features)

    def get_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability (0=real, 1=fake)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class label and probability.

        Returns:
            (predictions, probabilities): both shape (B,)
        """
        probs = self.get_probability(x).squeeze(1)
        preds = (probs >= threshold).long()
        return preds, probs

    def get_num_params(self) -> Dict[str, int]:
        """Return parameter count statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def freeze_backbone(self):
        """Freeze all backbone parameters (only train classifier)."""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None):
        """Save model checkpoint with metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu", **model_kwargs):
        """Load model from checkpoint file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
