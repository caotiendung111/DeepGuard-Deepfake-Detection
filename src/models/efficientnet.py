"""
DeepGuard — EfficientNet-B4 Detector
Uses timm's EfficientNet as feature extractor with custom classification head.
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")

from .detector import DeepfakeDetector


class EfficientNetDetector(DeepfakeDetector):
    """
    EfficientNet-based deepfake detector using timm pretrained models.

    Supported model_name values:
        - efficientnet_b4 (default, recommended)
        - efficientnet_b7 (heavier, higher accuracy)
        - efficientnetv2_m
        - efficientnetv2_l
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        self.model_name = model_name
        super().__init__(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
        )

    def build_model(self):
        """Build EfficientNet backbone + classification head."""
        # Load pretrained EfficientNet from timm
        base_model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=0,       # Remove original head
            global_pool="avg",   # Global average pooling
        )

        # Get feature dimension
        in_features = base_model.num_features

        self.backbone = base_model

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class MultiScaleEfficientNet(EfficientNetDetector):
    """
    Enhanced version with multi-scale feature extraction.
    Extracts features from multiple intermediate layers for richer representation.
    """

    def build_model(self):
        super().build_model()
        # Additional frequency branch (DCT-based artifact detection)
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 64),
        )

        # Adjust classifier input dimension
        in_features = self.backbone.num_features + 64
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=self.dropout_rate / 2),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.backbone(x)
        freq_feat = self.freq_branch(x)
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(combined)
