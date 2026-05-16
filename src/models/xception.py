"""
DeepGuard — Xception Detector
Custom Xception architecture for deepfake detection.
Based on: Rössler et al. "FaceForensics++: Learning to Detect Manipulated Facial Images"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector import DeepfakeDetector


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class XceptionBlock(nn.Module):
    """Xception middle flow block with residual connection."""

    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True):
        super().__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = None

        layers = []
        for i in range(reps):
            if start_with_relu or i > 0:
                layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3, padding=1
            ))
            layers.append(nn.BatchNorm2d(out_channels))

        if stride > 1:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        return self.block(x) + residual


class XceptionDetector(DeepfakeDetector):
    """
    Xception-based deepfake detector.
    Architecture adapted for binary classification (real vs fake).

    Input: (B, 3, 299, 299) — standard Xception input size
    """

    def build_model(self):
        # Entry flow
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False),
            XceptionBlock(128, 256, reps=2, stride=2),
            XceptionBlock(256, 728, reps=2, stride=2),
        )

        # Middle flow (8 repetitions)
        middle_blocks = [XceptionBlock(728, 728, reps=3) for _ in range(8)]
        self.middle_flow = nn.Sequential(*middle_blocks)

        # Exit flow
        self.exit_flow = nn.Sequential(
            XceptionBlock(728, 1024, reps=2, stride=2),
            SeparableConv2d(1024, 1536, kernel_size=3, padding=1),
            nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.backbone = nn.Sequential(self.entry_flow, self.middle_flow, self.exit_flow)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate / 2),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def build_xception_from_timm() -> nn.Module:
    """
    Alternative: load Xception from timm (if available).
    Falls back to custom implementation if timm doesn't have xception.
    """
    try:
        import timm
        model = timm.create_model("xception", pretrained=True, num_classes=0)
        return model
    except Exception:
        return None
