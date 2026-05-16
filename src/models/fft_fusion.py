import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

from .detector import DeepfakeDetector
from .efficientnet import EfficientNetDetector

class FFTFusionDetector(DeepfakeDetector):
    """
    Experiment C: Frequency Domain Feature Fusion.
    Fuses spatial features (EfficientNet) with frequency features (FFT).
    """
    def __init__(self, model_name: str = "efficientnet_b4", **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)
        
    def build_model(self):
        # 1. Spatial Branch (EfficientNet)
        self.spatial_net = EfficientNetDetector(model_name=self.model_name, pretrained=self.pretrained)
        # Remove its classifier so we just get features
        spatial_feature_dim = self.spatial_net.classifier[1].in_features
        self.spatial_net.classifier = nn.Identity()
        
        # 2. Frequency Branch (Simple CNN for FFT magnitude)
        # FFT magnitude will have 3 channels (RGB)
        self.freq_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        freq_feature_dim = 128
        
        # 3. Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(spatial_feature_dim + freq_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
        
        self.backbone = nn.ModuleList([self.spatial_net.backbone, self.freq_net])

    def get_fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute shifted 2D FFT magnitude of the input image."""
        # x is (B, C, H, W)
        fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        # Add small epsilon to prevent log(0)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
        return magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial features
        spatial_features = self.spatial_net(x) # (B, spatial_feature_dim)
        
        # Frequency features
        fft_mag = self.get_fft_magnitude(x)
        freq_features = self.freq_net(fft_mag) # (B, freq_feature_dim)
        
        # Fusion
        fused = torch.cat([spatial_features, freq_features], dim=1)
        return self.classifier(fused)
