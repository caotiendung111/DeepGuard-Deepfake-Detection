"""
DeepGuard — Grad-CAM Heatmap Visualization
Highlights which facial regions influence the model's decision using Captum.
Outputs a 3-panel image: Original | Heatmap | Overlay
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger

try:
    from captum.attr import LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logger.warning("captum not installed. Run: pip install captum")

from ..data.transforms import get_val_transforms


class GradCAMVisualizer:
    """
    Generates LayerGradCam heatmaps using Captum for model explainability.
    Outputs a combined 3-panel image (Original | Heatmap | Overlay).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: str = "auto",
        image_size: int = 224,
    ):
        if not CAPTUM_AVAILABLE:
            raise ImportError("Install captum: pip install captum")

        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device
        )
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        self.transform = get_val_transforms(image_size)

        # Auto-detect target layer if not specified
        self.target_layer = target_layer or self._auto_detect_layer()
        self.layer_gc = LayerGradCam(self.model, self.target_layer)

    def _auto_detect_layer(self) -> torch.nn.Module:
        """Try to automatically detect the last conv layer for Captum."""
        # For EfficientNet (timm)
        for name in ["backbone.conv_head", "backbone.blocks"]:
            parts = name.split(".")
            m = self.model
            try:
                for p in parts:
                    m = getattr(m, p)
                if isinstance(m, torch.nn.Sequential):
                    return m[-1]
                return m
            except AttributeError:
                continue

        # Fallback: find last Conv2d
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv:
            return last_conv

        raise ValueError("Could not auto-detect target layer. Please specify manually.")

    def _preprocess(self, image_input) -> Tuple[torch.Tensor, np.ndarray]:
        """Returns (tensor, normalized_rgb_float_array)."""
        if isinstance(image_input, (str, Path)):
            img_rgb = np.array(Image.open(image_input).convert("RGB"))
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img_rgb = image_input if image_input.shape[-1] == 3 else \
                      cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        # Resize for display
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size))
        img_float = img_resized.astype(np.float32) / 255.0

        # Transform for model
        transformed = self.transform(image=img_rgb)
        tensor = transformed["image"].unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        return tensor, img_float

    def generate(
        self,
        image_input,
        target: int = 1,   # Default to explaining the "Fake" class (though typically binary Output is shape (B,1))
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Generate 3-panel visualization: Original | Heatmap | Overlay

        Args:
            image_input: File path, PIL Image, or numpy array.
            target: Target class index (0 for BCE/binary classification usually).
            colormap: OpenCV colormap for heatmap.
            alpha: Blend factor for overlay.

        Returns:
            RGB numpy array with 3 panels horizontally concatenated (uint8, H x 3*W x 3).
        """
        tensor, img_float = self._preprocess(image_input)

        # For binary classification with output shape (B, 1), target is 0 for the single output node
        # Captum expects target=0 for the 0th output dimension
        attribution = self.layer_gc.attribute(tensor, target=0, relu_attributions=True)
        
        # Upsample attribution to match image size
        upsampled_attr = LayerGradCam.interpolate(attribution, (self.image_size, self.image_size))
        
        # Normalize to 0-1
        attr_np = upsampled_attr.squeeze().cpu().detach().numpy()
        if attr_np.max() > attr_np.min():
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
        else:
            attr_np = np.zeros_like(attr_np)

        # 1. Original Image (uint8)
        original = (img_float * 255).astype(np.uint8)
        
        # 2. Heatmap Image (uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_np), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Convert back to RGB
        
        # 3. Overlay Image (uint8)
        overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
        
        # Concatenate horizontally
        combined = np.concatenate((original, heatmap, overlay), axis=1)
        
        # Add labels on top
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "Captum LayerGradCam", (self.image_size + 10, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "Overlay", (self.image_size*2 + 10, 20), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return combined

    def save(self, combined_image: np.ndarray, output_path: str, quality: int = 95):
        """Save combined image to disk."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        logger.info(f"Grad-CAM saved to {output_path}")
