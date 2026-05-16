# src/models/__init__.py
from .detector import DeepfakeDetector
from .efficientnet import EfficientNetDetector
from .xception import XceptionDetector
from .fft_fusion import FFTFusionDetector


def build_model(backbone: str = "efficientnet_b4", **kwargs) -> DeepfakeDetector:
    """
    Factory function to build a detector model by backbone name.

    Args:
        backbone: One of "efficientnet_b4", "efficientnet_b7", "xception", "fft_b4"
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Instantiated DeepfakeDetector subclass.
    """
    backbone = backbone.lower()
    if backbone == "fft_b4":
        return FFTFusionDetector(model_name="efficientnet_b4", **kwargs)
    elif backbone.startswith("efficientnet") or backbone.startswith("vit"):
        return EfficientNetDetector(model_name=backbone, **kwargs)
    elif backbone == "xception":
        return XceptionDetector(**kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from: efficientnet_b4, xception, vit_base_patch16_224, fft_b4")


__all__ = ["DeepfakeDetector", "EfficientNetDetector", "XceptionDetector", "FFTFusionDetector", "build_model"]
