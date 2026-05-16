"""
DeepGuard — Augmentation Transforms
YAML-configurable pipelines using Albumentations.

Supports:
- get_train_transforms() — aggressive augmentation for training
- get_val_transforms()   — minimal (resize + normalize)
- get_tta_transforms()   — test-time augmentation list
- build_transforms_from_config() — build from Config dataclass or YAML dict
"""
from typing import Dict, List, Optional, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── ImageNet normalization constants ───────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════════
# Training transforms
# ═══════════════════════════════════════════════════════════════════════════════
def get_train_transforms(
    image_size: int = 224,
    jpeg_quality_range: tuple = (60, 100),
    coarse_dropout_holes: int = 8,
    coarse_dropout_size: int = 32,
    augment_level: str = "medium",  # "light" | "medium" | "heavy"
) -> A.Compose:
    """
    Aggressive augmentation pipeline optimized for deepfake detection.

    Key augmentations:
    - JpegCompression: simulates real-world compression artifacts that
      are one of the strongest cues for deepfake detection
    - GaussNoise: models sensor noise in real cameras
    - CoarseDropout: simulates occlusion (glasses, masks, hair)
    - RandomResizedCrop: forces model to detect manipulation from partial faces

    Args:
        image_size: Output size (square).
        jpeg_quality_range: (lower, upper) JPEG quality for compression augment.
        coarse_dropout_holes: Max number of dropout rectangles.
        coarse_dropout_size: Max size of each dropout rectangle.
        augment_level: "light" | "medium" | "heavy" — controls probability/strength.
    """
    # Scale probabilities by augment level
    p_scale = {"light": 0.5, "medium": 1.0, "heavy": 1.5}
    s = p_scale.get(augment_level, 1.0)

    def p(base: float) -> float:
        return min(base * s, 1.0)

    return A.Compose([
        # ── Geometry ──────────────────────────────────────────────────────────
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=p(0.3),
        ),
        A.Resize(height=image_size, width=image_size),  # Fallback if crop not applied
        A.HorizontalFlip(p=p(0.5)),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.12,
            rotate_limit=12,
            border_mode=0,
            p=p(0.4),
        ),

        # ── Color & Lighting ─────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=p(0.5),
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=p(0.3),
        ),
        A.OneOf([
            A.CLAHE(clip_limit=4.0),
            A.Equalize(),
        ], p=p(0.2)),

        # ── Noise & Blur (simulate camera/compression artifacts) ─────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
            A.Sharpen(alpha=(0.2, 0.5)),
        ], p=p(0.4)),  # Increased from 0.25

        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
        ], p=p(0.35)), # Increased from 0.3

        # ── JPEG Compression Artifacts (critical for deepfake detection) ──
        # Apply frequently to both classes so model doesn't use it as a shortcut
        A.ImageCompression(
            quality_range=(jpeg_quality_range[0], jpeg_quality_range[1]),
            p=p(0.6),  # Increased from 0.5
        ),

        A.RandomGamma(gamma_limit=(80, 120), p=p(0.2)),

        # ── Coarse Dropout (simulate occlusion) ─────────────────────────────
        A.CoarseDropout(
            num_holes_range=(1, coarse_dropout_holes),
            hole_height_range=(8, coarse_dropout_size),
            hole_width_range=(8, coarse_dropout_size),
            fill_value=0,
            p=p(0.3),
        ),

        # ── Normalize & to Tensor ────────────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Validation / Test transforms
# ═══════════════════════════════════════════════════════════════════════════════
def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Minimal transforms: resize + normalize. No augmentation."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Test-Time Augmentation
# ═══════════════════════════════════════════════════════════════════════════════
def get_tta_transforms(image_size: int = 224) -> List[A.Compose]:
    """
    Returns a list of transforms for TTA.
    Predictions are averaged over all variants.
    """
    return [
        # Original
        get_val_transforms(image_size),
        # Horizontal flip
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        # Slight center-preserving crop. Keep TTA deterministic; random
        # transforms at inference make repeated predictions unstable.
        A.Compose([
            A.SmallestMaxSize(max_size=int(image_size * 1.08)),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        # Slight compression
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.ImageCompression(quality_range=(75, 95), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        # --- NEW: Slight Blur (to filter high-frequency noise that confuses models) ---
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
        # --- NEW: Slight Sharpening ---
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Sharpen(alpha=(0.2, 0.3), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization-safe transforms (no normalize/tensor — returns uint8 numpy)
# ═══════════════════════════════════════════════════════════════════════════════
def get_train_transforms_visual(image_size: int = 224) -> A.Compose:
    """
    Same augmentations as training, but WITHOUT Normalize and ToTensorV2.
    Output is uint8 RGB numpy array — suitable for matplotlib display.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3,
        ),
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=12, border_mode=0, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
        ], p=0.25),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1)),
            A.ISONoise(),
        ], p=0.3),
        A.ImageCompression(quality_range=(60, 100), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill_value=0, p=0.3),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Config-driven builder
# ═══════════════════════════════════════════════════════════════════════════════
def build_transforms_from_config(config) -> dict:
    """
    Build train/val/test transforms from a Config dataclass or dict.

    Returns:
        {"train": A.Compose, "val": A.Compose, "test": A.Compose, "tta": list}
    """
    if hasattr(config, "image_size"):
        image_size = config.image_size
        aug_level = getattr(config, "augment_level", "medium")
        jpeg_lower = getattr(config, "jpeg_quality_lower", 60)
        jpeg_upper = getattr(config, "jpeg_quality_upper", 100)
    elif isinstance(config, dict):
        image_size = config.get("image_size", 224)
        aug_level = config.get("augment_level", "medium")
        jpeg_lower = config.get("jpeg_quality_lower", 60)
        jpeg_upper = config.get("jpeg_quality_upper", 100)
    else:
        image_size = 224
        aug_level = "medium"
        jpeg_lower, jpeg_upper = 60, 100

    return {
        "train": get_train_transforms(
            image_size=image_size,
            jpeg_quality_range=(jpeg_lower, jpeg_upper),
            augment_level=aug_level,
        ),
        "val": get_val_transforms(image_size),
        "test": get_val_transforms(image_size),
        "tta": get_tta_transforms(image_size),
        "visual": get_train_transforms_visual(image_size),
    }
