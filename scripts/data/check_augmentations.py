"""
DeepGuard — Augmentation Visualization Check
Hiển thị 16 ảnh sau augmentation (4x4 grid) để kiểm tra trực quan.

Usage:
    python scripts/data/check_augmentations.py --data-dir data/faces --output outputs/aug_check.png
    python scripts/data/check_augmentations.py --data-dir data/faces --n 32 --cols 8
    python scripts/data/check_augmentations.py --config configs/efficientnet_b4.yaml
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import DeepfakeFolderDataset, LABEL_NAMES
from src.data.transforms import get_train_transforms_visual, get_train_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.utils.config import load_config

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")


# ── Utility ────────────────────────────────────────────────────────────────────
def denormalize(tensor_or_np, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization for display."""
    if hasattr(tensor_or_np, "numpy"):
        img = tensor_or_np.numpy()
    else:
        img = tensor_or_np

    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # CHW → HWC

    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_augmentations(
    data_dir: str,
    output_path: str = "outputs/aug_check.png",
    n_images: int = 16,
    n_cols: int = 4,
    image_size: int = 224,
    use_normalized: bool = False,
    config_path: str = None,
    seed: int = 42,
):
    """
    Generate a grid of augmented images for visual inspection.

    Args:
        data_dir: Root directory with real/ and fake/ subdirs.
        output_path: Path to save the grid image.
        n_images: Number of images to display.
        n_cols: Number of columns in the grid.
        image_size: Image size for transforms.
        use_normalized: If True, show the actual normalized tensor (denormalized for display).
        config_path: Optional YAML config for transform params.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load config if provided
    if config_path:
        cfg = load_config(config_path)
        image_size = cfg.image_size
    else:
        cfg = None

    # Build transforms
    if use_normalized:
        # Shows what the model actually sees (after normalization)
        transform = get_train_transforms(image_size)
    else:
        # Shows human-readable augmented images
        transform = get_train_transforms_visual(image_size)

    # Load dataset without transforms (we'll apply manually)
    dataset = DeepfakeFolderDataset(data_dir, transform=None)

    if len(dataset) == 0:
        logger.error(f"No images found in {data_dir}")
        return

    n_images = min(n_images, len(dataset))
    n_rows = (n_images + n_cols - 1) // n_cols

    # Randomly sample indices
    indices = random.sample(range(len(dataset)), n_images)

    # ── Create figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 4.0))
    fig.patch.set_facecolor("#0f0c29")
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.15)

    logger.info(f"Generating {n_images} augmented images ({n_rows}×{n_cols} grid)...")

    for i, idx in enumerate(indices):
        _, label, filepath = dataset[idx]

        # Read raw image
        img = cv2.imread(filepath)
        if img is None:
            logger.warning(f"Cannot read: {filepath}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmented = transform(image=img_rgb)

        if use_normalized:
            # augmented["image"] is a tensor → denormalize
            display_img = denormalize(augmented["image"])
        else:
            # augmented["image"] is a numpy array (uint8)
            display_img = augmented["image"]

        # Get label info
        label_name = LABEL_NAMES.get(label, "?")
        short_path = Path(filepath).name

        # Draw border color based on label
        border_color = "#ef233c" if label == 1 else "#10b981"

        ax = fig.add_subplot(gs[i])
        ax.imshow(display_img)
        ax.set_title(
            f"{label_name.upper()} ({label})\n{short_path}",
            fontsize=8,
            color="white",
            fontweight="bold",
            pad=4,
        )
        ax.axis("off")

        # Draw colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    # ── Title ──────────────────────────────────────────────────────────────────
    title = "DeepGuard — Training Augmentation Preview"
    if use_normalized:
        title += " (Denormalized from model input)"
    fig.suptitle(title, fontsize=16, fontweight="bold", color="white", y=0.98)

    # ── Legend ─────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#10b981", label="REAL (0)"),
        Patch(facecolor="#ef233c", label="FAKE (1)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        fontsize=10,
        facecolor="#1a1a2e",
        edgecolor="#333",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.01),
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.success(f"Augmentation grid saved: {output_path}")

    return str(output_path)


def visualize_same_image_augmented(
    image_path: str,
    output_path: str = "outputs/aug_same_image.png",
    n_versions: int = 16,
    n_cols: int = 4,
    image_size: int = 224,
):
    """
    Apply augmentation N times to the SAME image to show the variety.
    Useful for checking augmentation diversity.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot read: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_train_transforms_visual(image_size)

    n_rows = (n_versions + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.patch.set_facecolor("#0f0c29")

    for i, ax in enumerate(axes.flat):
        if i < n_versions:
            aug = transform(image=img_rgb)["image"]
            ax.imshow(aug)
            ax.set_title(f"Variant {i+1}", fontsize=8, color="white")
        ax.axis("off")

    fig.suptitle(
        f"Same Image × {n_versions} Augmentations\n{Path(image_path).name}",
        fontsize=14, fontweight="bold", color="white",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.success(f"Same-image augmentation grid saved: {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Augmentation Checker")
    parser.add_argument("--data-dir", type=str, default="data/faces",
                        help="Root directory with real/ and fake/ subdirs")
    parser.add_argument("--output", type=str, default="outputs/aug_check.png",
                        help="Output image path")
    parser.add_argument("--n", type=int, default=16, help="Number of images to display")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--normalized", action="store_true",
                        help="Show denormalized model-input (after Normalize)")
    parser.add_argument("--seed", type=int, default=42)

    # Same-image mode
    parser.add_argument("--same-image", type=str, default=None,
                        help="Apply multiple augmentations to one image")
    parser.add_argument("--variants", type=int, default=16,
                        help="Number of augmentation variants for --same-image")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.same_image:
        visualize_same_image_augmented(
            image_path=args.same_image,
            output_path=args.output.replace("aug_check", "aug_same_image"),
            n_versions=args.variants,
            n_cols=args.cols,
            image_size=args.size,
        )
    else:
        visualize_augmentations(
            data_dir=args.data_dir,
            output_path=args.output,
            n_images=args.n,
            n_cols=args.cols,
            image_size=args.size,
            use_normalized=args.normalized,
            config_path=args.config,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
