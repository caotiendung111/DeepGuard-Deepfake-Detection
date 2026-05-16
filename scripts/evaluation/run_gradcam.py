"""
DeepGuard — Grad-CAM Analysis Script
Runs Captum LayerGradCam on specific images or a directory of images.

Usage:
    python scripts/evaluation/run_gradcam.py --checkpoint models/checkpoints/best_model.pth --image path/to/image.jpg
    python scripts/evaluation/run_gradcam.py --checkpoint models/checkpoints/best_model.pth --input-dir reports/evaluation/errors/false_positives
"""
import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from rich.progress import track

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.gradcam import GradCAMVisualizer
from src.models import build_model
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Grad-CAM Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing images to process")
    parser.add_argument("--output-dir", type=str, default="reports/evaluation/gradcam", help="Output directory")
    parser.add_argument("--backbone", type=str, default="efficientnet_b4", help="Model backbone if not in checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logger()

    if not args.image and not args.input_dir:
        logger.error("Must provide either --image or --input-dir")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── Load Model ────────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("config", {}).get("backbone", args.backbone)
    
    model = build_model(backbone=backbone)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    
    logger.info(f"Model loaded. Initializing Captum GradCAM Visualizer...")
    visualizer = GradCAMVisualizer(model=model, device=str(device))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Collect Images ────────────────────────────────────────────────────────
    images_to_process = []
    if args.image:
        images_to_process.append(Path(args.image))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            images_to_process.extend(list(input_dir.glob(f"*{ext}")))
            images_to_process.extend(list(input_dir.glob(f"*{ext.upper()}")))
            
    if not images_to_process:
        logger.warning("No images found to process.")
        return
        
    logger.info(f"Processing {len(images_to_process)} images...")
    
    for img_path in track(images_to_process, description="Generating Heatmaps..."):
        try:
            combined_img = visualizer.generate(str(img_path))
            out_path = output_dir / f"gradcam_{img_path.name}"
            visualizer.save(combined_img, str(out_path))
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")

    logger.success(f"Grad-CAM generation complete! Results saved in {output_dir}")

if __name__ == "__main__":
    main()
