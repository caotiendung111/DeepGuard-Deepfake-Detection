"""
DeepGuard — Model Diagnostics Tool
Runs inference on external images and generates Grad-CAM heatmaps to analyze failure modes.
Use this to understand why the model fails on real-world images.

Usage:
    1. Put images in data/external_test/ (create if doesn't exist)
    2. Run: python scripts/diagnose_model.py --data-dir data/external_test
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import build_model
from src.inference.predictor import ImagePredictor
from src.inference.gradcam import GradCAMVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Model Diagnostics")
    parser.add_argument("--data-dir", type=str, default="data/external_test",
                        help="Directory containing images to diagnose")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default="efficientnet_b4",
                        help="Model backbone architecture")
    parser.add_argument("--output-dir", type=str, default="reports/diagnostics",
                        help="Directory to save heatmaps and report")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on (cuda/cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info(f"Please create {data_dir} and put your test images there.")
        return

    # 1. Load Model
    logger.info(f"Loading model: {args.backbone} from {args.checkpoint}...")
    try:
        # We need to build the model first, then load weights because 
        # DeepfakeDetector.load_checkpoint expects the class, but we use build_model factory
        device = "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
        model = build_model(backbone=args.backbone, pretrained=False)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        
        # Fix prefix mismatch (backbone. prefix)
        new_state_dict = {}
        model_keys = model.state_dict().keys()
        
        for k, v in state_dict.items():
            if k in model_keys:
                new_state_dict[k] = v
            elif f"backbone.{k}" in model_keys:
                new_state_dict[f"backbone.{k}"] = v
            elif k.startswith("backbone.") and k[9:] in model_keys:
                new_state_dict[k[9:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        
        model.to(device).eval()
        logger.success("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Initialize Predictor and Grad-CAM
    predictor = ImagePredictor(model, device=device, threshold=args.threshold)
    visualizer = GradCAMVisualizer(model, device=device)

    # 3. Process Images
    image_paths = sorted([
        p for p in data_dir.rglob("*") 
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])
    
    if not image_paths:
        logger.warning(f"No images found in {data_dir}")
        return

    logger.info(f"Found {len(image_paths)} images. Starting diagnostics...")
    
    results = []
    for img_path in tqdm(image_paths, desc="Diagnosing"):
        try:
            # Predict
            res = predictor.predict(img_path)
            
            # Generate Heatmap
            # Target is 0 for the single output node in binary classification
            combined_viz = visualizer.generate(img_path, target=0)
            
            # Save Heatmap
            save_name = f"{res.label}_{res.probability:.2f}_{img_path.name}"
            save_path = heatmap_dir / save_name
            visualizer.save(combined_viz, str(save_path))
            
            results.append({
                "filename": img_path.name,
                "prediction": res.label,
                "prob_fake": res.probability,
                "confidence": res.confidence,
                "heatmap": save_name
            })
            
        except Exception as e:
            logger.warning(f"Failed to process {img_path.name}: {e}")

    # 4. Save Report
    df = pd.DataFrame(results)
    report_path = output_dir / "diagnostic_report.csv"
    df.to_csv(report_path, index=False)
    
    logger.success(f"Diagnostics complete!")
    logger.info(f"Heatmaps saved to: {heatmap_dir}")
    logger.info(f"Report saved to: {report_path}")
    
    # Simple summary
    fake_count = (df["prediction"] == "FAKE").sum()
    real_count = (df["prediction"] == "REAL").sum()
    logger.info(f"Summary: {fake_count} FAKE, {real_count} REAL")

if __name__ == "__main__":
    main()
