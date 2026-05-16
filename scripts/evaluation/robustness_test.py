"""
DeepGuard — Robustness Testing Script
Tests the model against various image corruptions (JPEG compression, Resize, Noise)
and plots the AUC degradation curves.

Usage:
    python scripts/evaluation/robustness_test.py --checkpoint models/checkpoints/best_model.pth --test-csv data/metadata/test.csv
"""
import argparse
import sys
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import DeepfakeCSVDataset, create_dataloader
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model
from src.training.metrics import compute_metrics
from src.utils.config import load_config
from src.utils.logger import setup_logger

from albumentations.pytorch import ToTensorV2

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Robustness Testing")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-csv", type=str, default="data/metadata/test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="reports/evaluation/robustness")
    return parser.parse_args()


def get_corruption_transform(corruption_type: str, severity: int, image_size: int = 224):
    """
    Returns an Albumentations transform pipeline with a specific corruption.
    Severity scale varies by corruption type.
    """
    base_tf = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ]
    
    if corruption_type == "baseline":
        return A.Compose(base_tf)
        
    elif corruption_type == "jpeg":
        # Severity = JPEG Quality (lower is worse). Severities: 80, 60, 40, 20
        quality = severity
        return A.Compose([
            A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0)
        ] + base_tf)
        
    elif corruption_type == "downscale":
        # Severity = Resize dimension before upscaling (lower is worse). Severities: 112, 64, 32
        scale = severity
        return A.Compose([
            A.Resize(scale, scale, interpolation=cv2.INTER_AREA),
        ] + base_tf)
        
    elif corruption_type == "noise":
        # Severity = Gaussian Noise variance limit (higher is worse). Severities: 10, 50, 100
        var = float(severity)
        return A.Compose([
            A.GaussNoise(var_limit=(var, var), p=1.0)
        ] + base_tf)
        
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def run_evaluation(model, dataloader, device):
    """Run inference and return AUC."""
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    metrics = compute_metrics(np.array(all_labels), (np.array(all_probs) >= 0.5).astype(int), np.array(all_probs))
    return metrics.auc_roc


def plot_degradation_curves(results: dict, output_dir: Path):
    """Plot AUC degradation for each corruption type."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    # 1. JPEG
    if "jpeg" in results:
        df = pd.DataFrame(results["jpeg"]).sort_values("severity", ascending=False)
        axes[0].plot(df["severity"], df["auc"], marker='o', color='#00b4d8', linewidth=2)
        axes[0].set_title('JPEG Compression', fontweight='bold')
        axes[0].set_xlabel('JPEG Quality (Higher is better)')
        axes[0].set_ylabel('AUC-ROC')
        axes[0].invert_xaxis() # 100 -> 20

    # 2. Downscale
    if "downscale" in results:
        df = pd.DataFrame(results["downscale"]).sort_values("severity", ascending=False)
        axes[1].plot(df["severity"], df["auc"], marker='s', color='#2ecc71', linewidth=2)
        axes[1].set_title('Low Resolution', fontweight='bold')
        axes[1].set_xlabel('Resolution px (Higher is better)')
        axes[1].invert_xaxis() # 224 -> 32

    # 3. Noise
    if "noise" in results:
        df = pd.DataFrame(results["noise"]).sort_values("severity")
        axes[2].plot(df["severity"], df["auc"], marker='^', color='#ef233c', linewidth=2)
        axes[2].set_title('Gaussian Noise', fontweight='bold')
        axes[2].set_xlabel('Noise Variance (Lower is better)')

    plt.tight_layout()
    output_path = output_dir / "robustness_degradation.png"
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Robustness plots saved to {output_path}")

import cv2

def main():
    args = parse_args()
    setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    cfg = load_config()
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("config", {}).get("backbone", "efficientnet_b4")
    model = build_model(backbone=backbone)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    # Define experiments
    experiments = {
        "baseline": [None],
        "jpeg": [100, 80, 60, 40, 20],
        "downscale": [224, 112, 64, 32],
        "noise": [0, 10, 50, 100]
    }

    results = {}
    table = Table(title="Robustness Testing Results", show_header=True, header_style="bold cyan")
    table.add_column("Corruption", style="white")
    table.add_column("Severity Parameter", justify="center")
    table.add_column("AUC-ROC", justify="right", style="green")

    for corr_type, severities in experiments.items():
        results[corr_type] = []
        for sev in severities:
            # Setup dataloader with corrupted transform
            tf = get_corruption_transform(corr_type, sev, cfg.image_size)
            test_ds = DeepfakeCSVDataset(args.test_csv, transform=tf)
            test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
            
            # Evaluate
            console.print(f"Testing [bold yellow]{corr_type}[/bold yellow] (severity: {sev})...")
            auc_val = run_evaluation(model, test_loader, device)
            
            display_sev = str(sev) if sev is not None else "None"
            table.add_row(corr_type, display_sev, f"{auc_val:.4f}")
            
            results[corr_type].append({"severity": sev if sev is not None else (100 if corr_type=="jpeg" else (224 if corr_type=="downscale" else 0)), "auc": auc_val})

    console.print(table)
    
    # Save CSV
    flat_results = []
    for c_type, res_list in results.items():
        for r in res_list:
            flat_results.append({"Corruption": c_type, "Severity": r["severity"], "AUC": r["auc"]})
    
    df = pd.DataFrame(flat_results)
    df.to_csv(output_dir / "robustness_results.csv", index=False)
    
    # Plot
    plot_degradation_curves(results, output_dir)

if __name__ == "__main__":
    main()
