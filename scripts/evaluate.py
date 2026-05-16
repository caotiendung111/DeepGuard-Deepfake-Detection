"""
DeepGuard — Model Evaluation Script

- Tính metrics (AUC, F1, PR, etc.)
- Tìm optimal threshold (Youden's J)
- Vẽ ROC, PR, Confusion Matrix
- Xuất PDF report

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth --test-csv data/metadata/test.csv
"""
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track
from sklearn.metrics import roc_curve
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import DeepfakeCSVDataset, create_dataloader
from src.data.transforms import get_val_transforms
from src.training.metrics import compute_metrics
from src.inference.model_loader import load_detector_checkpoint
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve
)

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--test-csv", type=str, default="data/metadata/test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="reports/evaluation")
    parser.add_argument("--threshold-output", type=str, default="configs/thresholds.yaml")
    return parser.parse_args()


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J statistic (TPR - FPR)."""
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present; falling back to threshold=0.5")
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def generate_pdf_report(
    metrics, 
    optimal_threshold: float,
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    output_path: str,
    model_name: str
):
    """Generate a comprehensive PDF report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title and Metrics
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        
        plt.text(0.5, 0.95, "DeepGuard — Evaluation Report", ha='center', va='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.90, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', va='center', fontsize=12)
        plt.text(0.5, 0.85, f"Model: {model_name}", ha='center', va='center', fontsize=14)
        
        metrics_text = (
            f"Evaluation Metrics (Threshold = {optimal_threshold:.4f})\n"
            f"{'-'*50}\n"
            f"AUC-ROC     : {metrics.auc_roc:.4f}\n"
            f"AUC-PR      : {metrics.auc_pr:.4f}\n"
            f"F1 Score    : {metrics.f1:.4f}\n"
            f"Accuracy    : {metrics.accuracy:.4f}\n"
            f"Precision   : {metrics.precision:.4f}\n"
            f"Recall      : {metrics.recall:.4f}\n"
            f"Specificity : {metrics.specificity:.4f}\n"
        )
        
        plt.text(0.1, 0.65, metrics_text, ha='left', va='top', fontsize=12, family='monospace',
                 bbox=dict(facecolor='#f0f0f0', edgecolor='gray', boxstyle='round,pad=1'))
        
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2: ROC and PR Curves
        fig_roc = plot_roc_curve(all_labels, all_probs)
        fig_roc.patch.set_facecolor('white')
        for ax in fig_roc.axes: ax.set_facecolor('white'); ax.tick_params(colors='black'); ax.xaxis.label.set_color('black'); ax.yaxis.label.set_color('black'); ax.title.set_color('black')
        pdf.savefig(fig_roc)
        plt.close(fig_roc)
        
        fig_pr = plot_pr_curve(all_labels, all_probs)
        fig_pr.patch.set_facecolor('white')
        for ax in fig_pr.axes: ax.set_facecolor('white'); ax.tick_params(colors='black'); ax.xaxis.label.set_color('black'); ax.yaxis.label.set_color('black'); ax.title.set_color('black')
        pdf.savefig(fig_pr)
        plt.close(fig_pr)
        
        # Page 3: Confusion Matrix
        fig_cm = plot_confusion_matrix(metrics.confusion_matrix, normalize=False)
        fig_cm.patch.set_facecolor('white')
        pdf.savefig(fig_cm)
        plt.close(fig_cm)


def main():
    import os
    args = parse_args()
    setup_logger()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, model_metadata = load_detector_checkpoint(args.checkpoint, cfg, device)
    backbone = model_metadata["backbone"]

    params = model.get_num_params()
    console.print(f"[bold cyan]Model:[/bold cyan] {backbone} | [bold cyan]Params:[/bold cyan] {params['total']:,}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    if not Path(args.test_csv).exists():
        logger.error(f"Test CSV not found: {args.test_csv}")
        sys.exit(1)

    test_ds = DeepfakeCSVDataset(
        args.test_csv,
        transform=get_val_transforms(cfg.image_size)
    )
    test_loader = create_dataloader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    console.print(f"[bold cyan]Test set:[/bold cyan] {len(test_ds)} samples")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_labels, all_probs, all_filepaths = [], [], []

    with torch.no_grad():
        for images, labels, filepaths in track(test_loader, description="Evaluating..."):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_filepaths.extend(filepaths)
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── Threshold Optimization ────────────────────────────────────────────────
    opt_threshold = find_optimal_threshold(all_labels, all_probs)
    all_preds = (all_probs >= opt_threshold).astype(int)

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(
        labels=all_labels,
        predictions=all_preds,
        probabilities=all_probs,
        threshold=opt_threshold,
    )

    # ── Print Results ─────────────────────────────────────────────────────────
    table = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Score", justify="right", style="green")
    
    table.add_row("Optimal Threshold", f"{opt_threshold:.4f}")
    table.add_row("AUC-ROC", f"{metrics.auc_roc:.4f}")
    table.add_row("AUC-PR", f"{metrics.auc_pr:.4f}")
    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("F1 Score", f"{metrics.f1:.4f}")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("Specificity", f"{metrics.specificity:.4f}")
    
    console.print(table)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV of predictions for Error Analysis
    df_preds = pd.DataFrame({
        "filepath": all_filepaths,
        "label": all_labels,
        "probability": all_probs,
        "prediction": all_preds
    })
    df_preds.to_csv(output_dir / "predictions.csv", index=False)
    
    # Generate PDF Report
    pdf_path = output_dir / "evaluation_report.pdf"
    generate_pdf_report(metrics, opt_threshold, all_labels, all_probs, str(pdf_path), backbone)

    threshold_output = Path(args.threshold_output)
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_output, "w") as f:
        yaml.safe_dump({
            "default_threshold": float(opt_threshold),
            "high_recall_threshold": float(opt_threshold),
            "source": str(args.test_csv),
            "model": backbone,
            "n_samples": int(len(all_labels)),
            "auc_roc": float(metrics.auc_roc),
            "f1": float(metrics.f1),
            "generated_by": "scripts/evaluate.py",
        }, f, sort_keys=False)
    
    logger.success(f"Evaluation complete! PDF Report saved to: {pdf_path}")
    logger.info(f"Predictions saved to: {output_dir / 'predictions.csv'}")
    logger.info(f"Threshold config saved to: {threshold_output}")

    # --- Kaggle Output Export ---
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"):
        logger.info("Kaggle environment detected. Exporting reports to /kaggle/working/...")
        try:
            kaggle_root = Path("/kaggle/working")
            
            # Copy PDF report to root
            if pdf_path.exists():
                dest_pdf = kaggle_root / "evaluation_report.pdf" if kaggle_root.exists() else Path("evaluation_report.pdf")
                shutil.copy(pdf_path, dest_pdf)
            
            # Copy predictions CSV to root
            preds_csv = output_dir / "predictions.csv"
            if preds_csv.exists():
                dest_csv = kaggle_root / "predictions.csv" if kaggle_root.exists() else Path("predictions.csv")
                shutil.copy(preds_csv, dest_csv)
                
            logger.success(f"Exported reports to {kaggle_root if kaggle_root.exists() else 'project root'} for easy access.")
        except Exception as e:
            logger.error(f"Failed to export Kaggle results: {e}")

if __name__ == "__main__":
    main()
