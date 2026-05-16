"""
DeepGuard — Model Benchmark Script
Evaluates multiple models on the same test set, measures latency and metrics, and plots comparisons.

Usage:
    python scripts/evaluation/benchmark_models.py \
        --checkpoints models/effnet.pth models/xception.pth \
        --names EfficientNet-B4 Xception \
        --test-csv data/metadata/test.csv
"""
import argparse
import sys
import time
from pathlib import Path

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
from src.data.transforms import get_val_transforms
from src.models import build_model
from src.training.metrics import compute_metrics
from src.utils.config import load_config
from src.utils.logger import setup_logger

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Model Benchmark")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths")
    parser.add_argument("--names", nargs="+", required=True, help="List of model names for plotting")
    parser.add_argument("--test-csv", type=str, default="data/metadata/test.csv", help="Test dataset CSV")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default="reports/evaluation/benchmark")
    return parser.parse_args()

def plot_benchmark_results(results_df: pd.DataFrame, output_dir: Path):
    """Plot bar charts comparing AUC, F1, and Inference Time."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    
    models = results_df['Model']
    x = np.arange(len(models))
    width = 0.5

    # 1. AUC
    axes[0].bar(x, results_df['AUC-ROC'], width, color='#00b4d8')
    axes[0].set_title('AUC-ROC Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_ylim(0.5, 1.05)
    for i, v in enumerate(results_df['AUC-ROC']):
        axes[0].text(i, v + 0.01, f"{v:.4f}", color='white', ha='center')

    # 2. F1 Score
    axes[1].bar(x, results_df['F1 Score'], width, color='#2ecc71')
    axes[1].set_title('F1 Score Comparison', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_ylim(0.0, 1.05)
    for i, v in enumerate(results_df['F1 Score']):
        axes[1].text(i, v + 0.01, f"{v:.4f}", color='white', ha='center')

    # 3. Inference Time
    axes[2].bar(x, results_df['Latency (ms)'], width, color='#ef233c')
    axes[2].set_title('Inference Time per Image (ms)', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15)
    for i, v in enumerate(results_df['Latency (ms)']):
        axes[2].text(i, v + 0.5, f"{v:.1f}", color='white', ha='center')

    plt.tight_layout()
    output_path = output_dir / "benchmark_comparison.png"
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Benchmark plot saved to {output_path}")


def main():
    args = parse_args()
    setup_logger()

    if len(args.checkpoints) != len(args.names):
        logger.error("Number of checkpoints must match number of names")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # We assume image size is 224 for all, or read from base config
    cfg = load_config()
    test_ds = DeepfakeCSVDataset(args.test_csv, transform=get_val_transforms(cfg.image_size))
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    results = []

    for name, ckpt_path in zip(args.names, args.checkpoints):
        console.print(f"\n[bold cyan]Evaluating Model: {name}[/bold cyan]")
        
        if not Path(ckpt_path).exists():
            logger.error(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        checkpoint = torch.load(ckpt_path, map_location=device)
        backbone = checkpoint.get("config", {}).get("backbone", "efficientnet_b4")
        
        model = build_model(backbone=backbone)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        
        params = model.get_num_params()['total']
        
        # Warmup for accurate latency
        dummy_input = torch.randn(1, 3, cfg.image_size, cfg.image_size).to(device)
        with torch.no_grad():
            for _ in range(5):
                model(dummy_input)
                
        all_labels, all_probs = [], []
        total_time = 0.0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc=f"Inference ({name})"):
                images = images.to(device)
                
                # Measure latency
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                logits = model(images)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                
                total_time += (t1 - t0)
                
                probs = torch.sigmoid(logits).squeeze(1)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                
        # Calculate Metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Simple threshold=0.5 for direct comparison
        preds = (all_probs >= 0.5).astype(int)
        metrics = compute_metrics(all_labels, preds, all_probs)
        
        latency_ms = (total_time / len(test_ds)) * 1000
        
        results.append({
            "Model": name,
            "Params (M)": round(params / 1e6, 2),
            "AUC-ROC": metrics.auc_roc,
            "F1 Score": metrics.f1,
            "Accuracy": metrics.accuracy,
            "Latency (ms)": latency_ms
        })
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        
    if not results:
        return

    # Print Table
    df = pd.DataFrame(results)
    table = Table(title="DeepGuard Benchmark Results", show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(col, justify="center")
        
    for _, row in df.iterrows():
        table.add_row(
            row['Model'], 
            f"{row['Params (M)']}",
            f"{row['AUC-ROC']:.4f}",
            f"{row['F1 Score']:.4f}",
            f"{row['Accuracy']:.4f}",
            f"{row['Latency (ms)']:.2f}"
        )
    console.print(table)
    
    # Save CSV and Plot
    df.to_csv(output_dir / "benchmark_results.csv", index=False)
    plot_benchmark_results(df, output_dir)
    logger.success(f"Benchmark complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
