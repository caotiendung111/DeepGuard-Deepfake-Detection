"""
DeepGuard — Training Script CLI

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --backbone xception --epochs 20 --lr 5e-5
"""
import argparse
import random
import sys
from pathlib import Path

import shutil
import numpy as np
import torch
from loguru import logger
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_datasets, create_dataloader
from src.data.transforms import build_transforms_from_config
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import setup_logger

console = Console()

def set_seed(seed: int = 42):
    """Ensure reproducibility by fixing random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Training")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Override backbone (efficientnet_b4 | xception | vit_base_patch16_224)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--run-name", type=str, default=None,
                        help="MLflow run name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone, only train classifier head")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for dataset (overrides config)")
    return parser.parse_args()


def print_system_info(cfg, train_len, val_len):
    """Print a beautiful summary of the training setup."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    info_text = f"""[bold cyan]Hardware:[/bold cyan] {device_name}
[bold cyan]Backbone:[/bold cyan] {cfg.backbone} (Pretrained: {cfg.pretrained})
[bold cyan]Training:[/bold cyan] {cfg.num_epochs} Epochs | Batch Size: {cfg.batch_size} | LR: {cfg.learning_rate}
[bold cyan]Augmentation:[/bold cyan] Level: {cfg.augment_level} | Weighted Sampler: {cfg.use_weighted_sampler}
[bold cyan]Data:[/bold cyan] Train: {train_len:,} samples | Val: {val_len:,} samples"""

    console.print(Panel(info_text, title="[bold green]DeepGuard Training Setup[/bold green]", expand=False))


def main():
    args = parse_args()
    setup_logger()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.backbone: cfg.backbone = args.backbone
    if args.epochs: cfg.num_epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.learning_rate = args.lr
    if args.data_root: cfg.data_root = args.data_root

    set_seed(cfg.seed)
    logger.info(f"Seed set to {cfg.seed}")

    # Build transforms
    transforms = build_transforms_from_config(cfg)

    # Build datasets (CSV-based or Folder-based automatically handled)
    try:
        train_ds, val_ds, test_ds = build_datasets(
            config=cfg,
            train_transform=transforms["train"],
            val_transform=transforms["val"]
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    print_system_info(cfg, len(train_ds), len(val_ds))

    # Build dataloaders
    train_loader = create_dataloader(
        train_ds, 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_weighted_sampler=cfg.use_weighted_sampler
    )
    val_loader = create_dataloader(
        val_ds, 
        batch_size=cfg.batch_size,
        shuffle=False, 
        num_workers=cfg.num_workers
    )

    # Build model
    logger.info(f"Building model: {cfg.backbone}")
    model = build_model(
        backbone=cfg.backbone,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
        pretrained=cfg.pretrained,
    )

    # Optionally freeze backbone
    if args.freeze_backbone:
        model.freeze_backbone()
        logger.info("Backbone frozen — training classifier head only")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=cfg.to_dict(),
        experiment_name=cfg.mlflow_experiment_name,
    )

    # Train!
    best_auc = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        resume_from=args.resume,
    )

    logger.success(f"Training session complete! Best Val AUC: {best_auc:.4f}")

    # --- Kaggle Output Export ---
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"):
        logger.info("Kaggle environment detected. Exporting results to /kaggle/working/...")
        try:
            # Kaggle output root
            kaggle_root = Path("/kaggle/working")
            best_model_src = Path(args.checkpoint_dir) / "best_model.pth"
            
            if best_model_src.exists():
                # Copy to absolute root for visibility
                dest = kaggle_root / "best_model.pth" if kaggle_root.exists() else Path("best_model.pth")
                shutil.copy(best_model_src, dest)
                logger.success(f"Exported best_model.pth to {dest}")
            
            # Also copy config
            if Path(args.config).exists():
                dest_cfg = kaggle_root / "final_config.yaml" if kaggle_root.exists() else Path("final_config.yaml")
                shutil.copy(args.config, dest_cfg)
        except Exception as e:
            logger.error(f"Failed to export Kaggle results: {e}")

    return best_auc


if __name__ == "__main__":
    main()
