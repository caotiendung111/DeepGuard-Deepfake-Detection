"""
DeepGuard — Training Loop with MLflow Integration and Rich Progress Bar
"""
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ..models.detector import DeepfakeDetector
from .losses import build_loss
from .metrics import MetricResult, MetricTracker, plot_confusion_matrix


class Trainer:
    """
    Full training orchestrator for DeepGuard models.

    Features:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Cosine annealing with Warmup
    - MLflow experiment tracking
    - Best model checkpointing & Resuming
    - Early stopping
    - Rich progress bars
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        config: dict,
        device: Optional[str] = None,
        experiment_name: str = "deepguard-experiments",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.experiment_name = experiment_name

        # Training state
        self.current_epoch = 0
        self.best_auc = 0.0
        self.patience_counter = 0

        # Build optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Build loss
        loss_type = config.get("loss_type", "focal")
        self.criterion = build_loss(
            loss_type=loss_type,
            alpha=config.get("focal_alpha", 0.25),
            gamma=config.get("focal_gamma", 2.0),
        )

        # Grad scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        # Metric trackers
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()

        logger.info(
            f"Trainer initialized | Device: {self.device} | "
            f"Model params: {model.get_num_params()['trainable']:,} trainable"
        )

    def _train_epoch(self, loader: DataLoader, progress: Progress, task_id) -> MetricResult:
        """Run one training epoch."""
        self.model.train()
        self.train_tracker.reset()

        for batch_idx, (images, labels, _) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("grad_clip", 1.0)
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits).squeeze(1)
                preds = (probs >= 0.5).long()
                self.train_tracker.update(labels, preds, probs, loss.item())

            progress.update(task_id, advance=1, description=f"[cyan]Train Epoch {self.current_epoch}[/cyan] (Loss: {loss.item():.4f})")

        return self.train_tracker.compute()

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, progress: Progress, task_id) -> MetricResult:
        """Run one validation epoch."""
        self.model.eval()
        self.val_tracker.reset()

        for images, labels, _ in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= 0.5).long()
            self.val_tracker.update(labels, preds, probs, loss.item())

            progress.update(task_id, advance=1)

        return self.val_tracker.compute()

    def _save_checkpoint(self, path: str, metrics: MetricResult, scheduler):
        """Save complete model state for resuming."""
        checkpoint = {
            "epoch": self.current_epoch,
            "best_auc": self.best_auc,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config,
            "metrics": metrics.to_dict(),
        }
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path} (AUC={metrics.auc_roc:.4f})")

    def load_checkpoint(self, path: str) -> Optional[dict]:
        """Resume from a checkpoint."""
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found: {path}")
            return None
            
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.current_epoch = checkpoint["epoch"]
        self.best_auc = checkpoint.get("best_auc", 0.0)
        
        return checkpoint

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        checkpoint_dir: str = "models/checkpoints",
        run_name: Optional[str] = None,
        resume_from: Optional[str] = None,
    ):
        """Full training loop with MLflow tracking."""
        num_epochs = num_epochs or self.config.get("num_epochs", 30)
        warmup_epochs = self.config.get("warmup_epochs", 5)
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        patience = self.config.get("early_stopping_patience", 7)

        # LR scheduler: Linear Warmup -> Cosine Annealing
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=self.config.get("min_lr", 1e-6),
        )
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # Resume if specified
        if resume_from:
            ckpt = self.load_checkpoint(resume_from)
            if ckpt and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

        # Setup MLflow
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=run_name or f"run_{int(time.time())}") as run:
            mlflow.log_params(self.config)
            # mlflow.log_param("device", str(self.device)) # Removed to avoid conflict
            mlflow.log_param("model_class", self.model.__class__.__name__)

            logger.info(f"MLflow Run ID: {run.info.run_id}")
            logger.info(f"Training for {num_epochs} epochs on {self.device}")

            best_model_path = str(checkpoint_dir / "best_model.pth")
            start_epoch = self.current_epoch + 1

            for epoch in range(start_epoch, num_epochs + 1):
                self.current_epoch = epoch
                t0 = time.time()

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                ) as progress:
                    train_task = progress.add_task(f"[cyan]Train Epoch {epoch}[/cyan]", total=len(train_loader))
                    train_metrics = self._train_epoch(train_loader, progress, train_task)

                    val_task = progress.add_task(f"[green]Val   Epoch {epoch}[/green]", total=len(val_loader))
                    val_metrics = self._val_epoch(val_loader, progress, val_task)

                scheduler.step()
                elapsed = time.time() - t0

                # Log metrics to MLflow
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.to_dict().items()}, step=epoch)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.to_dict().items()}, step=epoch)
                mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)

                # Plot and log confusion matrix
                if val_metrics.confusion_matrix is not None:
                    fig = plot_confusion_matrix(val_metrics.confusion_matrix)
                    mlflow.log_figure(fig, f"confusion_matrices/cm_epoch_{epoch:03d}.png")
                    import matplotlib.pyplot as plt
                    plt.close(fig)

                logger.info(
                    f"Epoch {epoch:02d} | Val AUC: {val_metrics.auc_roc:.4f} | "
                    f"Val F1: {val_metrics.f1:.4f} | Train Loss: {train_metrics.loss:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

                # Save best model
                if val_metrics.auc_roc > self.best_auc:
                    self.best_auc = val_metrics.auc_roc
                    self.patience_counter = 0
                    self._save_checkpoint(best_model_path, val_metrics, scheduler)
                    mlflow.log_metric("best_val_auc", self.best_auc, step=epoch)
                    logger.success(f"New best model saved! (AUC: {self.best_auc:.4f})")
                else:
                    self.patience_counter += 1

                # Save periodic checkpoint
                if epoch % self.config.get("save_every", 5) == 0:
                    self._save_checkpoint(str(checkpoint_dir / f"epoch_{epoch:03d}.pth"), val_metrics, scheduler)

                # Early stopping
                if self.patience_counter >= patience:
                    logger.warning(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    break

            # Upload the final best model to MLflow (saves space instead of uploading every epoch)
            logger.info("Uploading best model artifact to MLflow...")
            if Path(best_model_path).exists():
                try:
                    # Load the clean model state for MLflow logging
                    # Use weights_only=False to support numpy scalars often saved in checkpoints
                    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
                    
                    if hasattr(self.model, "config_dict"):
                        clean_model = type(self.model)(**self.model.config_dict)
                    else:
                        # Fallback for models without config_dict
                        clean_model = self.model

                    clean_model.load_state_dict(checkpoint["state_dict"])
                    mlflow.pytorch.log_model(clean_model, "best_model")
                    logger.success("Best model successfully logged to MLflow.")
                except Exception as e:
                    logger.error(f"Failed to log model to MLflow: {e}")
                    logger.info("The local best_model.pth is still available in the output.")

        logger.success(f"Training complete! Best Val AUC: {self.best_auc:.4f}")
        return self.best_auc
