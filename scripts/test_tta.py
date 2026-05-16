import argparse
import hashlib
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config
from src.data.dataset import build_datasets, create_dataloader
from src.models import build_model
from src.training.metrics import compute_metrics
import mlflow

def apply_tta(image: torch.Tensor) -> list[torch.Tensor]:
    """Generate 5 TTA augmented versions of the image batch."""
    # 1. Original
    img1 = image
    # 2. Horizontal Flip
    img2 = TF.hflip(image)
    # 3. Brightness + 10%
    img3 = TF.adjust_brightness(image, 1.1)
    # 4. Brightness - 10%
    img4 = TF.adjust_brightness(image, 0.9)
    # 5. Slight rotation (5 degrees)
    img5 = TF.rotate(image, 5)
    return [img1, img2, img3, img4, img5]

def test_tta():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ckpt", default="models/checkpoints/best_model.pth")
    parser.add_argument("--run-name", default="Exp_D_TTA")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        print(f"[!] Checkpoint not found: {args.ckpt}")
        if args.dry_run:
            print("[DRY-RUN] TTA would run here.")
            return
        sys.exit(1)

    hash_before = hashlib.md5(open(args.ckpt, "rb").read()).hexdigest()
    
    if args.dry_run:
        print("[DRY-RUN] TTA Evaluation (No actual inference)")
        return

    print("=== Running Experiment D: Test Time Augmentation (TTA) ===")
    
    cfg = load_config(args.config)
    
    # We only need val/test dataset
    try:
        from src.data.transforms import build_transforms_from_config
        transforms = build_transforms_from_config(cfg)
        _, val_ds, _ = build_datasets(config=cfg, train_transform=transforms["train"], val_transform=transforms["val"])
        val_loader = create_dataloader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    except Exception as e:
        print(f"[!] Could not load dataset for TTA: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone=cfg.backbone, num_classes=cfg.num_classes, pretrained=False)
    
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    print("Evaluating with 5x TTA...")
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            tta_images = apply_tta(images)
            batch_probs = []
            
            for tta_img in tta_images:
                logits = model(tta_img)
                probs = torch.sigmoid(logits).squeeze(1)
                batch_probs.append(probs)
                
            # Mean probability across 5 augmentations
            mean_probs = torch.stack(batch_probs).mean(dim=0)
            
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(mean_probs.cpu().numpy().tolist())

    metrics = compute_metrics(
        labels=np.array(all_labels),
        predictions=(np.array(all_probs) >= 0.5).astype(int),
        probabilities=np.array(all_probs)
    )
    
    print(f"TTA Results: AUC={metrics.auc_roc:.4f} | F1={metrics.f1:.4f} | Prec={metrics.precision:.4f} | Rec={metrics.recall:.4f}")

    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("tta_passes", 5)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.to_dict().items()})

    hash_after = hashlib.md5(open(args.ckpt, "rb").read()).hexdigest()
    assert hash_before == hash_after, "TTA đã vô tình modify checkpoint!"
    print("[PASS] Checkpoint integrity verified.")

if __name__ == "__main__":
    test_tta()
