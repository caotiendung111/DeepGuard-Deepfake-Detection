"""
Optimize the production decision threshold from real prediction outputs.

Expected CSV columns:
  label, probability

The previous version simulated predictions and wrote them to configs/thresholds.yaml,
which made the API look calibrated even when it was not. This script now refuses to
write a threshold unless it has real labels and probabilities.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize DeepGuard threshold.")
    parser.add_argument("--predictions", default="reports/evaluation/predictions.csv")
    parser.add_argument("--output", default="configs/thresholds.yaml")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_predictions(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "label" not in df.columns or "probability" not in df.columns:
        raise ValueError("Predictions CSV must contain 'label' and 'probability' columns")

    labels = pd.to_numeric(df["label"], errors="coerce")
    probs = pd.to_numeric(df["probability"], errors="coerce")
    valid = labels.notna() & probs.notna()
    labels = labels[valid].astype(int).to_numpy()
    probs = probs[valid].astype(float).to_numpy()

    if len(labels) == 0:
        raise ValueError("No valid prediction rows found")
    if len(np.unique(labels)) < 2:
        raise ValueError("Threshold optimization requires both real and fake labels")
    return labels, probs


def choose_thresholds(labels: np.ndarray, probs: np.ndarray, min_recall: float) -> dict:
    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    youden_idx = int(np.argmax(tpr - fpr))
    default_threshold = float(roc_thresholds[youden_idx])

    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    high_recall_threshold = default_threshold
    if len(pr_thresholds):
        valid = np.where(recall[:-1] >= min_recall)[0]
        if len(valid):
            f1_scores = [
                f1_score(labels, (probs >= pr_thresholds[idx]).astype(int))
                for idx in valid
            ]
            high_recall_threshold = float(pr_thresholds[valid[int(np.argmax(f1_scores))]])

    default_preds = (probs >= default_threshold).astype(int)
    return {
        "default_threshold": default_threshold,
        "high_recall_threshold": high_recall_threshold,
        "auc_roc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, default_preds)),
        "n_samples": int(len(labels)),
        "generated_by": "scripts/optimize_ensemble_threshold.py",
    }


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise SystemExit(
            f"Predictions not found: {predictions_path}. "
            "Run scripts/evaluate.py first or pass --predictions."
        )

    labels, probs = load_predictions(str(predictions_path))
    threshold_config = choose_thresholds(labels, probs, args.min_recall)
    threshold_config["source"] = str(predictions_path)

    print(yaml.safe_dump(threshold_config, sort_keys=False))
    if args.dry_run:
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(threshold_config, f, sort_keys=False)
    print(f"Saved threshold config to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
