"""
Tune decision threshold and adaptive-TTA band from validation predictions.

Input CSV formats:
1) Minimal:
   label,probability
2) With robust-combination columns:
   label,face_probability_fake,full_probability_fake

The script does not run model inference; use it after exporting predictions from
your validation set.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description="Tune DeepGuard inference thresholds")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output", default="reports/tuning/inference_thresholds.json")
    parser.add_argument("--markdown-output", default="reports/tuning/inference_thresholds.md")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--prob-column", default="probability")
    return parser.parse_args()


def robust_probability(face_prob: np.ndarray, full_prob: np.ndarray) -> np.ndarray:
    disagreement = np.abs(face_prob - full_prob)
    return np.where(disagreement >= 0.30, np.minimum(face_prob, full_prob), 0.5 * face_prob + 0.5 * full_prob)


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> dict:
    candidates = np.linspace(0.01, 0.99, 99)
    best = None
    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            zero_division=0,
        )
        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    best["auc_roc"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else None
    return best


def tune_adaptive_band(labels: np.ndarray, base_probs: np.ndarray, tta_probs: np.ndarray | None = None) -> dict:
    """
    Estimate best adaptive band. If tta_probs are unavailable, this estimates
    cost only and returns the default band.
    """
    if tta_probs is None:
        low, high = 0.4, 0.6
        return {
            "low": low,
            "high": high,
            "coverage": float(np.mean((base_probs >= low) & (base_probs <= high))),
            "note": "No tta_probability column found; default adaptive band retained.",
        }

    best = None
    for low in np.arange(0.2, 0.51, 0.05):
        for high in np.arange(0.5, 0.81, 0.05):
            if low >= high:
                continue
            use_tta = (base_probs >= low) & (base_probs <= high)
            final_probs = np.where(use_tta, tta_probs, base_probs)
            tuned = best_threshold(labels, final_probs)
            row = {
                "low": float(low),
                "high": float(high),
                "coverage": float(np.mean(use_tta)),
                "f1": tuned["f1"],
                "threshold": tuned["threshold"],
            }
            if best is None or (row["f1"], -row["coverage"]) > (best["f1"], -best["coverage"]):
                best = row
    return best


def markdown(report: dict) -> str:
    lines = [
        "# DeepGuard Threshold Tuning",
        "",
        "## Primary Probability",
        "",
        f"- threshold: `{report['primary']['threshold']:.4f}`",
        f"- F1: `{report['primary']['f1']:.4f}`",
        f"- precision: `{report['primary']['precision']:.4f}`",
        f"- recall: `{report['primary']['recall']:.4f}`",
        f"- AUC ROC: `{report['primary'].get('auc_roc')}`",
        "",
        "## Adaptive TTA Band",
        "",
        f"- low: `{report['adaptive_tta']['low']:.2f}`",
        f"- high: `{report['adaptive_tta']['high']:.2f}`",
        f"- estimated coverage: `{report['adaptive_tta']['coverage']:.2%}`",
    ]
    if "robust" in report:
        lines.extend([
            "",
            "## Robust Face + Full",
            "",
            f"- threshold: `{report['robust']['threshold']:.4f}`",
            f"- F1: `{report['robust']['f1']:.4f}`",
            f"- precision: `{report['robust']['precision']:.4f}`",
            f"- recall: `{report['robust']['recall']:.4f}`",
        ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions_csv)
    labels = df[args.label_column].astype(int).to_numpy()
    probs = df[args.prob_column].astype(float).to_numpy()

    report = {
        "source": args.predictions_csv,
        "n_samples": int(len(df)),
        "primary": best_threshold(labels, probs),
        "adaptive_tta": tune_adaptive_band(
            labels,
            probs,
            df["tta_probability"].astype(float).to_numpy() if "tta_probability" in df.columns else None,
        ),
    }

    if {"face_probability_fake", "full_probability_fake"}.issubset(df.columns):
        robust_probs = robust_probability(
            df["face_probability_fake"].astype(float).to_numpy(),
            df["full_probability_fake"].astype(float).to_numpy(),
        )
        report["robust"] = best_threshold(labels, robust_probs)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = Path(args.markdown_output)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(markdown(report), encoding="utf-8")

    print(f"Wrote tuning JSON to {output}")
    print(f"Wrote tuning Markdown to {md}")


if __name__ == "__main__":
    main()
