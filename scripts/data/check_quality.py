"""
DeepGuard — Dataset Quality Analyzer
Phát hiện ảnh blur, quá tối/sáng, và xuất danh sách cần loại bỏ.

Checks:
  - Blur detection: Laplacian variance < threshold
  - Brightness: mean pixel < 30 (too dark) or > 225 (too bright)
  - Low contrast: std pixel < 15
  - Tiny faces: width or height < 50px (before resize)
  - Zero-size / corrupt files

Usage:
    python scripts/data/check_quality.py --data-dir data/faces
    python scripts/data/check_quality.py --data-dir data/faces --blur-threshold 80 --remove
    python scripts/data/check_quality.py --data-dir data/faces --output-dir reports/quality
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/check_quality.log", rotation="10 MB", level="DEBUG")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class QualityIssue:
    filepath: str
    issue_type: str      # "blur" | "too_dark" | "too_bright" | "low_contrast" | "tiny" | "corrupt"
    severity: str        # "warning" | "critical"
    metric_name: str     # e.g., "laplacian_var", "mean_brightness"
    metric_value: float
    threshold: float
    label: str           # "real" or "fake"
    recommendation: str  # "remove" | "review"


@dataclass
class QualityReport:
    total_images: int = 0
    clean_images: int = 0
    blur_count: int = 0
    too_dark_count: int = 0
    too_bright_count: int = 0
    low_contrast_count: int = 0
    tiny_count: int = 0
    corrupt_count: int = 0
    total_issues: int = 0
    removal_candidates: int = 0
    review_candidates: int = 0

    # Per-class
    real_issues: int = 0
    fake_issues: int = 0

    # Global stats
    mean_blur_score: float = 0.0
    mean_brightness: float = 0.0
    mean_contrast: float = 0.0


# ── Quality checks ─────────────────────────────────────────────────────────────
def compute_blur_score(image_gray: np.ndarray) -> float:
    """
    Compute Laplacian variance as blur metric.
    Higher = sharper. Lower = blurrier.
    Typical threshold: 100 (very blurry below this).
    """
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def compute_brightness(image_gray: np.ndarray) -> float:
    """Mean pixel intensity (0-255)."""
    return float(np.mean(image_gray))


def compute_contrast(image_gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities."""
    return float(np.std(image_gray))


def check_image_quality(
    filepath: Path,
    label: str,
    blur_threshold: float = 100.0,
    dark_threshold: float = 30.0,
    bright_threshold: float = 225.0,
    contrast_threshold: float = 15.0,
    min_face_size: int = 50,
) -> Tuple[Optional[QualityIssue], Dict[str, float]]:
    """
    Run all quality checks on a single image.

    Returns:
        (QualityIssue or None, metrics_dict)
    """
    # Read image
    img = cv2.imread(str(filepath))
    if img is None:
        return QualityIssue(
            filepath=str(filepath),
            issue_type="corrupt",
            severity="critical",
            metric_name="readable",
            metric_value=0.0,
            threshold=1.0,
            label=label,
            recommendation="remove",
        ), {}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute metrics
    blur_score = compute_blur_score(gray)
    brightness = compute_brightness(gray)
    contrast = compute_contrast(gray)

    metrics = {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "width": w,
        "height": h,
    }

    # Check: tiny face
    if w < min_face_size or h < min_face_size:
        return QualityIssue(
            filepath=str(filepath),
            issue_type="tiny",
            severity="critical",
            metric_name="face_size",
            metric_value=min(w, h),
            threshold=min_face_size,
            label=label,
            recommendation="remove",
        ), metrics

    # Check: blur
    if blur_score < blur_threshold:
        severity = "critical" if blur_score < blur_threshold * 0.3 else "warning"
        return QualityIssue(
            filepath=str(filepath),
            issue_type="blur",
            severity=severity,
            metric_name="laplacian_var",
            metric_value=blur_score,
            threshold=blur_threshold,
            label=label,
            recommendation="remove" if severity == "critical" else "review",
        ), metrics

    # Check: too dark
    if brightness < dark_threshold:
        return QualityIssue(
            filepath=str(filepath),
            issue_type="too_dark",
            severity="warning",
            metric_name="mean_brightness",
            metric_value=brightness,
            threshold=dark_threshold,
            label=label,
            recommendation="review",
        ), metrics

    # Check: too bright
    if brightness > bright_threshold:
        return QualityIssue(
            filepath=str(filepath),
            issue_type="too_bright",
            severity="warning",
            metric_name="mean_brightness",
            metric_value=brightness,
            threshold=bright_threshold,
            label=label,
            recommendation="review",
        ), metrics

    # Check: low contrast
    if contrast < contrast_threshold:
        return QualityIssue(
            filepath=str(filepath),
            issue_type="low_contrast",
            severity="warning",
            metric_name="contrast_std",
            metric_value=contrast,
            threshold=contrast_threshold,
            label=label,
            recommendation="review",
        ), metrics

    # All checks passed
    return None, metrics


# ── Batch analysis ─────────────────────────────────────────────────────────────
def analyze_quality(
    data_dir: Path,
    blur_threshold: float = 100.0,
    dark_threshold: float = 30.0,
    bright_threshold: float = 225.0,
    contrast_threshold: float = 15.0,
    min_face_size: int = 50,
) -> Tuple[QualityReport, List[QualityIssue], List[Dict]]:
    """
    Analyze all images in data_dir/real/ and data_dir/fake/.

    Returns:
        (QualityReport, list of QualityIssue, list of all_metrics)
    """
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DeepGuard — Dataset Quality Analysis")
    logger.info("=" * 60)
    logger.info(f"Directory       : {data_dir.absolute()}")
    logger.info(f"Blur threshold  : {blur_threshold}")
    logger.info(f"Dark threshold  : < {dark_threshold}")
    logger.info(f"Bright threshold: > {bright_threshold}")
    logger.info(f"Contrast thresh : < {contrast_threshold}")
    logger.info(f"Min face size   : {min_face_size}px")

    # Discover images
    image_files: List[Tuple[Path, str]] = []
    for label in ["real", "fake"]:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        found = [
            (f, label) for f in sorted(label_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_files.extend(found)
        logger.info(f"  {label}: {len(found):,} images")

    if not image_files:
        logger.error(f"No images found in {data_dir}")
        return QualityReport(), [], []

    # Analyze each image
    issues: List[QualityIssue] = []
    all_metrics: List[Dict] = []
    blur_scores = []
    brightnesses = []
    contrasts = []

    for filepath, label in tqdm(image_files, desc="Quality analysis", ncols=90, unit="img"):
        issue, metrics = check_image_quality(
            filepath, label,
            blur_threshold=blur_threshold,
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,
            contrast_threshold=contrast_threshold,
            min_face_size=min_face_size,
        )

        if metrics:
            metrics["filepath"] = str(filepath)
            metrics["label"] = label
            metrics["has_issue"] = issue is not None
            all_metrics.append(metrics)

            blur_scores.append(metrics.get("blur_score", 0))
            brightnesses.append(metrics.get("brightness", 0))
            contrasts.append(metrics.get("contrast", 0))

        if issue:
            issues.append(issue)

    # Build report
    report = QualityReport(
        total_images=len(image_files),
        clean_images=len(image_files) - len(issues),
        blur_count=sum(1 for i in issues if i.issue_type == "blur"),
        too_dark_count=sum(1 for i in issues if i.issue_type == "too_dark"),
        too_bright_count=sum(1 for i in issues if i.issue_type == "too_bright"),
        low_contrast_count=sum(1 for i in issues if i.issue_type == "low_contrast"),
        tiny_count=sum(1 for i in issues if i.issue_type == "tiny"),
        corrupt_count=sum(1 for i in issues if i.issue_type == "corrupt"),
        total_issues=len(issues),
        removal_candidates=sum(1 for i in issues if i.recommendation == "remove"),
        review_candidates=sum(1 for i in issues if i.recommendation == "review"),
        real_issues=sum(1 for i in issues if i.label == "real"),
        fake_issues=sum(1 for i in issues if i.label == "fake"),
        mean_blur_score=float(np.mean(blur_scores)) if blur_scores else 0.0,
        mean_brightness=float(np.mean(brightnesses)) if brightnesses else 0.0,
        mean_contrast=float(np.mean(contrasts)) if contrasts else 0.0,
    )

    elapsed = time.perf_counter() - t0

    # Print report
    logger.info("\n" + "=" * 60)
    logger.info("QUALITY REPORT")
    logger.info("=" * 60)
    logger.info(f"  Total images     : {report.total_images:,}")
    logger.info(f"  Clean images     : {report.clean_images:,} ({report.clean_images/max(report.total_images,1)*100:.1f}%)")
    logger.info(f"  Total issues     : {report.total_issues:,}")
    logger.info(f"    🌀 Blur        : {report.blur_count:,}")
    logger.info(f"    🌑 Too dark    : {report.too_dark_count:,}")
    logger.info(f"    ☀️  Too bright  : {report.too_bright_count:,}")
    logger.info(f"    📉 Low contrast: {report.low_contrast_count:,}")
    logger.info(f"    🔍 Tiny face   : {report.tiny_count:,}")
    logger.info(f"    💥 Corrupt     : {report.corrupt_count:,}")
    logger.info(f"  ─────────────────────────────")
    logger.info(f"  🗑️  Remove candidates: {report.removal_candidates:,}")
    logger.info(f"  👁️  Review candidates: {report.review_candidates:,}")
    logger.info(f"  Issues in real   : {report.real_issues:,}")
    logger.info(f"  Issues in fake   : {report.fake_issues:,}")
    logger.info(f"\n  Global averages:")
    logger.info(f"    Blur score  : {report.mean_blur_score:.1f}")
    logger.info(f"    Brightness  : {report.mean_brightness:.1f}")
    logger.info(f"    Contrast    : {report.mean_contrast:.1f}")
    logger.info(f"\n  Elapsed: {elapsed:.1f}s")
    logger.info("=" * 60)

    return report, issues, all_metrics


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(
    report: QualityReport,
    issues: List[QualityIssue],
    all_metrics: List[Dict],
    output_dir: Path,
    remove_flagged: bool = False,
):
    """Save analysis results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Report JSON
    report_path = output_dir / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"Report saved: {report_path}")

    # 2. Issues CSV
    if issues:
        issues_path = output_dir / "quality_issues.csv"
        fields = ["filepath", "issue_type", "severity", "metric_name",
                  "metric_value", "threshold", "label", "recommendation"]
        with open(issues_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for issue in issues:
                writer.writerow(asdict(issue))
        logger.info(f"Issues CSV saved: {issues_path}")

    # 3. Removal list (simple txt)
    removal = [i for i in issues if i.recommendation == "remove"]
    if removal:
        removal_path = output_dir / "remove_list.txt"
        with open(removal_path, "w", encoding="utf-8") as f:
            for issue in removal:
                f.write(f"{issue.filepath}\t{issue.issue_type}\t{issue.metric_value:.2f}\n")
        logger.info(f"Removal list ({len(removal)} files): {removal_path}")

        # Actually remove files if requested
        if remove_flagged:
            removed = 0
            for issue in removal:
                try:
                    Path(issue.filepath).unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {issue.filepath}: {e}")
            logger.success(f"Removed {removed}/{len(removal)} flagged files")

    # 4. Full metrics CSV
    if all_metrics:
        metrics_path = output_dir / "image_metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        logger.info(f"All metrics CSV: {metrics_path}")

    # 5. Distribution plots
    try:
        _save_distribution_plots(all_metrics, output_dir)
    except Exception as e:
        logger.warning(f"Failed to save distribution plots: {e}")


def _save_distribution_plots(all_metrics: List[Dict], output_dir: Path):
    """Generate histograms for blur, brightness, contrast distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blur_scores = [m["blur_score"] for m in all_metrics if "blur_score" in m]
    brightnesses = [m["brightness"] for m in all_metrics if "brightness" in m]
    contrasts = [m["contrast"] for m in all_metrics if "contrast" in m]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Blur
    axes[0].hist(blur_scores, bins=50, color="#a78bfa", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[0].axvline(x=100, color="#ef233c", linestyle="--", linewidth=2, label="Threshold=100")
    axes[0].set_xlabel("Blur Score (Laplacian Var)")
    axes[0].set_title("Blur Distribution")
    axes[0].legend(facecolor="#16213e", labelcolor="white")

    # Brightness
    axes[1].hist(brightnesses, bins=50, color="#60a5fa", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[1].axvline(x=30, color="#ef233c", linestyle="--", linewidth=2, label="Dark < 30")
    axes[1].axvline(x=225, color="#fbbf24", linestyle="--", linewidth=2, label="Bright > 225")
    axes[1].set_xlabel("Mean Brightness")
    axes[1].set_title("Brightness Distribution")
    axes[1].legend(facecolor="#16213e", labelcolor="white", fontsize=8)

    # Contrast
    axes[2].hist(contrasts, bins=50, color="#34d399", alpha=0.85, edgecolor="white", linewidth=0.3)
    axes[2].axvline(x=15, color="#ef233c", linestyle="--", linewidth=2, label="Low < 15")
    axes[2].set_xlabel("Contrast (Std)")
    axes[2].set_title("Contrast Distribution")
    axes[2].legend(facecolor="#16213e", labelcolor="white")

    fig.suptitle("DeepGuard — Image Quality Distributions", fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()
    plot_path = output_dir / "quality_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Distribution plots saved: {plot_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Dataset Quality Checker")
    parser.add_argument("--data-dir", type=str, default="data/faces",
                        help="Root directory with real/ and fake/ subdirs")
    parser.add_argument("--output-dir", type=str, default="reports/quality",
                        help="Output directory for reports")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Laplacian variance threshold (default: 100)")
    parser.add_argument("--dark-threshold", type=float, default=30.0,
                        help="Mean brightness below this = too dark")
    parser.add_argument("--bright-threshold", type=float, default=225.0,
                        help="Mean brightness above this = too bright")
    parser.add_argument("--contrast-threshold", type=float, default=15.0,
                        help="Std below this = low contrast")
    parser.add_argument("--min-face-size", type=int, default=50,
                        help="Minimum face size in pixels")
    parser.add_argument("--remove", action="store_true",
                        help="Actually delete flagged 'remove' files")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        sys.exit(1)

    report, issues, all_metrics = analyze_quality(
        data_dir=data_dir,
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        contrast_threshold=args.contrast_threshold,
        min_face_size=args.min_face_size,
    )

    save_results(report, issues, all_metrics, output_dir, remove_flagged=args.remove)
    logger.success(f"\n✅ Quality analysis complete. Reports: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
