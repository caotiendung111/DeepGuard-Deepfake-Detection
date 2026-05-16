"""
DeepGuard — Dataset Statistics & Quality Checker
Thống kê dataset, phát hiện ảnh hỏng và duplicate (MD5).

Usage:
    python scripts/data/dataset_stats.py --data-dir data/faces
    python scripts/data/dataset_stats.py --data-dir data/faces --output-dir reports/
    python scripts/data/dataset_stats.py --data-dir data/faces --remove-duplicates
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/dataset_stats.log", rotation="10 MB", level="DEBUG")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class ImageInfo:
    path: str
    label: str
    width: int
    height: int
    channels: int
    file_size_bytes: int
    md5: str
    is_valid: bool
    error_msg: str = ""


@dataclass
class DatasetReport:
    # Counts
    total_images: int = 0
    real_count: int = 0
    fake_count: int = 0
    valid_images: int = 0
    corrupt_images: int = 0
    duplicate_count: int = 0
    unique_images: int = 0

    # Size stats (pixels)
    min_width: int = 0
    max_width: int = 0
    mean_width: float = 0.0
    min_height: int = 0
    max_height: int = 0
    mean_height: float = 0.0

    # File size
    total_size_mb: float = 0.0
    mean_size_kb: float = 0.0

    # Resolution distribution
    resolution_counts: Dict[str, int] = field(default_factory=dict)

    # Duplicate files
    duplicate_groups: int = 0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not isinstance(v, dict)}
        d["imbalance_ratio"] = (
            round(self.fake_count / self.real_count, 2)
            if self.real_count > 0 else float("inf")
        )
        return d


# ── Core utilities ─────────────────────────────────────────────────────────────
def compute_md5(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except (IOError, OSError) as e:
        logger.warning(f"MD5 error for {filepath.name}: {e}")
        return ""


def read_image_info(image_path: Path, label: str, compute_hash: bool = True) -> ImageInfo:
    """Read image metadata and optionally compute MD5."""
    file_size = image_path.stat().st_size if image_path.exists() else 0
    md5 = compute_md5(image_path) if compute_hash else ""

    # Try reading with OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        return ImageInfo(
            path=str(image_path), label=label,
            width=0, height=0, channels=0,
            file_size_bytes=file_size, md5=md5,
            is_valid=False,
            error_msg="OpenCV cannot read image",
        )

    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1

    return ImageInfo(
        path=str(image_path), label=label,
        width=w, height=h, channels=c,
        file_size_bytes=file_size, md5=md5,
        is_valid=True,
    )


def discover_images(data_dir: Path) -> List[Tuple[Path, str]]:
    """Find all images in data_dir/real/ and data_dir/fake/."""
    images = []
    for label in ["real", "fake"]:
        label_dir = data_dir / label
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue
        found = [
            (f, label)
            for f in label_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images.extend(found)
        logger.info(f"  Found {len(found):,} {label} images")
    return images


# ── Main analysis ──────────────────────────────────────────────────────────────
def analyze_dataset(
    data_dir: Path,
    compute_hashes: bool = True,
    remove_duplicates: bool = False,
    remove_corrupt: bool = False,
) -> Tuple[DatasetReport, List[ImageInfo]]:
    """
    Full dataset analysis.

    Args:
        data_dir: Root directory with real/ and fake/ subdirectories.
        compute_hashes: Compute MD5 for duplicate detection.
        remove_duplicates: Delete duplicate files.
        remove_corrupt: Delete corrupt/unreadable images.

    Returns:
        (DatasetReport, list of ImageInfo)
    """
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Dataset Analysis")
    logger.info(f"Directory: {data_dir.absolute()}")
    logger.info("=" * 60)

    image_files = discover_images(data_dir)
    if not image_files:
        logger.error("No images found!")
        return DatasetReport(), []

    logger.info(f"\nAnalyzing {len(image_files):,} images...")
    if not compute_hashes:
        logger.info("(MD5 hashing disabled — duplicate detection skipped)")

    # Process each image
    infos: List[ImageInfo] = []
    for img_path, label in tqdm(image_files, desc="Scanning images", ncols=90, unit="img"):
        info = read_image_info(img_path, label, compute_hash=compute_hashes)
        infos.append(info)

    # Separate valid and corrupt
    valid = [i for i in infos if i.is_valid]
    corrupt = [i for i in infos if not i.is_valid]

    logger.info(f"\nValid: {len(valid):,} | Corrupt: {len(corrupt):,}")

    if corrupt:
        logger.warning(f"\n⚠️  {len(corrupt)} corrupt images:")
        for info in corrupt[:10]:
            logger.warning(f"  ✗ {Path(info.path).name}: {info.error_msg}")
        if len(corrupt) > 10:
            logger.warning(f"  ... and {len(corrupt) - 10} more")

        if remove_corrupt:
            for info in corrupt:
                try:
                    Path(info.path).unlink()
                    logger.debug(f"Deleted corrupt: {info.path}")
                except Exception as e:
                    logger.error(f"Failed to delete {info.path}: {e}")
            logger.info(f"Removed {len(corrupt)} corrupt files")

    # Duplicate detection
    hash_groups: Dict[str, List[str]] = defaultdict(list)
    if compute_hashes:
        for info in valid:
            if info.md5:
                hash_groups[info.md5].append(info.path)

    duplicate_paths: Set[str] = set()
    dup_groups = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}

    for paths in dup_groups.values():
        # Keep first occurrence, mark rest as duplicates
        for dup_path in paths[1:]:
            duplicate_paths.add(dup_path)

    if dup_groups:
        logger.warning(f"\n⚠️  Found {len(duplicate_paths)} duplicate images in {len(dup_groups)} groups")
        for i, (md5, paths) in enumerate(list(dup_groups.items())[:5]):
            logger.warning(f"  Group {i+1} (md5={md5[:8]}...):")
            for p in paths[:3]:
                logger.warning(f"    → {Path(p).name}")

        if remove_duplicates:
            removed = 0
            for dup_path in duplicate_paths:
                try:
                    Path(dup_path).unlink()
                    removed += 1
                except Exception as e:
                    logger.error(f"Failed to delete {dup_path}: {e}")
            logger.info(f"Removed {removed} duplicate files")

    # Compute statistics from valid images
    widths = [i.width for i in valid]
    heights = [i.height for i in valid]
    file_sizes = [i.file_size_bytes for i in valid]

    real_valid = [i for i in valid if i.label == "real"]
    fake_valid = [i for i in valid if i.label == "fake"]

    # Resolution distribution
    res_counter = Counter(f"{i.width}×{i.height}" for i in valid)

    report = DatasetReport(
        total_images=len(infos),
        real_count=len([i for i in infos if i.label == "real"]),
        fake_count=len([i for i in infos if i.label == "fake"]),
        valid_images=len(valid),
        corrupt_images=len(corrupt),
        duplicate_count=len(duplicate_paths),
        unique_images=len(valid) - len(duplicate_paths),
        min_width=min(widths) if widths else 0,
        max_width=max(widths) if widths else 0,
        mean_width=float(np.mean(widths)) if widths else 0.0,
        min_height=min(heights) if heights else 0,
        max_height=max(heights) if heights else 0,
        mean_height=float(np.mean(heights)) if heights else 0.0,
        total_size_mb=sum(file_sizes) / 1e6,
        mean_size_kb=float(np.mean(file_sizes)) / 1000 if file_sizes else 0.0,
        resolution_counts=dict(res_counter.most_common(20)),
        duplicate_groups=len(dup_groups),
    )

    elapsed = time.perf_counter() - t0
    logger.info(f"\nAnalysis completed in {elapsed:.1f}s")

    return report, infos


def print_report(report: DatasetReport):
    """Print formatted dataset report."""
    logger.info("\n" + "=" * 60)
    logger.info("DATASET REPORT")
    logger.info("=" * 60)
    logger.info(f"  Total images     : {report.total_images:,}")
    logger.info(f"    Real           : {report.real_count:,}")
    logger.info(f"    Fake           : {report.fake_count:,}")
    if report.real_count > 0:
        ratio = report.fake_count / report.real_count
        logger.info(f"    Fake/Real ratio: {ratio:.2f}x")
    logger.info(f"  Valid images     : {report.valid_images:,}")
    logger.info(f"  Corrupt images   : {report.corrupt_images:,}")
    logger.info(f"  Duplicates found : {report.duplicate_count:,} ({report.duplicate_groups} groups)")
    logger.info(f"  Unique images    : {report.unique_images:,}")
    logger.info(f"\n  Width   : min={report.min_width} | max={report.max_width} | mean={report.mean_width:.0f}")
    logger.info(f"  Height  : min={report.min_height} | max={report.max_height} | mean={report.mean_height:.0f}")
    logger.info(f"\n  Total size       : {report.total_size_mb:.1f} MB")
    logger.info(f"  Mean file size   : {report.mean_size_kb:.1f} KB")

    if report.resolution_counts:
        logger.info("\n  Top resolutions:")
        for res, count in list(report.resolution_counts.items())[:5]:
            pct = count / report.total_images * 100
            logger.info(f"    {res:12s}: {count:,} ({pct:.1f}%)")
    logger.info("=" * 60)


def save_report(
    report: DatasetReport,
    infos: List[ImageInfo],
    output_dir: Path,
):
    """Save report as JSON, CSV, and per-image CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary JSON
    json_path = output_dir / "dataset_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved: {json_path}")

    # 2. Summary CSV
    csv_path = output_dir / "dataset_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report.to_dict().keys())
        writer.writeheader()
        writer.writerow(report.to_dict())
    logger.info(f"Summary CSV: {csv_path}")

    # 3. Per-image CSV
    img_csv_path = output_dir / "image_catalog.csv"
    if infos:
        fields = ["path", "label", "width", "height", "channels",
                  "file_size_bytes", "md5", "is_valid", "error_msg"]
        with open(img_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for info in infos:
                writer.writerow({k: getattr(info, k) for k in fields})
    logger.info(f"Image catalog: {img_csv_path}")

    # 4. Corrupt files list
    corrupt = [i for i in infos if not i.is_valid]
    if corrupt:
        corrupt_path = output_dir / "corrupt_images.txt"
        with open(corrupt_path, "w", encoding="utf-8") as f:
            for info in corrupt:
                f.write(f"{info.path}\t{info.error_msg}\n")
        logger.info(f"Corrupt list: {corrupt_path}")

    # 5. Duplicate groups
    hash_groups: Dict[str, List[str]] = defaultdict(list)
    for info in infos:
        if info.is_valid and info.md5:
            hash_groups[info.md5].append(info.path)
    dup_groups = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}

    if dup_groups:
        dup_path = output_dir / "duplicate_groups.json"
        with open(dup_path, "w", encoding="utf-8") as f:
            json.dump(dup_groups, f, indent=2)
        logger.info(f"Duplicate groups: {dup_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Dataset Statistics")
    parser.add_argument("--data-dir", type=str, default="data/faces",
                        help="Root directory with real/ and fake/ subdirs")
    parser.add_argument("--output-dir", type=str, default="reports/dataset",
                        help="Output directory for reports")
    parser.add_argument("--no-hash", action="store_true",
                        help="Skip MD5 hashing (faster but no duplicate detection)")
    parser.add_argument("--remove-duplicates", action="store_true",
                        help="Delete duplicate files (keeps first occurrence)")
    parser.add_argument("--remove-corrupt", action="store_true",
                        help="Delete unreadable/corrupt image files")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        sys.exit(1)

    report, infos = analyze_dataset(
        data_dir=data_dir,
        compute_hashes=not args.no_hash,
        remove_duplicates=args.remove_duplicates,
        remove_corrupt=args.remove_corrupt,
    )

    print_report(report)
    save_report(report, infos, output_dir)

    logger.success(f"\n✅ Reports saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
