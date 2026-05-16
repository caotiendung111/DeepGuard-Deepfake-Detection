"""
DeepGuard — Dataset Splitter (Video-level split)
Chia train/val/test theo VIDEO (không theo frame) để tránh data leakage.

Strategy:
  - Group frames by video ID
  - Split video IDs → train/val/test
  - All frames từ cùng 1 video đi vào cùng 1 split

Usage:
    python scripts/data/split_dataset.py --data-dir data/faces --output-dir data/metadata
    python scripts/data/split_dataset.py --data-dir data/faces --output-dir data/metadata --train 0.7 --val 0.15 --test 0.15
    python scripts/data/split_dataset.py --data-dir data/faces --output-dir data/metadata --seed 42 --stratify
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/split_dataset.log", rotation="10 MB", level="DEBUG")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class VideoGroup:
    """All frames belonging to one video."""
    video_id: str
    label: str       # "real" or "fake"
    label_int: int   # 0=real, 1=fake
    frame_paths: List[str]
    n_frames: int


@dataclass
class SplitRecord:
    """One row in the output CSV."""
    filepath: str
    label: int       # 0=real, 1=fake
    label_name: str  # "real" or "fake"
    video_id: str
    split: str       # "train" | "val" | "test"


@dataclass
class SplitStats:
    split: str
    n_videos_real: int
    n_videos_fake: int
    n_frames_real: int
    n_frames_fake: int

    @property
    def n_videos(self): return self.n_videos_real + self.n_videos_fake
    @property
    def n_frames(self): return self.n_frames_real + self.n_frames_fake
    @property
    def fake_ratio(self):
        total = self.n_frames_real + self.n_frames_fake
        return self.n_frames_fake / total if total > 0 else 0


# ── Discovery ──────────────────────────────────────────────────────────────────
def discover_video_groups(data_dir: Path) -> Dict[str, VideoGroup]:
    """
    Scan data_dir/real/ and data_dir/fake/ for video subdirectories.

    Expected structure (from detect_faces.py output):
        data/faces/real/<video_id>/frame_0000001.jpg
        data/faces/fake/<video_id>/frame_0000001.jpg

    Also supports flat structure:
        data/faces/real/frame_0000001.jpg  → video_id = "real_flat"

    Returns:
        Dict mapping video_id → VideoGroup
    """
    groups: Dict[str, VideoGroup] = {}

    for label_name, label_int in [("real", 0), ("fake", 1)]:
        label_dir = data_dir / label_name
        if not label_dir.exists():
            logger.warning(f"Directory not found: {label_dir}")
            continue

        # Check if there are video subdirectories
        subdirs = [d for d in label_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Structured: real/<video_id>/frames
            for video_dir in tqdm(subdirs, desc=f"Discovering {label_name} videos", ncols=80):
                frames = sorted([
                    str(f) for f in video_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                ])
                if not frames:
                    continue

                video_id = f"{label_name}_{video_dir.name}"
                groups[video_id] = VideoGroup(
                    video_id=video_id,
                    label=label_name,
                    label_int=label_int,
                    frame_paths=frames,
                    n_frames=len(frames),
                )
        else:
            # Flat: real/*.jpg
            frames = sorted([
                str(f) for f in label_dir.glob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ])
            if frames:
                video_id = f"{label_name}_flat"
                logger.warning(
                    f"Flat structure detected in {label_dir}. "
                    f"Using '{video_id}' as single video group ({len(frames)} frames). "
                    "This may cause data leakage if frames are from multiple videos!"
                )
                groups[video_id] = VideoGroup(
                    video_id=video_id,
                    label=label_name,
                    label_int=label_int,
                    frame_paths=frames,
                    n_frames=len(frames),
                )

        logger.info(
            f"  {label_name}: {len([g for g in groups.values() if g.label == label_name])} videos, "
            f"{sum(g.n_frames for g in groups.values() if g.label == label_name):,} frames"
        )

    return groups


# ── Splitting logic ────────────────────────────────────────────────────────────
def split_video_ids(
    video_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split list of video IDs into train/val/test sets.
    Ratios are normalized to sum to 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    ids = list(video_ids)
    random.seed(seed)
    random.shuffle(ids)

    n = len(ids)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_test - n_val

    if n_train <= 0:
        logger.warning(f"Not enough videos ({n}) for 3-way split! Adjusting...")
        n_train = max(1, n - 2)
        n_val = 1
        n_test = max(0, n - n_train - n_val)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def stratified_split(
    groups: Dict[str, VideoGroup],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratified split: split real and fake videos separately,
    then merge to maintain class balance within each split.
    """
    real_ids = [vid for vid, g in groups.items() if g.label == "real"]
    fake_ids = [vid for vid, g in groups.items() if g.label == "fake"]

    real_train, real_val, real_test = split_video_ids(real_ids, train_ratio, val_ratio, test_ratio, seed)
    fake_train, fake_val, fake_test = split_video_ids(fake_ids, train_ratio, val_ratio, test_ratio, seed + 1)

    train = real_train + fake_train
    val = real_val + fake_val
    test = real_test + fake_test

    return train, val, test


def create_split_records(
    groups: Dict[str, VideoGroup],
    split_assignment: Dict[str, str],
) -> List[SplitRecord]:
    """Create flat list of SplitRecord (one per image frame)."""
    records = []
    for video_id, split_name in split_assignment.items():
        group = groups[video_id]
        for frame_path in group.frame_paths:
            records.append(SplitRecord(
                filepath=frame_path,
                label=group.label_int,
                label_name=group.label,
                video_id=video_id,
                split=split_name,
            ))
    return records


def compute_split_stats(
    records: List[SplitRecord],
    split_name: str,
    groups: Dict[str, VideoGroup],
) -> SplitStats:
    """Compute statistics for a single split."""
    split_records = [r for r in records if r.split == split_name]
    video_ids = set(r.video_id for r in split_records)

    n_videos_real = len([v for v in video_ids if groups[v].label == "real"])
    n_videos_fake = len([v for v in video_ids if groups[v].label == "fake"])
    n_frames_real = sum(1 for r in split_records if r.label == 0)
    n_frames_fake = sum(1 for r in split_records if r.label == 1)

    return SplitStats(
        split=split_name,
        n_videos_real=n_videos_real,
        n_videos_fake=n_videos_fake,
        n_frames_real=n_frames_real,
        n_frames_fake=n_frames_fake,
    )


def save_split_csvs(
    records: List[SplitRecord],
    output_dir: Path,
    save_combined: bool = True,
) -> Dict[str, Path]:
    """Save train/val/test CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {}
    fields = ["filepath", "label", "label_name", "video_id"]

    for split_name in ["train", "val", "test"]:
        split_records = [r for r in records if r.split == split_name]
        if not split_records:
            continue

        csv_path = output_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for record in split_records:
                writer.writerow({k: getattr(record, k) for k in fields})

        split_paths[split_name] = csv_path
        logger.info(f"Saved: {csv_path} ({len(split_records):,} frames)")

    # Combined CSV with split column
    if save_combined:
        combined_path = output_dir / "all_splits.csv"
        all_fields = fields + ["split"]
        with open(combined_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for record in records:
                writer.writerow({k: getattr(record, k) for k in all_fields})
        split_paths["all"] = combined_path
        logger.info(f"Saved: {combined_path} ({len(records):,} total frames)")

    return split_paths


def verify_no_leakage(records: List[SplitRecord]) -> bool:
    """
    Verify no video appears in multiple splits.
    Returns True if no leakage detected.
    """
    video_splits: Dict[str, Set[str]] = defaultdict(set)
    for record in records:
        video_splits[record.video_id].add(record.split)

    leakage_videos = {vid: splits for vid, splits in video_splits.items() if len(splits) > 1}

    if leakage_videos:
        logger.error(f"⚠️  DATA LEAKAGE DETECTED! {len(leakage_videos)} videos in multiple splits:")
        for vid, splits in list(leakage_videos.items())[:5]:
            logger.error(f"  {vid}: {splits}")
        return False

    logger.success("✅ No data leakage — all videos appear in exactly one split")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def create_splits(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    save_combined: bool = True,
) -> Dict[str, SplitStats]:
    """
    Full pipeline: discover → split → save → verify.
    """
    logger.info("=" * 60)
    logger.info("DeepGuard — Video-Level Dataset Splitter")
    logger.info("=" * 60)
    logger.info(f"Data dir  : {data_dir.absolute()}")
    logger.info(f"Output dir: {output_dir.absolute()}")
    logger.info(f"Split     : train={train_ratio:.0%} | val={val_ratio:.0%} | test={test_ratio:.0%}")
    logger.info(f"Seed      : {seed} | Stratified: {stratify}")

    # Discover
    groups = discover_video_groups(data_dir)
    if not groups:
        logger.error("No video groups found!")
        return {}

    n_real = sum(1 for g in groups.values() if g.label == "real")
    n_fake = sum(1 for g in groups.values() if g.label == "fake")
    total_frames = sum(g.n_frames for g in groups.values())

    logger.info(f"\nDiscovery complete:")
    logger.info(f"  Videos: {len(groups):,} total ({n_real} real | {n_fake} fake)")
    logger.info(f"  Frames: {total_frames:,} total")

    if len(groups) < 3:
        logger.error(f"Need at least 3 videos for split, found {len(groups)}")
        return {}

    # Split
    if stratify:
        train_ids, val_ids, test_ids = stratified_split(
            groups, train_ratio, val_ratio, test_ratio, seed
        )
    else:
        all_ids = list(groups.keys())
        train_ids, val_ids, test_ids = split_video_ids(
            all_ids, train_ratio, val_ratio, test_ratio, seed
        )

    # Build assignment map
    assignment: Dict[str, str] = {}
    for vid in train_ids:
        assignment[vid] = "train"
    for vid in val_ids:
        assignment[vid] = "val"
    for vid in test_ids:
        assignment[vid] = "test"

    # Unassigned videos (if any due to rounding) → train
    for vid in groups:
        if vid not in assignment:
            assignment[vid] = "train"
            logger.debug(f"Unassigned video → train: {vid}")

    # Create records
    records = create_split_records(groups, assignment)

    # Verify no leakage
    no_leakage = verify_no_leakage(records)
    if not no_leakage:
        logger.error("Aborting due to data leakage!")
        return {}

    # Save CSVs
    save_split_csvs(records, output_dir, save_combined)

    # Compute and print stats
    split_stats = {}
    logger.info("\n" + "=" * 60)
    logger.info("SPLIT STATISTICS")
    logger.info("=" * 60)
    logger.info(f"{'Split':<8} {'Videos':>8} {'R-Vid':>7} {'F-Vid':>7} {'Frames':>10} {'Real':>8} {'Fake':>8} {'Fake%':>7}")
    logger.info("-" * 68)

    for split_name in ["train", "val", "test"]:
        stats = compute_split_stats(records, split_name, groups)
        split_stats[split_name] = stats
        logger.info(
            f"{split_name:<8} {stats.n_videos:>8,} {stats.n_videos_real:>7,} {stats.n_videos_fake:>7,} "
            f"{stats.n_frames:>10,} {stats.n_frames_real:>8,} {stats.n_frames_fake:>8,} "
            f"{stats.fake_ratio:>6.1%}"
        )

    logger.info("-" * 68)
    total_stats = SplitStats(
        split="TOTAL",
        n_videos_real=n_real,
        n_videos_fake=n_fake,
        n_frames_real=sum(g.n_frames for g in groups.values() if g.label == "real"),
        n_frames_fake=sum(g.n_frames for g in groups.values() if g.label == "fake"),
    )
    logger.info(
        f"{'TOTAL':<8} {total_stats.n_videos:>8,} {total_stats.n_videos_real:>7,} "
        f"{total_stats.n_videos_fake:>7,} {total_stats.n_frames:>10,} "
        f"{total_stats.n_frames_real:>8,} {total_stats.n_frames_fake:>8,} "
        f"{total_stats.fake_ratio:>6.1%}"
    )
    logger.info("=" * 60)

    # Save split metadata JSON
    meta = {
        "seed": seed,
        "stratify": stratify,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {
            name: {
                "n_videos": s.n_videos,
                "n_videos_real": s.n_videos_real,
                "n_videos_fake": s.n_videos_fake,
                "n_frames": s.n_frames,
                "n_frames_real": s.n_frames_real,
                "n_frames_fake": s.n_frames_fake,
                "fake_ratio": s.fake_ratio,
            }
            for name, s in split_stats.items()
        },
        "video_assignment": assignment,
    }
    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSplit metadata saved: {meta_path}")

    return split_stats


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepGuard Video-level Dataset Splitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Default 70/15/15 split:
    python scripts/data/split_dataset.py --data-dir data/faces --output-dir data/metadata

  Custom ratios:
    python scripts/data/split_dataset.py --data-dir data/faces --output-dir data/metadata \\
        --train 0.8 --val 0.1 --test 0.1

  Non-stratified split:
    python scripts/data/split_dataset.py --data-dir data/faces --no-stratify
        """,
    )
    parser.add_argument("--data-dir", type=str, default="data/faces",
                        help="Root directory with real/ and fake/ subdirs")
    parser.add_argument("--output-dir", type=str, default="data/metadata",
                        help="Output directory for CSV files")
    parser.add_argument("--train", type=float, default=0.70, dest="train_ratio",
                        help="Training set ratio (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.15, dest="val_ratio",
                        help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.15, dest="test_ratio",
                        help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified split (mix real/fake randomly)")
    parser.add_argument("--no-combined", action="store_true",
                        help="Don't save combined all_splits.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total <= 1.01):
        logger.warning(f"Ratios sum to {total:.2f}, normalizing to 1.0")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    stats = create_splits(
        data_dir=data_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,
        save_combined=not args.no_combined,
    )

    if stats:
        logger.success(f"\n✅ Split complete! CSVs saved to: {output_dir.absolute()}")
        logger.info("Next step: python scripts/train.py --config configs/efficientnet_b4.yaml")
    else:
        logger.error("Split failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
