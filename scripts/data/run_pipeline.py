"""
DeepGuard — Full Data Pipeline Runner
Chạy toàn bộ pipeline: extract → detect faces → stats → split

Usage:
    python scripts/data/run_pipeline.py --raw-dir data/raw --help
    python scripts/data/run_pipeline.py --raw-dir data/raw --fps 1 --face-size 224 --workers 4
    python scripts/data/run_pipeline.py --skip-extract --skip-detect --only-split  # chỉ chạy split
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")


def step_banner(step: int, title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"STEP {step}: {title}")
    logger.info("=" * 60)


def run_pipeline(args):
    t_total = time.perf_counter()

    raw_dir = Path(args.raw_dir)
    frames_dir = Path(args.frames_dir)
    faces_dir = Path(args.faces_dir)
    metadata_dir = Path(args.metadata_dir)
    reports_dir = Path(args.reports_dir)

    logger.info("=" * 60)
    logger.info("DeepGuard — Full Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"Raw data    : {raw_dir}")
    logger.info(f"Frames      : {frames_dir}")
    logger.info(f"Faces       : {faces_dir}")
    logger.info(f"Metadata    : {metadata_dir}")

    # ── STEP 1: Extract Frames ────────────────────────────────────────────────
    if not args.skip_extract:
        step_banner(1, "Extract Frames from Videos")
        from scripts.data.extract_frames import extract_all_videos
        summary = extract_all_videos(
            input_dir=raw_dir,
            output_dir=frames_dir,
            fps_sample=args.fps,
            max_frames=args.max_frames,
            num_workers=args.workers,
            skip_existing=not args.no_skip,
        )
        logger.success(f"Frames extracted: {summary.total_frames_extracted:,}")
    else:
        logger.info("⏭️  Skipping frame extraction")

    # ── STEP 2: Detect & Crop Faces ───────────────────────────────────────────
    if not args.skip_detect:
        step_banner(2, "Face Detection & Cropping")
        from scripts.data.detect_faces import process_all_frames
        result = process_all_frames(
            frames_dir=frames_dir,
            output_dir=faces_dir,
            backend=args.face_backend,
            output_size=args.face_size,
            padding=args.face_padding,
            num_workers=max(1, args.workers // 2),
            skip_existing=not args.no_skip,
        )
        logger.success(f"Faces saved: {result.get('total_faces', 0):,}")
    else:
        logger.info("⏭️  Skipping face detection")

    # ── STEP 3: Dataset Statistics ────────────────────────────────────────────
    if not args.skip_stats:
        step_banner(3, "Dataset Statistics & Quality Check")
        from scripts.data.dataset_stats import analyze_dataset, print_report, save_report
        data_source = faces_dir if faces_dir.exists() and any(faces_dir.rglob("*.jpg")) else frames_dir
        report, infos = analyze_dataset(
            data_dir=data_source,
            compute_hashes=not args.no_hash,
            remove_duplicates=args.remove_duplicates,
        )
        print_report(report)
        save_report(report, infos, reports_dir)
    else:
        logger.info("⏭️  Skipping statistics")

    # ── STEP 4: Train/Val/Test Split ──────────────────────────────────────────
    if not args.skip_split:
        step_banner(4, "Video-level Train/Val/Test Split")
        from scripts.data.split_dataset import create_splits
        data_source = faces_dir if faces_dir.exists() and any(faces_dir.rglob("*.jpg")) else frames_dir
        stats = create_splits(
            data_dir=data_source,
            output_dir=metadata_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
            stratify=not args.no_stratify,
        )

        if stats:
            logger.success("Split complete!")
            for split_name, s in stats.items():
                logger.info(f"  {split_name}: {s.n_frames:,} frames ({s.n_videos} videos)")
    else:
        logger.info("⏭️  Skipping split")

    # ── Done ───────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    logger.info("")
    logger.info("=" * 60)
    logger.success(f"✅ Pipeline complete! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Review reports/dataset/dataset_report.json")
    logger.info("  2. Start training: python scripts/train.py --config configs/efficientnet_b4.yaml")
    logger.info("  3. Track experiments: make mlflow-ui")


def parse_args():
    p = argparse.ArgumentParser(description="DeepGuard Full Data Pipeline")

    # Directories
    p.add_argument("--raw-dir", type=str, default="data/raw")
    p.add_argument("--frames-dir", type=str, default="data/frames")
    p.add_argument("--faces-dir", type=str, default="data/faces")
    p.add_argument("--metadata-dir", type=str, default="data/metadata")
    p.add_argument("--reports-dir", type=str, default="reports/dataset")

    # Frame extraction
    p.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames per video")

    # Face detection
    p.add_argument("--face-backend", choices=["mtcnn", "retinaface", "haar"],
                   default="mtcnn", help="Face detection backend")
    p.add_argument("--face-size", type=int, default=224, help="Output face size")
    p.add_argument("--face-padding", type=float, default=0.2, help="Padding around face")

    # Stats
    p.add_argument("--no-hash", action="store_true", help="Skip MD5 hashing")
    p.add_argument("--remove-duplicates", action="store_true", help="Remove duplicate images")

    # Split
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-stratify", action="store_true")

    # General
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-skip", action="store_true", help="Re-process existing files")

    # Skip flags
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--skip-stats", action="store_true")
    p.add_argument("--skip-split", action="store_true")
    p.add_argument("--only-split", action="store_true",
                   help="Shortcut: skip all steps except split")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.only_split:
        args.skip_extract = True
        args.skip_detect = True
        args.skip_stats = True

    run_pipeline(args)
