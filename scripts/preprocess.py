"""
DeepGuard — Data Preprocessing Script
Extracts frames from raw video files and prepares train/val/test splits.

Usage:
    python scripts/preprocess.py --data-dir data/raw --output-dir data/processed
    python scripts/preprocess.py --help
"""
import argparse
import os
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Data Preprocessing")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Root directory with 'real/' and 'fake/' subdirs")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for processed frames")
    parser.add_argument("--face-size", type=int, default=224,
                        help="Face crop size in pixels")
    parser.add_argument("--fps-sample", type=int, default=3,
                        help="Frames per second to extract from videos")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames per video (None = unlimited)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split ratio (0-1)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Test split ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-face-detect", action="store_true",
                        help="Skip face detection (process full frames)")
    return parser.parse_args()


def process_videos(data_dir: Path, output_dir: Path, args) -> list:
    """
    Process all videos in data_dir/real/ and data_dir/fake/.
    Returns list of dicts: {video_id, frame_dir, label, n_frames}
    """
    from src.data.video_processor import VideoProcessor

    processor = VideoProcessor(
        face_size=args.face_size,
        fps_sample=args.fps_sample,
        max_frames=args.max_frames,
        use_face_detector=not args.no_face_detect,
    )

    records = []
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        video_dir = data_dir / label_name
        if not video_dir.exists():
            print(f"⚠️  Directory not found: {video_dir}")
            continue

        video_files = (
            list(video_dir.glob("*.mp4")) +
            list(video_dir.glob("*.avi")) +
            list(video_dir.glob("*.mov"))
        )
        print(f"\n📁 Processing {len(video_files)} {label_name} videos...")

        for video_path in tqdm(video_files, desc=label_name.upper()):
            video_id = video_path.stem
            frame_out_dir = output_dir / label_name / video_id

            if frame_out_dir.exists() and any(frame_out_dir.iterdir()):
                # Already processed
                n_frames = len(list(frame_out_dir.glob("*.jpg")))
            else:
                frame_paths = processor.extract_frames(str(video_path), str(frame_out_dir))
                n_frames = len(frame_paths)

            if n_frames > 0:
                records.append({
                    "video_id": video_id,
                    "frame_dir": str(frame_out_dir),
                    "label": label_id,
                    "n_frames": n_frames,
                })

    return records


def process_images(data_dir: Path, output_dir: Path, args) -> list:
    """Process image files (no video extraction needed)."""
    records = []
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        img_dir = data_dir / label_name
        if not img_dir.exists():
            continue

        image_files = (
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.jpeg")) +
            list(img_dir.glob("*.png"))
        )
        print(f"🖼️  Found {len(image_files)} {label_name} images")

        # Copy/link images to output dir
        out_img_dir = output_dir / label_name
        out_img_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            records.append({
                "filepath": str(img_path),
                "label": label_id,
            })

    return records


def create_splits(records: list, val_split: float, test_split: float, seed: int):
    """Create train/val/test splits, stratified by label."""
    random.seed(seed)

    real_records = [r for r in records if r["label"] == 0]
    fake_records = [r for r in records if r["label"] == 1]

    def split(items):
        random.shuffle(items)
        n = len(items)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        return (
            items[n_test + n_val:],    # train
            items[n_test:n_test + n_val],  # val
            items[:n_test],             # test
        )

    real_train, real_val, real_test = split(real_records)
    fake_train, fake_val, fake_test = split(fake_records)

    train = real_train + fake_train
    val = real_val + fake_val
    test = real_test + fake_test

    random.shuffle(train)
    return train, val, test


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    metadata_dir = Path("data/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Data dir: {data_dir.absolute()}")
    print(f"📂 Output dir: {output_dir.absolute()}")

    # Auto-detect: video or image dataset
    has_videos = any(data_dir.rglob("*.mp4")) or any(data_dir.rglob("*.avi"))
    has_images = any(data_dir.rglob("*.jpg")) or any(data_dir.rglob("*.png"))

    if has_videos:
        print("🎥 Detected VIDEO dataset — extracting frames...")
        records = process_videos(data_dir, output_dir, args)
        df = pd.DataFrame(records)
        train, val, test = create_splits(records, args.val_split, args.test_split, args.seed)
        pd.DataFrame(train).to_csv(metadata_dir / "train.csv", index=False)
        pd.DataFrame(val).to_csv(metadata_dir / "val.csv", index=False)
        pd.DataFrame(test).to_csv(metadata_dir / "test.csv", index=False)
    elif has_images:
        print("🖼️  Detected IMAGE dataset...")
        records = process_images(data_dir, output_dir, args)
        train, val, test = create_splits(records, args.val_split, args.test_split, args.seed)
        pd.DataFrame(train).to_csv(metadata_dir / "train.csv", index=False)
        pd.DataFrame(val).to_csv(metadata_dir / "val.csv", index=False)
        pd.DataFrame(test).to_csv(metadata_dir / "test.csv", index=False)
    else:
        print("❌ No video or image files found. Check --data-dir")
        return

    print(f"\n✅ Preprocessing complete!")
    print(f"   Train: {len(train)} samples")
    print(f"   Val:   {len(val)} samples")
    print(f"   Test:  {len(test)} samples")
    print(f"   CSVs saved to: {metadata_dir}")


if __name__ == "__main__":
    main()
