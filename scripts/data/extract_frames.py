"""
DeepGuard — Frame Extractor
Trích xuất frames từ video với OpenCV, có progress bar và logging.

Usage:
    python scripts/data/extract_frames.py --input-dir data/raw --output-dir data/frames
    python scripts/data/extract_frames.py --input-dir data/raw --output-dir data/frames --fps 2 --max-frames 30
    python scripts/data/extract_frames.py --video data/raw/real/vid001.mp4 --output-dir data/frames/real --fps 1
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from loguru import logger
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/extract_frames.log", rotation="10 MB", level="DEBUG")

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class VideoExtractionResult:
    video_path: str
    label: str              # "real" or "fake"
    total_frames: int
    extracted_frames: int
    skipped_frames: int
    duration_sec: float
    fps_original: float
    fps_sample: float
    output_dir: str
    success: bool
    error_msg: str = ""
    elapsed_sec: float = 0.0


@dataclass
class ExtractionSummary:
    total_videos: int
    successful: int
    failed: int
    total_frames_extracted: int
    real_frames: int
    fake_frames: int
    elapsed_sec: float


# ── Core extraction logic ──────────────────────────────────────────────────────
def get_video_info(video_path: Path) -> dict:
    """Get video metadata without reading all frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    fps_sample: float = 1.0,
    max_frames: Optional[int] = None,
    image_quality: int = 95,
    skip_existing: bool = True,
) -> VideoExtractionResult:
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save extracted frames.
        fps_sample: Frames per second to extract (1.0 = 1 frame/sec).
        max_frames: Maximum frames to extract (None = unlimited).
        image_quality: JPEG quality (1-100).
        skip_existing: Skip if output_dir already has frames.

    Returns:
        VideoExtractionResult with extraction statistics.
    """
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine label from parent folder name
    label = "real" if "real" in video_path.parent.name.lower() else "fake"

    result = VideoExtractionResult(
        video_path=str(video_path),
        label=label,
        total_frames=0,
        extracted_frames=0,
        skipped_frames=0,
        duration_sec=0.0,
        fps_original=0.0,
        fps_sample=fps_sample,
        output_dir=str(output_dir),
        success=False,
    )

    # Skip if already extracted
    if skip_existing:
        existing = list(output_dir.glob("frame_*.jpg"))
        if existing:
            logger.debug(f"Skipping (already extracted {len(existing)} frames): {video_path.name}")
            result.extracted_frames = len(existing)
            result.success = True
            return result

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.error_msg = f"Cannot open video: {video_path}"
        logger.error(result.error_msg)
        return result

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        video_fps = 25.0  # Default assumption
        logger.warning(f"Invalid FPS for {video_path.name}, defaulting to 25")

    result.fps_original = video_fps
    result.total_frames = total_frames
    result.duration_sec = total_frames / video_fps

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps_sample))

    frame_idx = 0
    extracted = 0
    skipped = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Save frame
                frame_filename = output_dir / f"frame_{frame_idx:07d}.jpg"
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]

                if cv2.imwrite(str(frame_filename), frame, encode_params):
                    extracted += 1
                else:
                    skipped += 1
                    logger.warning(f"Failed to write frame {frame_idx} from {video_path.name}")

                if max_frames and extracted >= max_frames:
                    break

            frame_idx += 1

    except Exception as e:
        result.error_msg = str(e)
        logger.error(f"Error processing {video_path.name}: {e}")
    finally:
        cap.release()

    result.extracted_frames = extracted
    result.skipped_frames = skipped
    result.success = extracted > 0
    result.elapsed_sec = time.perf_counter() - t0

    logger.debug(
        f"{video_path.name}: {extracted} frames extracted "
        f"({video_fps:.1f}fps → sample {fps_sample}fps, interval={frame_interval}) "
        f"in {result.elapsed_sec:.1f}s"
    )
    return result


# ── Batch processor ────────────────────────────────────────────────────────────
def discover_videos(input_dir: Path) -> List[Tuple[Path, str]]:
    """
    Find all video files under input_dir/real/ and input_dir/fake/.
    Returns list of (video_path, label) tuples.
    """
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
    videos = []

    for label in ["real", "fake"]:
        label_dir = input_dir / label
        if not label_dir.exists():
            logger.warning(f"Directory not found: {label_dir}")
            continue

        found = [
            (f, label)
            for f in label_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        videos.extend(found)
        logger.info(f"  Found {len(found):,} {label} videos in {label_dir}")

    return videos


def extract_all_videos(
    input_dir: Path,
    output_dir: Path,
    fps_sample: float = 1.0,
    max_frames: Optional[int] = None,
    image_quality: int = 95,
    num_workers: int = 4,
    skip_existing: bool = True,
    save_log: bool = True,
) -> ExtractionSummary:
    """
    Extract frames from all videos in input_dir/real/ and input_dir/fake/.

    Args:
        input_dir: Root directory with real/ and fake/ subdirectories.
        output_dir: Output root for extracted frames.
        fps_sample: Target extraction FPS.
        max_frames: Max frames per video.
        image_quality: JPEG quality.
        num_workers: Parallel workers for extraction.
        skip_existing: Skip videos already processed.
        save_log: Save per-video stats to CSV.

    Returns:
        ExtractionSummary with overall stats.
    """
    t_start = time.perf_counter()

    videos = discover_videos(input_dir)
    if not videos:
        logger.error(f"No videos found in {input_dir}")
        return ExtractionSummary(0, 0, 0, 0, 0, 0, 0)

    logger.info(f"\nTotal videos to process: {len(videos):,}")
    logger.info(f"Extraction FPS: {fps_sample} | Max frames/video: {max_frames or 'unlimited'}")
    logger.info(f"Workers: {num_workers} | Skip existing: {skip_existing}")
    logger.info(f"Output: {output_dir.absolute()}\n")

    results: List[VideoExtractionResult] = []

    def process_video(args):
        video_path, label = args
        video_output_dir = output_dir / label / video_path.stem
        return extract_frames_from_video(
            video_path=video_path,
            output_dir=video_output_dir,
            fps_sample=fps_sample,
            max_frames=max_frames,
            image_quality=image_quality,
            skip_existing=skip_existing,
        )

    # Parallel execution with progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, v): v for v in videos}

        with tqdm(total=len(videos), desc="Extracting frames", unit="video", ncols=90) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "✓" if result.success else "✗"
                pbar.set_postfix({
                    "ok": sum(1 for r in results if r.success),
                    "fail": sum(1 for r in results if not r.success),
                    "frames": sum(r.extracted_frames for r in results),
                })
                pbar.update(1)

    # Compute summary
    elapsed = time.perf_counter() - t_start
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    summary = ExtractionSummary(
        total_videos=len(results),
        successful=len(successful),
        failed=len(failed),
        total_frames_extracted=sum(r.extracted_frames for r in results),
        real_frames=sum(r.extracted_frames for r in results if r.label == "real"),
        fake_frames=sum(r.extracted_frames for r in results if r.label == "fake"),
        elapsed_sec=elapsed,
    )

    # Log failed videos
    if failed:
        logger.warning(f"\n{len(failed)} videos failed:")
        for r in failed[:10]:
            logger.warning(f"  ✗ {Path(r.video_path).name}: {r.error_msg}")
        if len(failed) > 10:
            logger.warning(f"  ... and {len(failed) - 10} more (see log file)")

    # Save per-video CSV log
    if save_log:
        log_path = output_dir / "extraction_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                writer.writerows([asdict(r) for r in results])
        logger.info(f"Per-video log saved: {log_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total videos  : {summary.total_videos:,}")
    logger.info(f"  Successful    : {summary.successful:,}")
    logger.info(f"  Failed        : {summary.failed:,}")
    logger.info(f"  Total frames  : {summary.total_frames_extracted:,}")
    logger.info(f"    - Real      : {summary.real_frames:,}")
    logger.info(f"    - Fake      : {summary.fake_frames:,}")
    logger.info(f"  Elapsed       : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepGuard Frame Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract 1 frame/sec from all videos:
    python scripts/data/extract_frames.py --input-dir data/raw --output-dir data/frames

  Extract 3 frames/sec, max 50 frames per video:
    python scripts/data/extract_frames.py --input-dir data/raw --output-dir data/frames --fps 3 --max-frames 50

  Extract from a single video:
    python scripts/data/extract_frames.py --video data/raw/fake/vid001.mp4 --output-dir data/frames/fake/vid001

  Use 8 parallel workers:
    python scripts/data/extract_frames.py --input-dir data/raw --output-dir data/frames --workers 8
        """,
    )
    parser.add_argument("--input-dir", type=str, default="data/raw",
                        help="Root directory containing real/ and fake/ subdirs")
    parser.add_argument("--output-dir", type=str, default="data/frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--video", type=str, default=None,
                        help="Process a single video file instead of batch")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to extract per video")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG output quality 1-100 (default: 95)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-extract even if output folder already exists")
    parser.add_argument("--no-log", action="store_true",
                        help="Don't save per-video CSV log")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DeepGuard — Frame Extractor")
    logger.info("=" * 60)

    # Single video mode
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)

        output_dir = Path(args.output_dir)
        info = get_video_info(video_path)
        logger.info(f"Video: {video_path.name}")
        logger.info(f"  Resolution: {info.get('width')}x{info.get('height')}")
        logger.info(f"  FPS: {info.get('fps'):.2f} | Duration: {info.get('duration_sec', 0):.1f}s")
        logger.info(f"  Total frames: {info.get('total_frames'):,}")

        result = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            fps_sample=args.fps,
            max_frames=args.max_frames,
            image_quality=args.quality,
            skip_existing=not args.no_skip,
        )

        if result.success:
            logger.success(f"Extracted {result.extracted_frames:,} frames → {output_dir}")
        else:
            logger.error(f"Extraction failed: {result.error_msg}")
        return

    # Batch mode
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    summary = extract_all_videos(
        input_dir=input_dir,
        output_dir=output_dir,
        fps_sample=args.fps,
        max_frames=args.max_frames,
        image_quality=args.quality,
        num_workers=args.workers,
        skip_existing=not args.no_skip,
        save_log=not args.no_log,
    )

    if summary.failed > 0:
        sys.exit(1 if summary.successful == 0 else 0)


if __name__ == "__main__":
    main()
