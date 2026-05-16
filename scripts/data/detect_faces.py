"""
DeepGuard — Face Detector & Cropper
Phát hiện và crop khuôn mặt từ frames đã extract.
Hỗ trợ: MTCNN (mặc định), RetinaFace (nếu cài).

Usage:
    python scripts/data/detect_faces.py --input-dir data/frames --output-dir data/faces
    python scripts/data/detect_faces.py --input-dir data/frames --output-dir data/faces --size 299 --backend retinaface
    python scripts/data/detect_faces.py --image data/frames/real/vid001/frame_0000001.jpg --output-dir /tmp/test
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")
logger.add("logs/detect_faces.log", rotation="10 MB", level="DEBUG")

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class FaceExtractionStats:
    source_dir: str
    label: str
    total_images: int
    faces_found: int
    faces_skipped: int
    no_face_count: int
    success: bool
    elapsed_sec: float = 0.0


# ── Face Detector backends ─────────────────────────────────────────────────────
class MTCNNDetector:
    """MTCNN face detector using facenet-pytorch."""

    def __init__(self, device: str = "auto", min_face_size: int = 20):
        import torch
        from facenet_pytorch import MTCNN
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.detector = MTCNN(
            keep_all=True,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            device=self.device,
            post_process=False,
        )
        logger.info(f"MTCNN initialized on {self.device}")

    def detect(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) bounding boxes."""
        from PIL import Image
        pil_img = Image.fromarray(image_rgb)
        boxes, probs = self.detector.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            return []
        # Filter low-confidence detections
        return [
            (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            for b, p in zip(boxes, probs)
            if p is not None and p >= 0.9
        ]


class RetinaFaceDetector:
    """RetinaFace detector (requires retinaface package)."""

    def __init__(self):
        try:
            from retinaface import RetinaFace
            self._detect_fn = RetinaFace.detect_faces
            logger.info("RetinaFace initialized")
        except ImportError:
            raise ImportError("Install RetinaFace: pip install retina-face")

    def detect(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self._detect_fn(image_bgr)
        boxes = []
        if isinstance(faces, dict):
            for face_info in faces.values():
                area = face_info.get("facial_area", [])
                if len(area) == 4:
                    x1, y1, x2, y2 = area
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes


class HaarCascadeDetector:
    """OpenCV Haar Cascade fallback detector (no GPU, less accurate)."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)
        logger.info("Haar Cascade detector initialized (fallback mode)")

    def detect(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return []
        return [(x, y, x + w, y + h) for x, y, w, h in faces]


def build_detector(backend: str = "mtcnn", device: str = "auto"):
    """Factory function to build the requested detector."""
    backend = backend.lower()
    if backend == "mtcnn":
        try:
            return MTCNNDetector(device=device)
        except ImportError:
            logger.warning("facenet-pytorch not installed. Falling back to Haar Cascade.")
            return HaarCascadeDetector()
    elif backend == "retinaface":
        try:
            return RetinaFaceDetector()
        except ImportError:
            logger.warning("RetinaFace not installed. Falling back to MTCNN.")
            return build_detector("mtcnn", device)
    elif backend == "haar":
        return HaarCascadeDetector()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use: mtcnn | retinaface | haar")


# ── Face crop utility ──────────────────────────────────────────────────────────
def crop_face(
    image_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    output_size: int = 224,
    padding: float = 0.2,
    square: bool = True,
) -> Optional[np.ndarray]:
    """
    Crop face from image with padding and resize.

    Args:
        image_rgb: Input RGB image.
        box: Bounding box (x1, y1, x2, y2).
        output_size: Output image size (square).
        padding: Fraction of face size to add as padding.
        square: Force square crop.

    Returns:
        Cropped & resized face as RGB numpy array, or None if invalid.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box

    face_w = x2 - x1
    face_h = y2 - y1

    if face_w <= 0 or face_h <= 0:
        return None

    # Add padding
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    # Square crop (use max side)
    if square:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = max(x2 - x1, y2 - y1) // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

    face_crop = image_rgb[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)


def select_largest_face(
    boxes: List[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, int, int, int]]:
    """Return the largest face bounding box by area."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


# ── Per-image processing ───────────────────────────────────────────────────────
def process_image(
    image_path: Path,
    output_path: Path,
    detector,
    output_size: int = 224,
    padding: float = 0.2,
    select_mode: str = "largest",  # "largest" | "all"
    jpeg_quality: int = 95,
) -> Tuple[bool, str]:
    """
    Detect face in one image and save cropped result.

    Returns:
        (success, message)
    """
    # Read image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return False, f"Cannot read image: {image_path.name}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect faces
    try:
        boxes = detector.detect(img_rgb)
    except Exception as e:
        return False, f"Detection error: {e}"

    if not boxes:
        return False, "no_face"

    # Select face(s)
    if select_mode == "largest":
        selected_boxes = [select_largest_face(boxes)]
    else:
        selected_boxes = boxes

    saved = 0
    for i, box in enumerate(selected_boxes):
        if box is None:
            continue

        face = crop_face(img_rgb, box, output_size=output_size, padding=padding)
        if face is None:
            continue

        # Output path
        if select_mode == "all" and len(selected_boxes) > 1:
            out_path = output_path.parent / f"{output_path.stem}_face{i}{output_path.suffix}"
        else:
            out_path = output_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        saved += 1

    return (saved > 0), f"saved {saved} face(s)"


# ── Batch processing ───────────────────────────────────────────────────────────
def process_video_frames(
    frames_dir: Path,
    output_dir: Path,
    detector,
    label: str,
    output_size: int = 224,
    padding: float = 0.2,
    jpeg_quality: int = 95,
    skip_existing: bool = True,
) -> FaceExtractionStats:
    """Process all frames in a single video's frame directory."""
    t0 = time.perf_counter()

    image_files = sorted(
        [f for f in frames_dir.glob("*.jpg")] +
        [f for f in frames_dir.glob("*.png")]
    )

    stats = FaceExtractionStats(
        source_dir=str(frames_dir),
        label=label,
        total_images=len(image_files),
        faces_found=0,
        faces_skipped=0,
        no_face_count=0,
        success=True,
    )

    if not image_files:
        stats.success = False
        return stats

    video_output_dir = output_dir / label / frames_dir.name

    for img_path in image_files:
        out_path = video_output_dir / img_path.name

        if skip_existing and out_path.exists():
            stats.faces_skipped += 1
            continue

        success, msg = process_image(
            image_path=img_path,
            output_path=out_path,
            detector=detector,
            output_size=output_size,
            padding=padding,
            jpeg_quality=jpeg_quality,
        )

        if success:
            stats.faces_found += 1
        elif msg == "no_face":
            stats.no_face_count += 1
        else:
            stats.faces_skipped += 1
            logger.debug(f"Skipped {img_path.name}: {msg}")

    stats.elapsed_sec = time.perf_counter() - t0
    return stats


def process_all_frames(
    frames_dir: Path,
    output_dir: Path,
    backend: str = "mtcnn",
    device: str = "auto",
    output_size: int = 224,
    padding: float = 0.2,
    jpeg_quality: int = 95,
    num_workers: int = 2,
    skip_existing: bool = True,
    save_log: bool = True,
) -> dict:
    """
    Process all extracted frames from data/frames/real/ and data/frames/fake/.
    """
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DeepGuard — Face Detection & Cropping")
    logger.info("=" * 60)
    logger.info(f"Backend     : {backend.upper()}")
    logger.info(f"Device      : {device}")
    logger.info(f"Output size : {output_size}×{output_size}")
    logger.info(f"Padding     : {padding:.0%}")
    logger.info(f"Workers     : {num_workers}")
    logger.info(f"Input dir   : {frames_dir.absolute()}")
    logger.info(f"Output dir  : {output_dir.absolute()}")

    # Build detector (single instance, shared across threads)
    # Note: MTCNN is not thread-safe → use 1 worker or build per-thread
    detector = build_detector(backend, device)

    # Discover video frame directories
    video_dirs = []
    for label in ["real", "fake"]:
        label_dir = frames_dir / label
        if not label_dir.exists():
            logger.warning(f"Directory not found: {label_dir}")
            continue
        for video_dir in sorted(label_dir.iterdir()):
            if video_dir.is_dir():
                video_dirs.append((video_dir, label))

    if not video_dirs:
        # Maybe frames are directly in real/ fake/ without video subdirs
        for label in ["real", "fake"]:
            label_dir = frames_dir / label
            if label_dir.exists() and any(label_dir.glob("*.jpg")):
                video_dirs.append((label_dir, label))

    if not video_dirs:
        logger.error(f"No frame directories found under {frames_dir}")
        return {}

    logger.info(f"Found {len(video_dirs)} video directories to process\n")

    all_stats: List[FaceExtractionStats] = []

    # Sequential (MTCNN not thread-safe) or parallel (Haar/RetinaFace)
    use_parallel = backend in ("haar",) and num_workers > 1

    if use_parallel:
        def process_one(args):
            vdir, lbl = args
            det = build_detector(backend, device)
            return process_video_frames(vdir, output_dir, det, lbl, output_size, padding, jpeg_quality, skip_existing)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_one, v): v for v in video_dirs}
            with tqdm(total=len(video_dirs), desc="Processing videos", unit="video", ncols=90) as pbar:
                for future in as_completed(futures):
                    stats = future.result()
                    all_stats.append(stats)
                    pbar.set_postfix({
                        "faces": sum(s.faces_found for s in all_stats),
                        "no_face": sum(s.no_face_count for s in all_stats),
                    })
                    pbar.update(1)
    else:
        # Sequential processing
        for video_dir, label in tqdm(video_dirs, desc="Processing videos", unit="video", ncols=90):
            stats = process_video_frames(
                video_dir, output_dir, detector, label,
                output_size, padding, jpeg_quality, skip_existing
            )
            all_stats.append(stats)

    # Summary
    elapsed = time.perf_counter() - t_start
    total_faces = sum(s.faces_found for s in all_stats)
    total_no_face = sum(s.no_face_count for s in all_stats)
    total_skipped = sum(s.faces_skipped for s in all_stats)
    real_faces = sum(s.faces_found for s in all_stats if s.label == "real")
    fake_faces = sum(s.faces_found for s in all_stats if s.label == "fake")

    logger.info("\n" + "=" * 60)
    logger.info("FACE DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Videos processed : {len(all_stats):,}")
    logger.info(f"  Faces saved      : {total_faces:,}")
    logger.info(f"    - Real         : {real_faces:,}")
    logger.info(f"    - Fake         : {fake_faces:,}")
    logger.info(f"  No face found    : {total_no_face:,} frames (skipped)")
    logger.info(f"  Already existed  : {total_skipped:,} frames")
    logger.info(f"  Elapsed          : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if total_faces + total_no_face > 0:
        rate = total_faces / (total_faces + total_no_face) * 100
        logger.info(f"  Face detect rate : {rate:.1f}%")
    logger.info("=" * 60)

    # Save log
    if save_log and all_stats:
        log_path = output_dir / "face_detection_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(all_stats[0]).keys())
            writer.writeheader()
            writer.writerows([asdict(s) for s in all_stats])
        logger.info(f"Log saved: {log_path}")

    return {
        "total_faces": total_faces,
        "real_faces": real_faces,
        "fake_faces": fake_faces,
        "no_face_frames": total_no_face,
        "elapsed_sec": elapsed,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard Face Detector")
    parser.add_argument("--input-dir", type=str, default="data/frames",
                        help="Root dir with real/ and fake/ frame subdirectories")
    parser.add_argument("--output-dir", type=str, default="data/faces",
                        help="Output directory for cropped faces")
    parser.add_argument("--image", type=str, default=None,
                        help="Process a single image file")
    parser.add_argument("--backend", choices=["mtcnn", "retinaface", "haar"],
                        default="mtcnn", help="Face detection backend")
    parser.add_argument("--device", default="auto", help="Device: auto | cpu | cuda")
    parser.add_argument("--size", type=int, default=224,
                        help="Output face crop size in pixels (default: 224)")
    parser.add_argument("--padding", type=float, default=0.2,
                        help="Padding around face (default: 0.2 = 20%%)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG output quality (default: 95)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers (MTCNN: use 1)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-process existing output files")
    parser.add_argument("--no-log", action="store_true",
                        help="Don't save CSV log")
    return parser.parse_args()


def main():
    args = parse_args()

    # Single image mode
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)

        detector = build_detector(args.backend, args.device)
        output_path = Path(args.output_dir) / image_path.name
        success, msg = process_image(
            image_path=image_path,
            output_path=output_path,
            detector=detector,
            output_size=args.size,
            padding=args.padding,
            jpeg_quality=args.quality,
        )

        if success:
            logger.success(f"Face saved: {output_path}")
        else:
            logger.warning(f"No face saved: {msg}")
        return

    # Batch mode
    frames_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not frames_dir.exists():
        logger.error(f"Input directory not found: {frames_dir}")
        sys.exit(1)

    process_all_frames(
        frames_dir=frames_dir,
        output_dir=output_dir,
        backend=args.backend,
        device=args.device,
        output_size=args.size,
        padding=args.padding,
        jpeg_quality=args.quality,
        num_workers=args.workers,
        skip_existing=not args.no_skip,
        save_log=not args.no_log,
    )


if __name__ == "__main__":
    main()
