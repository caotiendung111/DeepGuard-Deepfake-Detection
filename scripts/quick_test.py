"""
Quick CPU smoke test for DeepGuard.

Examples:
    python scripts/quick_test.py --image test_samples/real_human.jpg
    python scripts/quick_test.py --video data/bench/videos/sample.mp4 --max-frames 16
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.face_detector import FaceDetector
from src.inference.model_loader import load_detector_checkpoint
from src.inference.predictor import predict_probability, predict_probabilities_batch
from src.inference.video_processor import InferenceVideoProcessor
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run a quick DeepGuard image/video CPU test")
    parser.add_argument("--config", default=os.getenv("CONFIG_PATH", "configs/base.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--backend", default=None, choices=["insightface", "mtcnn", "haar", "auto", None])
    parser.add_argument("--tta", default=None, choices=["false", "true", "adaptive", None])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    return parser.parse_args()


def tta_value(value, default):
    if value is None:
        return default
    if value == "adaptive":
        return "adaptive"
    return value == "true"


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def print_result(label: str, prob: float, threshold: float, elapsed_s: float, extra: str = "") -> None:
    is_fake = prob >= threshold
    print(f"{label}: {'FAKE' if is_fake else 'REAL'}")
    print(f"probability_fake: {prob:.4f}")
    print(f"probability_real: {1.0 - prob:.4f}")
    print(f"confidence: {max(prob, 1.0 - prob):.4f}")
    print(f"threshold: {threshold:.4f}")
    print(f"elapsed_s: {elapsed_s:.3f}")
    if extra:
        print(extra)


def main():
    args = parse_args()
    if not args.image and not args.video:
        raise SystemExit("Provide --image or --video")

    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(max(1, min(2, args.threads)))
    except RuntimeError:
        pass

    cfg = load_config(args.config)
    cfg.device = "cpu"
    cfg.inference_amp = False
    cfg.inference_batch_size = args.batch_size
    cfg.video_chunk_size = args.chunk_size

    checkpoint = args.checkpoint or cfg.checkpoint_path
    threshold = cfg.threshold if args.threshold is None else args.threshold
    backend = args.backend or cfg.face_detector_backend
    use_tta = tta_value(args.tta, cfg.inference_tta)

    device = torch.device("cpu")
    model, _ = load_detector_checkpoint(checkpoint, cfg, device)
    model.eval()

    detector = FaceDetector(
        backend=backend,
        device="cpu",
        face_size=cfg.image_size,
    )
    print(f"device: cpu")
    print(f"image_size: {cfg.image_size}")
    print(f"face_backend: {backend} -> {detector.active_backend}")
    print(f"tta: {use_tta}")

    if args.image:
        started = time.perf_counter()
        image_rgb = load_rgb(args.image)
        face_rgb = detector.detect_and_crop(image_rgb)
        if face_rgb is None:
            face_rgb = cv2.resize(image_rgb, (cfg.image_size, cfg.image_size))
        prob, tta_probs = predict_probability(
            model=model,
            image_rgb=face_rgb,
            image_size=cfg.image_size,
            device=device,
            use_tta=use_tta,
            batch_size=args.batch_size,
            use_amp=False,
        )
        elapsed_s = time.perf_counter() - started
        print_result("image", prob, threshold, elapsed_s, f"tta_variants: {len(tta_probs)}")

    if args.video:
        started = time.perf_counter()
        processor = InferenceVideoProcessor(
            face_detector=detector,
            image_size=cfg.image_size,
            face_batch_size=args.batch_size,
        )
        probs = []
        frames = 0
        chunks = 0
        for chunk in processor.iter_face_crops(
            args.video,
            n_frames=args.max_frames,
            chunk_size=args.chunk_size,
            fallback_full_frame=True,
            use_box_cache=True,
            max_cache_gap=cfg.face_cache_gap,
        ):
            chunks += 1
            faces = [face_rgb for _, face_rgb in chunk]
            chunk_probs, _ = predict_probabilities_batch(
                model=model,
                images_rgb=faces,
                image_size=cfg.image_size,
                device=device,
                use_tta=False,
                batch_size=args.batch_size,
                use_amp=False,
            )
            frames += len(faces)
            probs.extend(chunk_probs)

        elapsed_s = time.perf_counter() - started
        if not probs:
            raise SystemExit("No frames/faces were processed from the video")
        prob = float(np.mean(probs))
        extra = f"frames_analyzed: {frames}\nchunks: {chunks}\neffective_fps: {frames / elapsed_s:.2f}"
        print_result("video", prob, threshold, elapsed_s, extra)


if __name__ == "__main__":
    main()
