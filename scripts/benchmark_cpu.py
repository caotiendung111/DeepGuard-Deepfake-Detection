"""
CPU benchmark for DeepGuard image/video inference.

Examples:
    python scripts/benchmark_cpu.py --real-dir data/bench/real --fake-dir data/bench/fake --videos-dir data/bench/videos
    python scripts/benchmark_cpu.py --images-dir data/external_test --videos-dir data/bench/videos --image-limit 100
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.face_detector import FaceDetector
from src.inference.model_loader import load_detector_checkpoint
from src.inference.predictor import predict_probabilities_batch
from src.inference.video_processor import InferenceVideoProcessor
from src.utils.config import load_config

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DeepGuard on CPU")
    parser.add_argument("--config", default=os.getenv("CONFIG_PATH", "configs/base.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--images-dir")
    parser.add_argument("--real-dir")
    parser.add_argument("--fake-dir")
    parser.add_argument("--videos-dir")
    parser.add_argument("--image-limit", type=int, default=100)
    parser.add_argument("--video-limit", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--backends", nargs="+", default=["insightface", "mtcnn", "haar"])
    parser.add_argument("--tta-modes", nargs="+", default=["false", "adaptive", "true"])
    parser.add_argument("--output", default="reports/benchmark/cpu_benchmark.json")
    parser.add_argument("--markdown-output", default="reports/benchmark/cpu_benchmark.md")
    parser.add_argument("--fallback-full-frame", action="store_true", default=True)
    return parser.parse_args()


def files_under(root: str | None, extensions: set[str], limit: int | None = None) -> List[Path]:
    if not root:
        return []
    path = Path(root)
    if not path.exists():
        return []
    files = sorted(fp for fp in path.rglob("*") if fp.is_file() and fp.suffix.lower() in extensions)
    return files[:limit] if limit else files


def collect_images(args) -> List[dict]:
    if args.real_dir or args.fake_dir:
        real = [{"path": path, "label": "real"} for path in files_under(args.real_dir, IMAGE_EXTENSIONS, 50)]
        fake = [{"path": path, "label": "fake"} for path in files_under(args.fake_dir, IMAGE_EXTENSIONS, 50)]
        return (real + fake)[:args.image_limit]
    return [
        {"path": path, "label": "unknown"}
        for path in files_under(args.images_dir, IMAGE_EXTENSIONS, args.image_limit)
    ]


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def summarize_ms(values: Iterable[float]) -> dict:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def tta_value(mode: str):
    mode = mode.lower().strip()
    if mode in {"adaptive", "auto"}:
        return "adaptive"
    return mode in {"1", "true", "yes", "on"}


class ResourceMonitor:
    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        if psutil is not None:
            psutil.cpu_percent(interval=None)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self):
        process = psutil.Process(os.getpid()) if psutil is not None else None
        while not self._stop.is_set():
            sample = {
                "ts": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_percent": psutil.virtual_memory().percent,
            }
            if process is not None:
                rss = process.memory_info().rss
                sample["process_rss_mb"] = rss / 1024 / 1024
            self.samples.append(sample)
            self._stop.wait(self.interval_s)


def summarize_resources(samples: list[dict]) -> dict:
    if not samples:
        return {
            "peak_cpu_percent": None,
            "mean_cpu_percent": None,
            "peak_ram_percent": None,
            "peak_process_rss_mb": None,
        }
    cpu_values = [sample.get("cpu_percent", 0.0) for sample in samples]
    ram_values = [sample.get("ram_percent", 0.0) for sample in samples]
    rss_values = [sample.get("process_rss_mb", 0.0) for sample in samples]
    return {
        "peak_cpu_percent": max(cpu_values),
        "mean_cpu_percent": statistics.mean(cpu_values),
        "peak_ram_percent": max(ram_values),
        "peak_process_rss_mb": max(rss_values),
    }


def benchmark_images(model, device, cfg, detector, images: List[dict], args) -> dict:
    crops = []
    detection_rows = []
    for item in images:
        image_rgb = load_rgb(item["path"])
        started = time.perf_counter()
        crop = detector.detect_and_crop(image_rgb)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if crop is None:
            crop = cv2.resize(image_rgb, (cfg.image_size, cfg.image_size))
        crops.append(crop)
        detection_rows.append({
            "file": str(item["path"]),
            "label": item["label"],
            "face_detection_ms": elapsed_ms,
        })

    tta_results = {}
    for mode in args.tta_modes:
        started = time.perf_counter()
        probs, per_image_probs = predict_probabilities_batch(
            model=model,
            images_rgb=crops,
            image_size=cfg.image_size,
            device=device,
            use_tta=tta_value(mode),
            batch_size=args.batch_size,
            use_amp=False,
        )
        elapsed_s = time.perf_counter() - started
        count = max(1, len(crops))
        tta_results[mode] = {
            "count": len(crops),
            "total_s": elapsed_s,
            "latency_ms": (elapsed_s * 1000) / count,
            "throughput_images_per_s": len(crops) / elapsed_s if elapsed_s > 0 else 0.0,
            "mean_probability_fake": float(np.mean(probs)) if probs else None,
            "mean_tta_variants": float(np.mean([len(x) for x in per_image_probs])) if per_image_probs else 0.0,
        }

    return {
        "face_detection": summarize_ms(row["face_detection_ms"] for row in detection_rows),
        "tta": tta_results,
        "rows": detection_rows,
    }


def benchmark_videos(model, device, cfg, detector, videos: List[Path], args) -> dict:
    rows = []
    processor = InferenceVideoProcessor(
        face_detector=detector,
        image_size=cfg.image_size,
        face_batch_size=args.batch_size,
    )
    for video_path in videos:
        started = time.perf_counter()
        frame_probs = []
        frames_analyzed = 0
        chunks = 0
        for chunk in processor.iter_face_crops(
            str(video_path),
            n_frames=args.max_frames,
            chunk_size=args.chunk_size,
            fallback_full_frame=args.fallback_full_frame,
            use_box_cache=True,
            max_cache_gap=getattr(cfg, "face_cache_gap", 5),
        ):
            chunks += 1
            faces = [face_rgb for _, face_rgb in chunk]
            probs, _ = predict_probabilities_batch(
                model=model,
                images_rgb=faces,
                image_size=cfg.image_size,
                device=device,
                use_tta=False,
                batch_size=args.batch_size,
                use_amp=False,
            )
            frames_analyzed += len(faces)
            frame_probs.extend(probs)

        elapsed_s = time.perf_counter() - started
        rows.append({
            "file": str(video_path),
            "processing_s": elapsed_s,
            "frames_analyzed": frames_analyzed,
            "chunks": chunks,
            "fps_effective": frames_analyzed / elapsed_s if elapsed_s > 0 else 0.0,
            "mean_probability_fake": float(np.mean(frame_probs)) if frame_probs else None,
        })

    return {
        "summary": {
            **summarize_ms(row["processing_s"] * 1000 for row in rows),
            "total_frames_analyzed": sum(row["frames_analyzed"] for row in rows),
            "mean_effective_fps": statistics.mean([row["fps_effective"] for row in rows]) if rows else 0.0,
        },
        "rows": rows,
    }


def markdown_report(results: dict) -> str:
    lines = [
        "# DeepGuard CPU Benchmark",
        "",
        f"- device: `{results['device']}`",
        f"- torch_threads: `{results['torch_threads']}`",
        f"- image_size: `{results['image_size']}`",
        f"- image_count: `{results['image_count']}`",
        f"- video_count: `{results['video_count']}`",
        f"- peak_cpu_percent: `{results['resources']['peak_cpu_percent']}`",
        f"- peak_ram_percent: `{results['resources']['peak_ram_percent']}`",
        "",
        "## Image Face Detection",
        "| requested_backend | active_backend | count | mean_ms | p95_ms |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for backend, data in results["backends"].items():
        summary = data["images"]["face_detection"]
        lines.append(
            f"| {backend} | {data['active_backend']} | {summary['count']} | "
            f"{summary['mean_ms']:.2f} | {summary['p95_ms']:.2f} |"
        )

    lines.extend([
        "",
        "## Image Model Throughput",
        "| backend | tta | latency_ms_per_image | images_per_s | mean_tta_variants |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for backend, data in results["backends"].items():
        for mode, summary in data["images"]["tta"].items():
            lines.append(
                f"| {backend} | {mode} | {summary['latency_ms']:.2f} | "
                f"{summary['throughput_images_per_s']:.2f} | {summary['mean_tta_variants']:.1f} |"
            )

    lines.extend([
        "",
        "## Video Processing",
        "| backend | count | mean_s | p95_s | total_frames | effective_fps | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for backend, data in results["backends"].items():
        summary = data["videos"]["summary"]
        mean_s = summary["mean_ms"] / 1000
        status = "ok" if mean_s <= 30 or summary["count"] == 0 else "slow"
        lines.append(
            f"| {backend} | {summary['count']} | {mean_s:.2f} | "
            f"{summary['p95_ms'] / 1000:.2f} | {summary['total_frames_analyzed']} | "
            f"{summary['mean_effective_fps']:.2f} | {status} |"
        )

    lines.extend([
        "",
        "## Acceptance Check",
        "| backend | tta | image_latency_ms | image_status |",
        "| --- | --- | ---: | --- |",
    ])
    for backend, data in results["backends"].items():
        for mode, summary in data["images"]["tta"].items():
            status = "ok" if summary["latency_ms"] < 1000 else "slow"
            lines.append(f"| {backend} | {mode} | {summary['latency_ms']:.2f} | {status} |")

    lines.extend([
        "",
        "## CPU Notes",
        "- Prefer `insightface` first. If its model package is not cached, first startup may download the ONNX package.",
        "- `mtcnn` is usually the slowest CPU backend; keep it for accuracy checks, not high-throughput API serving.",
        "- `haar` is fastest but less accurate and should be treated as a load-shedding fallback.",
        "- CPU TTA uses original + horizontal flip only; `adaptive` runs TTA only for uncertain samples.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
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
    if args.image_size:
        cfg.image_size = args.image_size

    images = collect_images(args)
    videos = files_under(args.videos_dir, VIDEO_EXTENSIONS, args.video_limit)
    if not images:
        raise SystemExit("No benchmark images found. Use --images-dir or --real-dir/--fake-dir.")

    checkpoint = args.checkpoint or cfg.checkpoint_path
    device = torch.device("cpu")
    model, metadata = load_detector_checkpoint(checkpoint, cfg, device)
    model.eval()

    results = {
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "image_size": cfg.image_size,
        "batch_size": args.batch_size,
        "max_frames": args.max_frames,
        "chunk_size": args.chunk_size,
        "checkpoint": checkpoint,
        "model_metadata": metadata,
        "image_count": len(images),
        "video_count": len(videos),
        "resources": {},
        "resource_samples": [],
        "backends": {},
    }

    with ResourceMonitor() as monitor:
        for backend in args.backends:
            detector = FaceDetector(
                backend=backend,
                device="cpu",
                face_size=cfg.image_size,
            )
            results["backends"][backend] = {
                "active_backend": detector.active_backend,
                "images": benchmark_images(model, device, cfg, detector, images, args),
                "videos": benchmark_videos(model, device, cfg, detector, videos, args) if videos else {
                    "summary": {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "total_frames_analyzed": 0, "mean_effective_fps": 0.0},
                    "rows": [],
                },
            }
        results["resource_samples"] = monitor.samples
        results["resources"] = summarize_resources(monitor.samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_path = Path(args.markdown_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_report(results), encoding="utf-8")

    print(f"Wrote JSON benchmark to {output_path}")
    print(f"Wrote Markdown benchmark to {md_path}")


if __name__ == "__main__":
    main()
