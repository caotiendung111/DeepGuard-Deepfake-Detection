"""
DeepGuard API benchmark for image/video inference.

Recommended RTX 3090 run:
    python scripts/benchmark.py \
      --images-dir data/bench/images \
      --videos-dir data/bench/videos \
      --image-limit 200 \
      --video-limit 20 \
      --concurrency 4
"""
import argparse
import concurrent.futures
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Iterable, List

import requests

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DeepGuard API performance")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--videos-dir")
    parser.add_argument("--output", default="reports/benchmark/benchmark.json")
    parser.add_argument("--markdown-output", default="reports/benchmark/benchmark.md")
    parser.add_argument("--plots-dir", default="reports/benchmark/plots")
    parser.add_argument("--image-limit", type=int, default=200)
    parser.add_argument("--video-limit", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--image-modes", nargs="+", default=["false", "true", "adaptive"])
    parser.add_argument("--video-frames", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--video-tta", default="adaptive", choices=["false", "true", "adaptive"])
    return parser.parse_args()


def files_under(root: str, extensions: set[str], limit: int) -> List[Path]:
    path = Path(root)
    if not path.exists():
        return []
    files = sorted(
        fp for fp in path.rglob("*")
        if fp.is_file() and fp.suffix.lower() in extensions
    )
    return files[:limit]


def gpu_snapshot() -> dict:
    if torch is None or not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "gpu_free_mb": free_bytes / 1024 / 1024,
        "gpu_used_mb": (total_bytes - free_bytes) / 1024 / 1024,
        "gpu_total_mb": total_bytes / 1024 / 1024,
    }


def system_snapshot() -> dict:
    snapshot = {}
    if psutil is not None:
        disk = psutil.disk_io_counters()
        snapshot.update({
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": psutil.virtual_memory().percent,
            "disk_read_mb": disk.read_bytes / 1024 / 1024 if disk else None,
            "disk_write_mb": disk.write_bytes / 1024 / 1024 if disk else None,
        })
    snapshot.update(gpu_snapshot())
    return snapshot


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            sample = {"ts": time.time()}
            sample.update(system_snapshot())
            self.samples.append(sample)
            self._stop.wait(self.interval)


def post_image(api_url: str, image_path: Path, use_tta: str, timeout: float) -> dict:
    started = time.perf_counter()
    with image_path.open("rb") as f:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict/image",
            data={"use_tta": use_tta, "return_heatmap": "false"},
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=timeout,
        )
    elapsed = time.perf_counter() - started
    payload = {}
    try:
        payload = response.json()
    except Exception:
        pass
    return {
        "file": str(image_path),
        "mode": use_tta,
        "status_code": response.status_code,
        "ok": response.ok,
        "latency_s": elapsed,
        "api_processing_ms": payload.get("processing_time_ms"),
        "probability_fake": payload.get("probability_fake"),
        "label": payload.get("label"),
        "error": payload.get("detail") if not response.ok else None,
    }


def post_video(api_url: str, video_path: Path, max_frames: int, use_tta: str, timeout: float) -> dict:
    started = time.perf_counter()
    with video_path.open("rb") as f:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict/video",
            data={
                "max_frames": str(max_frames),
                "timeout_seconds": str(int(timeout)),
                "use_tta": use_tta,
            },
            files={"file": (video_path.name, f, "video/mp4")},
            timeout=timeout,
        )
    if not response.ok:
        return {
            "file": str(video_path),
            "max_frames": max_frames,
            "status": "submit_failed",
            "status_code": response.status_code,
            "latency_s": time.perf_counter() - started,
            "error": response.text,
        }

    job_id = response.json()["job_id"]
    while True:
        status_response = requests.get(
            f"{api_url.rstrip('/')}/predict/video/{job_id}",
            timeout=timeout,
        )
        status_response.raise_for_status()
        payload = status_response.json()
        if payload["status"] in {"done", "failed", "cancelled"}:
            break
        if time.perf_counter() - started > timeout:
            raise TimeoutError(f"Video job {job_id} timed out")
        time.sleep(0.5)

    elapsed = time.perf_counter() - started
    return {
        "file": str(video_path),
        "job_id": job_id,
        "max_frames": max_frames,
        "use_tta": use_tta,
        "status": payload["status"],
        "latency_s": elapsed,
        "api_processing_ms": payload.get("processing_time_ms"),
        "frames_analyzed": payload.get("frames_analyzed") or payload.get("n_frames_analyzed"),
        "probability_fake": payload.get("probability_fake"),
        "label": payload.get("label"),
        "error": payload.get("error"),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def summarize_latencies(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    latencies = [row["latency_s"] for row in rows if row.get("ok", True) and row.get("latency_s") is not None]
    if not latencies:
        return {"count": 0, "errors": sum(1 for row in rows if not row.get("ok", True))}
    return {
        "count": len(latencies),
        "errors": sum(1 for row in rows if not row.get("ok", True) or row.get("status") == "failed"),
        "mean_s": statistics.mean(latencies),
        "p50_s": percentile(latencies, 0.50),
        "p95_s": percentile(latencies, 0.95),
        "p99_s": percentile(latencies, 0.99),
        "min_s": min(latencies),
        "max_s": max(latencies),
    }


def run_concurrent_images(api_url: str, images: List[Path], mode: str, concurrency: int, timeout: float) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(post_image, api_url, image_path, mode, timeout)
            for image_path in images
        ]
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed = time.perf_counter() - started
    return rows, {
        "mode": mode,
        "concurrency": concurrency,
        "requests": len(rows),
        "total_s": elapsed,
        "requests_per_second": len(rows) / elapsed if elapsed > 0 else 0,
        "latency_summary": summarize_latencies(rows),
    }


def make_plots(results: dict, plots_dir: Path) -> None:
    if plt is None:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)

    image_modes = list(results["image_summary"].keys())
    image_means = [results["image_summary"][mode].get("mean_s", 0) for mode in image_modes]
    image_p95 = [results["image_summary"][mode].get("p95_s", 0) for mode in image_modes]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(image_modes))
    ax.bar([i - 0.2 for i in x], image_means, width=0.4, label="mean")
    ax.bar([i + 0.2 for i in x], image_p95, width=0.4, label="p95")
    ax.set_xticks(list(x), image_modes)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Image Latency by TTA Mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "image_latency.png", dpi=150)
    plt.close(fig)

    video_keys = list(results["video_summary"].keys())
    video_means = [results["video_summary"][key].get("mean_s", 0) for key in video_keys]
    video_p95 = [results["video_summary"][key].get("p95_s", 0) for key in video_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(video_keys))
    ax.plot(list(x), video_means, marker="o", label="mean")
    ax.plot(list(x), video_p95, marker="o", label="p95")
    ax.set_xticks(list(x), video_keys)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Video Latency by max_frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "video_latency.png", dpi=150)
    plt.close(fig)

    samples = results.get("resource_samples", [])
    if samples:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        t0 = samples[0]["ts"]
        xs = [sample["ts"] - t0 for sample in samples]
        axes[0].plot(xs, [sample.get("cpu_percent", 0) for sample in samples])
        axes[0].set_ylabel("CPU %")
        axes[1].plot(xs, [sample.get("ram_percent", 0) for sample in samples])
        axes[1].set_ylabel("RAM %")
        axes[2].plot(xs, [sample.get("gpu_used_mb", 0) for sample in samples])
        axes[2].set_ylabel("GPU MB")
        axes[2].set_xlabel("Seconds")
        fig.tight_layout()
        fig.savefig(plots_dir / "resources.png", dpi=150)
        plt.close(fig)


def markdown_table(results: dict) -> str:
    lines = ["# DeepGuard Benchmark", ""]
    lines.append("## Image Latency")
    lines.append("| mode | count | errors | mean_s | p95_s | p99_s | rps |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode, summary in results["image_summary"].items():
        throughput = results["image_throughput"].get(mode, {})
        lines.append(
            f"| {mode} | {summary.get('count', 0)} | {summary.get('errors', 0)} | "
            f"{summary.get('mean_s', 0):.4f} | {summary.get('p95_s', 0):.4f} | "
            f"{summary.get('p99_s', 0):.4f} | {throughput.get('requests_per_second', 0):.2f} |"
        )

    lines.extend(["", "## Video Latency", "| max_frames | count | errors | mean_s | p95_s | p99_s |"])
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for frames, summary in results["video_summary"].items():
        lines.append(
            f"| {frames} | {summary.get('count', 0)} | {summary.get('errors', 0)} | "
            f"{summary.get('mean_s', 0):.4f} | {summary.get('p95_s', 0):.4f} | "
            f"{summary.get('p99_s', 0):.4f} |"
        )

    lines.extend([
        "",
        "## Outputs",
        f"- plots: `{results['plots_dir']}`",
        f"- resource samples: `{len(results.get('resource_samples', []))}`",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    api_url = args.api_url.rstrip("/")
    images = files_under(args.images_dir, IMAGE_EXTENSIONS, args.image_limit)
    videos = files_under(args.videos_dir, VIDEO_EXTENSIONS, args.video_limit) if args.videos_dir else []
    if not images:
        raise SystemExit(f"No images found in {args.images_dir}")

    plots_dir = Path(args.plots_dir)
    results = {
        "api_url": api_url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "images_dir": args.images_dir,
        "videos_dir": args.videos_dir,
        "image_count": len(images),
        "video_count": len(videos),
        "concurrency": args.concurrency,
        "system_before": system_snapshot(),
        "images": {},
        "videos": {},
        "image_throughput": {},
        "plots_dir": str(plots_dir),
    }

    with ResourceMonitor(args.sample_interval) as monitor:
        for mode in args.image_modes:
            rows, throughput = run_concurrent_images(api_url, images, mode, args.concurrency, args.timeout)
            results["images"][mode] = rows
            results["image_throughput"][mode] = throughput

        for max_frames in args.video_frames:
            rows = []
            for video_path in videos:
                rows.append(post_video(api_url, video_path, max_frames, args.video_tta, args.timeout))
            results["videos"][str(max_frames)] = rows

        results["resource_samples"] = monitor.samples

    results["system_after"] = system_snapshot()
    results["image_summary"] = {
        mode: summarize_latencies(rows)
        for mode, rows in results["images"].items()
    }
    results["video_summary"] = {
        frames: summarize_latencies(rows)
        for frames, rows in results["videos"].items()
    }

    make_plots(results, plots_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_path = Path(args.markdown_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_table(results), encoding="utf-8")

    print(f"Wrote JSON benchmark to {output_path}")
    print(f"Wrote Markdown benchmark to {md_path}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
