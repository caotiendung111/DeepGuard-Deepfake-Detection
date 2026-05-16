"""
Validate a DeepGuard checkpoint before deployment.

Checks:
- checkpoint path exists
- checkpoint keys match the model architecture selected by config/checkpoint metadata
- random inference succeeds on CPU/CUDA
- reports latency, probability range, and memory footprint

Usage:
    python scripts/validate_checkpoint.py --checkpoint models/checkpoints/best_model.pth
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.model_loader import load_detector_checkpoint
from src.inference.predictor import predict_probabilities_batch
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a DeepGuard checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-tta", default="false", choices=["false", "true", "adaptive"])
    parser.add_argument("--output", default="reports/checkpoint_validation.json")
    parser.add_argument("--markdown-output", default="reports/checkpoint_validation.md")
    return parser.parse_args()


def memory_snapshot(device: torch.device) -> dict:
    snapshot = {}
    if psutil is not None:
        proc = psutil.Process()
        snapshot["rss_mb"] = proc.memory_info().rss / 1024 / 1024
    if device.type == "cuda" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot["cuda_free_mb"] = free_bytes / 1024 / 1024
        snapshot["cuda_used_mb"] = (total_bytes - free_bytes) / 1024 / 1024
        snapshot["cuda_total_mb"] = total_bytes / 1024 / 1024
        snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024 / 1024
        snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024 / 1024
    return snapshot


def markdown_report(report: dict) -> str:
    lines = [
        "# DeepGuard Checkpoint Validation",
        "",
        f"- status: `{report['status']}`",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- backbone: `{report.get('backbone')}`",
        f"- device: `{report['device']}`",
        f"- image_size: `{report['image_size']}`",
        f"- num_samples: `{report['num_samples']}`",
        "",
        "## Inference",
        "",
        f"- average latency: `{report['latency_ms_avg']:.2f} ms`",
        f"- min latency: `{report['latency_ms_min']:.2f} ms`",
        f"- max latency: `{report['latency_ms_max']:.2f} ms`",
        f"- probability mean: `{report['probability_mean']:.4f}`",
        f"- probability min: `{report['probability_min']:.4f}`",
        f"- probability max: `{report['probability_max']:.4f}`",
        "",
        "## Memory",
        "",
        "```json",
        json.dumps(report["memory_after"], indent=2),
        "```",
    ]
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = Path(args.checkpoint or cfg.checkpoint_path)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    warnings = []
    started_load = time.perf_counter()
    model, metadata = load_detector_checkpoint(str(checkpoint), cfg, device)
    load_ms = (time.perf_counter() - started_load) * 1000
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    images = [
        np.random.randint(0, 256, (cfg.image_size, cfg.image_size, 3), dtype=np.uint8)
        for _ in range(args.num_samples)
    ]

    for _ in range(args.warmup):
        predict_probabilities_batch(
            model=model,
            images_rgb=images[: min(2, len(images))],
            image_size=cfg.image_size,
            device=device,
            use_tta=False,
            batch_size=args.batch_size,
            use_amp=bool(getattr(cfg, "inference_amp", False)),
        )

    latencies = []
    probabilities = []
    memory_before = memory_snapshot(device)
    for image in images:
        t0 = time.perf_counter()
        probs, _ = predict_probabilities_batch(
            model=model,
            images_rgb=[image],
            image_size=cfg.image_size,
            device=device,
            use_tta=args.use_tta,
            batch_size=args.batch_size,
            use_amp=bool(getattr(cfg, "inference_amp", False)),
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        probabilities.extend(probs)

    memory_after = memory_snapshot(device)
    if any(not np.isfinite(prob) for prob in probabilities):
        warnings.append("Non-finite probability detected")
    if not all(0.0 <= prob <= 1.0 for prob in probabilities):
        warnings.append("Probability outside [0, 1] detected")

    report = {
        "status": "ok" if not warnings else "warning",
        "checkpoint": str(checkpoint),
        "config": args.config,
        "backbone": metadata.get("backbone"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "image_size": cfg.image_size,
        "num_samples": args.num_samples,
        "parameters": params,
        "trainable_parameters": trainable_params,
        "load_ms": load_ms,
        "latency_ms_avg": float(np.mean(latencies)),
        "latency_ms_min": float(np.min(latencies)),
        "latency_ms_max": float(np.max(latencies)),
        "probability_mean": float(np.mean(probabilities)),
        "probability_min": float(np.min(probabilities)),
        "probability_max": float(np.max(probabilities)),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "warnings": warnings,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown_report(report), encoding="utf-8")

    print(f"Validation status: {report['status']}")
    print(f"Wrote JSON report to {output}")
    print(f"Wrote Markdown report to {markdown_output}")


if __name__ == "__main__":
    main()
