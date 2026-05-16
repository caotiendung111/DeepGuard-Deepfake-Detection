"""
DeepGuard — ONNX Model Export Script

Exports trained PyTorch model to ONNX format for:
- Faster CPU inference
- Deployment with ONNX Runtime
- Cross-platform compatibility

Usage:
    python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger
from src.models import build_model
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Export DeepGuard to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="efficientnet_b4")
    parser.add_argument("--output", type=str, default="models/checkpoints/deepguard.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-axes", action="store_true", default=True,
                        help="Enable dynamic batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger()

    device = torch.device("cpu")

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("metadata", {}).get("config", {}).get("backbone", args.backbone)

    model = build_model(backbone=backbone)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_axes:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    logger.info(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.success(f"ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(output_path))
        input_name = sess.get_inputs()[0].name
        dummy_np = dummy_input.numpy()
        output = sess.run(None, {input_name: dummy_np})
        logger.success(f"ONNX verification passed! Output shape: {output[0].shape}")
    except ImportError:
        logger.warning("onnxruntime not installed — skipping verification")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")


if __name__ == "__main__":
    main()
