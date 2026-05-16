"""
Utilities for loading detector checkpoints in inference/evaluation code.

The project has produced checkpoints in two formats over time:
- full DeepfakeDetector keys: backbone.conv_stem.*, classifier.*
- raw timm backbone keys: conv_stem.*, blocks.*, classifier.*

This loader accepts both, so API/evaluation code does not silently drift from
the training checkpoint format.
"""
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from loguru import logger

from src.models import build_model


def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _add_backbone_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(("backbone.", "classifier.")):
            remapped[key] = value
        else:
            remapped[f"backbone.{key}"] = value
    return remapped


def _checkpoint_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
        if tensor_values:
            return checkpoint
    raise ValueError("Checkpoint does not contain a usable state_dict")


def _checkpoint_config(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config") or checkpoint.get("metadata", {}).get("config")
        if isinstance(config, dict):
            return config
    return {}


def _load_with_best_mapping(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    base = _strip_prefix(state_dict, "module.")
    candidates = [
        ("as_saved", base),
        ("raw_backbone_prefixed", _add_backbone_prefix(base)),
        ("double_backbone_stripped", _strip_prefix(base, "backbone.")),
    ]

    best = None
    expected_keys = set(model.state_dict().keys())
    for name, candidate in candidates:
        candidate_keys = set(candidate.keys())
        matched = len(expected_keys & candidate_keys)
        if best is None or matched > best[0]:
            best = (matched, name, candidate)

        missing, unexpected = model.load_state_dict(candidate, strict=False)
        meaningful_missing = [
            key for key in missing
            if not key.endswith("num_batches_tracked")
        ]
        if not meaningful_missing and not unexpected:
            logger.info(f"Loaded checkpoint with mapping: {name}")
            return

    _, best_name, best_candidate = best
    missing, unexpected = model.load_state_dict(best_candidate, strict=False)
    meaningful_missing = [
        key for key in missing
        if not key.endswith("num_batches_tracked")
    ]
    if meaningful_missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not match model architecture. "
            f"Best mapping={best_name}, missing={meaningful_missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    logger.info(f"Loaded checkpoint with best-effort mapping: {best_name}")


def load_detector_checkpoint(
    checkpoint_path: str,
    cfg,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Build a detector and load weights from a checkpoint.

    Returns:
        (model, checkpoint_metadata)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ckpt_config = _checkpoint_config(checkpoint)

    backbone = ckpt_config.get("backbone", getattr(cfg, "backbone", "efficientnet_b4"))
    num_classes = ckpt_config.get("num_classes", getattr(cfg, "num_classes", 1))
    dropout_rate = ckpt_config.get("dropout_rate", getattr(cfg, "dropout_rate", 0.3))

    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=False,
    )
    _load_with_best_mapping(model, _checkpoint_state_dict(checkpoint))
    model.to(device).eval()

    metadata = {
        "backbone": backbone,
        "checkpoint_path": str(path),
        "config": ckpt_config,
    }
    return model, metadata
