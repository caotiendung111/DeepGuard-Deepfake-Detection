"""
DeepGuard — Configuration System
YAML-based config with dataclass support and environment variable overrides.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """
    Master configuration dataclass for DeepGuard.
    Values can be overridden by environment variables.
    """
    # --- Model ---
    backbone: str = "efficientnet_b4"
    num_classes: int = 1
    dropout_rate: float = 0.3
    pretrained: bool = True
    image_size: int = 224
    threshold: float = 0.75

    # --- Training ---
    seed: int = 42
    num_epochs: int = 30
    warmup_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    early_stopping_patience: int = 7
    save_every: int = 5

    # --- Data ---
    data_root: str = "./data"
    train_csv: str = "data/metadata/train.csv"
    val_csv: str = "data/metadata/val.csv"
    test_csv: str = "data/metadata/test.csv"
    num_workers: int = 4
    n_frames: int = 16     # frames per video

    # --- Augmentation ---
    augment_level: str = "medium"  # "light" | "medium" | "heavy"
    jpeg_quality_lower: int = 60
    jpeg_quality_upper: int = 100
    coarse_dropout_holes: int = 8
    coarse_dropout_size: int = 32

    # --- Class Balancing ---
    use_weighted_sampler: bool = True  # WeightedRandomSampler in DataLoader

    # --- Loss ---
    loss_type: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    pos_weight: float = 1.0

    # --- Inference ---
    checkpoint_path: str = "models/checkpoints/best_model.pth"
    video_aggregation: str = "mean"
    inference_tta: Union[bool, str] = True
    face_detector_backend: str = "insightface"  # "insightface" | "mtcnn" | "haar" | "auto"
    inference_batch_size: Optional[int] = None
    inference_amp: bool = False
    video_chunk_size: int = 16
    face_batch_size: int = 8
    face_cache_gap: int = 5
    adaptive_tta_threshold_low: float = 0.4
    adaptive_tta_threshold_high: float = 0.6
    threshold_config_path: str = "configs/thresholds.yaml"
    auto_cpu_optimizations: bool = True
    cpu_inference_batch_size: int = 4
    cpu_image_size: int = 224
    cpu_video_chunk_size: int = 16
    cpu_max_threads: Optional[int] = None
    cpu_disable_tta_load_percent: float = 85.0

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "deepguard-experiments"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_size_mb: int = 100

    # --- Device ---
    device: str = "auto"

    def __post_init__(self):
        """Override values from environment variables."""
        env_overrides = set()
        env_map = {
            "DEVICE": "device",
            "MODEL_BACKBONE": "backbone",
            "MODEL_CHECKPOINT_PATH": "checkpoint_path",
            "MODEL_THRESHOLD": "threshold",
            "IMAGE_SIZE": "image_size",
            "INFERENCE_TTA": "inference_tta",
            "FACE_DETECTOR_BACKEND": "face_detector_backend",
            "INFERENCE_BATCH_SIZE": "inference_batch_size",
            "INFERENCE_AMP": "inference_amp",
            "VIDEO_CHUNK_SIZE": "video_chunk_size",
            "FACE_BATCH_SIZE": "face_batch_size",
            "FACE_CACHE_GAP": "face_cache_gap",
            "ADAPTIVE_TTA_THRESHOLD_LOW": "adaptive_tta_threshold_low",
            "ADAPTIVE_TTA_THRESHOLD_HIGH": "adaptive_tta_threshold_high",
            "AUTO_CPU_OPTIMIZATIONS": "auto_cpu_optimizations",
            "CPU_INFERENCE_BATCH_SIZE": "cpu_inference_batch_size",
            "CPU_IMAGE_SIZE": "cpu_image_size",
            "CPU_VIDEO_CHUNK_SIZE": "cpu_video_chunk_size",
            "CPU_MAX_THREADS": "cpu_max_threads",
            "CPU_DISABLE_TTA_LOAD_PERCENT": "cpu_disable_tta_load_percent",
            "BATCH_SIZE": "batch_size",
            "NUM_EPOCHS": "num_epochs",
            "LEARNING_RATE": "learning_rate",
            "NUM_WORKERS": "num_workers",
            "MLFLOW_TRACKING_URI": "mlflow_tracking_uri",
            "MLFLOW_EXPERIMENT_NAME": "mlflow_experiment_name",
            "API_PORT": "api_port",
        }
        for env_key, attr in env_map.items():
            env_val = os.getenv(env_key)
            if env_val is not None:
                env_overrides.add(attr)
                # Cast to correct type
                current_value = getattr(self, attr)
                expected_type = type(current_value)
                try:
                    if attr == "inference_tta":
                        lowered = env_val.lower().strip()
                        setattr(
                            self,
                            attr,
                            "adaptive" if lowered in {"adaptive", "auto"} else lowered in {"1", "true", "yes", "on"},
                        )
                    elif current_value is None:
                        setattr(self, attr, int(env_val))
                    elif expected_type is bool:
                        setattr(self, attr, env_val.lower() in {"1", "true", "yes", "on"})
                    else:
                        setattr(self, attr, expected_type(env_val))
                except (ValueError, TypeError):
                    pass
        self.apply_runtime_optimizations(env_overrides)

    def _resolved_device_type(self) -> str:
        if self.device != "auto":
            return str(self.device).lower()
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def apply_runtime_optimizations(self, env_overrides: Optional[set[str]] = None) -> None:
        """
        Apply conservative CPU defaults when CUDA is unavailable.

        Environment variables keep priority, so a deployment can opt back into
        heavier settings without editing YAML.
        """
        env_overrides = env_overrides or set()
        if not self.auto_cpu_optimizations or self._resolved_device_type() != "cpu":
            return

        if "inference_amp" not in env_overrides:
            self.inference_amp = False
        if "inference_batch_size" not in env_overrides:
            self.inference_batch_size = max(1, int(self.cpu_inference_batch_size))
        if "video_chunk_size" not in env_overrides:
            self.video_chunk_size = max(1, int(self.cpu_video_chunk_size))
        if "face_detector_backend" not in env_overrides and self.face_detector_backend == "mtcnn":
            self.face_detector_backend = "insightface"
        if "image_size" not in env_overrides and self.image_size > self.cpu_image_size:
            self.image_size = int(self.cpu_image_size)

        if "inference_tta" not in env_overrides:
            self.inference_tta = "adaptive"
            try:
                import psutil
                cpu_load = psutil.cpu_percent(interval=0.05)
                if cpu_load >= float(self.cpu_disable_tta_load_percent):
                    self.inference_tta = False
            except Exception:
                pass

        thread_count = self.cpu_max_threads
        if thread_count is None:
            try:
                thread_count = min(8, max(1, os.cpu_count() or 1))
            except Exception:
                thread_count = 4

        try:
            import torch
            torch.set_num_threads(max(1, int(thread_count)))
            torch.set_num_interop_threads(max(1, min(2, int(thread_count))))
        except Exception:
            pass

    def to_dict(self) -> dict:
        """Convert to plain dict for MLflow logging."""
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        """Save config to YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(yaml_path: Optional[str] = None) -> Config:
    """
    Load config from YAML file, then apply env variable overrides.

    Args:
        yaml_path: Path to YAML config file. Uses defaults if None.

    Returns:
        Config dataclass instance.
    """
    if yaml_path is None and Path("configs/base.yaml").exists():
        yaml_path = "configs/base.yaml"

    cfg_dict = {}
    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

    # Filter only known Config fields
    valid_keys = Config.__dataclass_fields__.keys()
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}

    cfg = Config(**filtered)

    threshold_path = Path(cfg.threshold_config_path)
    if os.getenv("MODEL_THRESHOLD") is None and threshold_path.exists():
        try:
            with open(threshold_path, "r") as f:
                threshold_cfg = yaml.safe_load(f) or {}
            calibrated_threshold = threshold_cfg.get("default_threshold")
            if calibrated_threshold is not None:
                cfg.threshold = float(calibrated_threshold)
        except (OSError, TypeError, ValueError):
            pass

    return cfg
