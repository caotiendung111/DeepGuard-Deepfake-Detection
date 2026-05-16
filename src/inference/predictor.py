"""
DeepGuard — Inference Predictor
Single image and video deepfake detection pipeline.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from ..data.transforms import get_tta_transforms, get_val_transforms
from ..data.video_processor import VideoProcessor
from ..models.detector import DeepfakeDetector


TtaMode = Union[bool, str]


@lru_cache(maxsize=8)
def _cached_val_transforms(image_size: int):
    return get_val_transforms(image_size)


@lru_cache(maxsize=8)
def _cached_tta_transforms(image_size: int):
    return tuple(get_tta_transforms(image_size))


@lru_cache(maxsize=8)
def _cached_cpu_tta_transforms(image_size: int):
    # CPU TTA is intentionally small: original + horizontal flip.
    return _cached_tta_transforms(image_size)[:2]


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _forward_probabilities(
    model: torch.nn.Module,
    tensors: Sequence[torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
    use_amp: bool = False,
) -> np.ndarray:
    if not tensors:
        return np.array([], dtype=np.float32)

    probs: List[torch.Tensor] = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(list(tensors[start:start + batch_size])).to(device, non_blocking=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with amp_context:
            logits = model(batch)
        probs.append(torch.sigmoid(logits).reshape(-1).detach().cpu())

    return torch.cat(probs).numpy()


def _resolve_batch_size(device: torch.device, requested: Optional[int], image_size: int) -> int:
    if requested and requested > 0:
        return requested
    if device.type != "cuda":
        return 4

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return 16

    if free_bytes < 512 * 1024 * 1024:
        logger.warning(
            f"Low CUDA free memory ({free_bytes / 1024 / 1024:.0f} MB); "
            "falling back to inference batch_size=1"
        )
        return 1

    # Conservative estimate for activations + model workspace. The input tensor
    # itself is small; convolution activations dominate and scale with pixels.
    bytes_per_sample = max(32 * 1024 * 1024, image_size * image_size * 3 * 4 * 96)
    estimated = int((free_bytes * 0.35) // bytes_per_sample)
    if estimated < 1:
        logger.warning("Unable to reserve memory for a normal CUDA batch; using batch_size=1")
        return 1
    return int(max(1, min(64, estimated)))


def _normalize_tta_mode(use_tta: TtaMode) -> TtaMode:
    if isinstance(use_tta, str):
        mode = use_tta.lower().strip()
        if mode in {"adaptive", "auto"}:
            return "adaptive"
        if mode in {"1", "true", "yes", "on"}:
            return True
        if mode in {"0", "false", "no", "off"}:
            return False
    return bool(use_tta)


def _tta_transforms_for_device(image_size: int, device: torch.device):
    if device.type == "cuda":
        return _cached_tta_transforms(image_size)
    return _cached_cpu_tta_transforms(image_size)


def _is_uncertain(probability: float, adaptive_range: Tuple[float, float]) -> bool:
    low, high = adaptive_range
    return low <= probability <= high


@dataclass
class PredictionResult:
    """Result from a single inference call."""
    label: str              # "REAL" or "FAKE"
    is_fake: bool
    probability: float      # P(fake) in [0, 1]
    confidence: float       # max(P(fake), 1-P(fake))
    processing_time_ms: float
    threshold: float = 0.5
    tta_probabilities: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "is_fake": self.is_fake,
            "probability_fake": round(self.probability, 4),
            "probability_real": round(1 - self.probability, 4),
            "confidence": round(self.confidence, 4),
            "threshold": round(self.threshold, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "tta_probabilities": [round(p, 4) for p in self.tta_probabilities],
        }


@dataclass
class VideoPredictionResult:
    """Aggregated result from video inference."""
    label: str
    is_fake: bool
    probability: float
    confidence: float
    frame_probabilities: List[float]
    n_frames_analyzed: int
    processing_time_ms: float
    fake_frame_ratio: float   # fraction of frames classified as fake

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "is_fake": self.is_fake,
            "probability_fake": round(self.probability, 4),
            "probability_real": round(1 - self.probability, 4),
            "confidence": round(self.confidence, 4),
            "n_frames_analyzed": self.n_frames_analyzed,
            "fake_frame_ratio": round(self.fake_frame_ratio, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class ImagePredictor:
    """
    Predicts whether a single image is real or deepfake.

    Usage:
        predictor = ImagePredictor.from_checkpoint("models/checkpoints/best_model.pth")
        result = predictor.predict("face.jpg")
        print(result.label, result.probability)
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        device: str = "auto",
        image_size: int = 224,
        threshold: float = 0.5,
        use_tta: bool = False,
    ):
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else device
        )
        self.model = model.to(self.device).eval()
        self.transform = _cached_val_transforms(image_size)
        self.tta_transforms = None
        self.threshold = threshold
        self.use_tta = use_tta
        self.image_size = image_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class,
        model_kwargs: dict = None,
        **predictor_kwargs
    ) -> "ImagePredictor":
        """Load predictor from a saved checkpoint."""
        model = model_class.load_checkpoint(
            checkpoint_path,
            **(model_kwargs or {})
        )
        return cls(model, **predictor_kwargs)

    def _load_rgb(self, image_input) -> np.ndarray:
        """Accept file path, PIL Image, or numpy array."""
        if isinstance(image_input, (str, Path)):
            img = np.array(Image.open(image_input).convert("RGB"))
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img = image_input if image_input.shape[-1] == 3 else cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
        return img

    def _preprocess(self, image_input) -> torch.Tensor:
        """Accept file path, PIL Image, or numpy array."""
        img = self._load_rgb(image_input)
        transformed = self.transform(image=img)
        tensor = transformed["image"].unsqueeze(0)  # (1, C, H, W)
        return tensor.to(self.device)

    def _predict_probability(self, image_input) -> Tuple[float, List[float]]:
        prob, probs = predict_probability(
            model=self.model,
            image_rgb=self._load_rgb(image_input),
            image_size=self.image_size,
            device=self.device,
            use_tta=self.use_tta,
        )
        return prob, probs

    @torch.no_grad()
    def predict(self, image_input, threshold: Optional[float] = None) -> PredictionResult:
        """Predict real/fake for a single image."""
        import time
        threshold = self.threshold if threshold is None else threshold

        t0 = time.perf_counter()
        prob, tta_probs = self._predict_probability(image_input)
        is_fake = prob >= threshold
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PredictionResult(
            label="FAKE" if is_fake else "REAL",
            is_fake=is_fake,
            probability=prob,
            confidence=max(prob, 1 - prob),
            processing_time_ms=elapsed_ms,
            threshold=threshold,
            tta_probabilities=tta_probs,
        )

    def predict_batch(
        self, image_inputs: List, threshold: Optional[float] = None
    ) -> List[PredictionResult]:
        """Predict a list of images in batch."""
        import time
        threshold = self.threshold if threshold is None else threshold
        t0 = time.perf_counter()

        images = [self._load_rgb(img) for img in image_inputs]
        probs, tta_probs = predict_probabilities_batch(
            model=self.model,
            images_rgb=images,
            image_size=self.image_size,
            device=self.device,
            use_tta=self.use_tta,
        )
        elapsed_each = ((time.perf_counter() - t0) * 1000) / max(len(probs), 1)

        results = []
        for prob, per_image_probs in zip(probs, tta_probs):
            is_fake = prob >= threshold
            results.append(PredictionResult(
                label="FAKE" if is_fake else "REAL",
                is_fake=is_fake,
                probability=prob,
                confidence=max(prob, 1 - prob),
                processing_time_ms=elapsed_each,
                threshold=threshold,
                tta_probabilities=per_image_probs,
            ))
        return results


@torch.no_grad()
def predict_probability(
    model: DeepfakeDetector,
    image_rgb: np.ndarray,
    image_size: int = 224,
    device: Optional[torch.device] = None,
    use_tta: TtaMode = True,
    batch_size: Optional[int] = None,
    adaptive_range: Tuple[float, float] = (0.4, 0.6),
    use_amp: bool = False,
) -> Tuple[float, List[float]]:
    """
    Return P(fake) for an RGB image, optionally averaged over TTA variants.

    This helper is used by API and video inference so they share the same
    preprocessing and test-time augmentation path.
    """
    if device is None:
        device = _model_device(model)

    mode = _normalize_tta_mode(use_tta)
    resolved_batch_size = _resolve_batch_size(device, batch_size, image_size)

    if mode == "adaptive":
        base_tensor = _cached_val_transforms(image_size)(image=image_rgb)["image"]
        base_prob = float(_forward_probabilities(
            model, [base_tensor], device, batch_size=resolved_batch_size, use_amp=use_amp
        )[0])
        if not _is_uncertain(base_prob, adaptive_range):
            return base_prob, [base_prob]
        transforms = _tta_transforms_for_device(image_size, device)
    else:
        transforms = _tta_transforms_for_device(image_size, device) if mode else (_cached_val_transforms(image_size),)

    tensors = [transform(image=image_rgb)["image"] for transform in transforms]
    probs = _forward_probabilities(
        model, tensors, device, batch_size=resolved_batch_size, use_amp=use_amp
    ).astype(float).tolist()

    # Final probability is the MEDIAN of all TTA variants to handle outliers
    return float(np.median(probs)), probs


@torch.no_grad()
def predict_probabilities_batch(
    model: DeepfakeDetector,
    images_rgb: Sequence[np.ndarray],
    image_size: int = 224,
    device: Optional[torch.device] = None,
    use_tta: TtaMode = True,
    batch_size: Optional[int] = None,
    max_tensor_batch: Optional[int] = None,
    adaptive_range: Tuple[float, float] = (0.4, 0.6),
    use_amp: bool = False,
) -> Tuple[List[float], List[List[float]]]:
    """
    Return P(fake) for many RGB images using batched model forwards.

    Output order matches input order. When TTA is enabled, each image is
    expanded into all TTA variants and then reduced with the same median rule
    used by ``predict_probability``.
    """
    if device is None:
        device = _model_device(model)
    if not images_rgb:
        return [], []

    mode = _normalize_tta_mode(use_tta)
    resolved_batch_size = _resolve_batch_size(device, batch_size, image_size)

    if mode == "adaptive":
        val_transform = _cached_val_transforms(image_size)
        base_tensors = [val_transform(image=image_rgb)["image"] for image_rgb in images_rgb]
        base_probs = _forward_probabilities(
            model, base_tensors, device, batch_size=resolved_batch_size, use_amp=use_amp
        ).astype(float)
        final_probs = base_probs.copy()
        per_image_probs = [[float(prob)] for prob in base_probs]
        uncertain_indices = [
            idx for idx, prob in enumerate(base_probs)
            if _is_uncertain(float(prob), adaptive_range)
        ]

        if uncertain_indices:
            tta_images = [images_rgb[idx] for idx in uncertain_indices]
            tta_final, tta_per_image = predict_probabilities_batch(
                model=model,
                images_rgb=tta_images,
                image_size=image_size,
                device=device,
                use_tta=True,
                batch_size=resolved_batch_size,
                max_tensor_batch=max_tensor_batch,
                adaptive_range=adaptive_range,
                use_amp=use_amp,
            )
            for original_idx, prob, probs in zip(uncertain_indices, tta_final, tta_per_image):
                final_probs[original_idx] = prob
                per_image_probs[original_idx] = probs

        return final_probs.astype(float).tolist(), per_image_probs

    transforms = _tta_transforms_for_device(image_size, device) if mode else (_cached_val_transforms(image_size),)
    tensor_chunk_size = max_tensor_batch or max(resolved_batch_size, resolved_batch_size * len(transforms))
    tensor_chunk_size = max(resolved_batch_size, tensor_chunk_size)

    raw_chunks = []
    tensors: List[torch.Tensor] = []
    for image_rgb in images_rgb:
        for transform in transforms:
            tensors.append(transform(image=image_rgb)["image"])
        if len(tensors) >= tensor_chunk_size:
            raw_chunks.append(_forward_probabilities(
                model, tensors, device, batch_size=resolved_batch_size, use_amp=use_amp
            ))
            tensors = []

    if tensors:
        raw_chunks.append(_forward_probabilities(
            model, tensors, device, batch_size=resolved_batch_size, use_amp=use_amp
        ))

    raw_probs = np.concatenate(raw_chunks) if raw_chunks else np.array([], dtype=np.float32)
    per_image = raw_probs.reshape(len(images_rgb), len(transforms))
    final_probs = np.median(per_image, axis=1)
    return final_probs.astype(float).tolist(), per_image.astype(float).tolist()


class VideoPredictor:
    """
    Predicts whether a video is a deepfake by analyzing multiple frames.

    Strategy:
    - Extract N evenly-spaced frames
    - Predict each frame independently
    - Aggregate: use mean probability, classify as FAKE if mean > threshold
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        device: str = "auto",
        image_size: int = 224,
        threshold: float = 0.5,
        n_frames: int = 16,
        aggregation: str = "mean",  # "mean" | "max" | "vote"
    ):
        self.image_predictor = ImagePredictor(model, device, image_size, threshold)
        self.n_frames = n_frames
        self.aggregation = aggregation
        self.threshold = threshold
        self.video_processor = VideoProcessor(
            face_size=image_size,
            use_face_detector=True,
            max_frames=n_frames * 3,  # Extract more, then subsample
        )

    @torch.no_grad()
    def predict(self, video_path: str) -> VideoPredictionResult:
        """Predict real/fake for a video file."""
        import time
        t0 = time.perf_counter()

        # Extract frames
        frames = self.video_processor.extract_frames_for_inference(
            video_path, n_frames=self.n_frames
        )

        if not frames:
            logger.warning(f"No frames extracted from {video_path}")
            return VideoPredictionResult(
                label="UNKNOWN", is_fake=False, probability=0.5,
                confidence=0.5, frame_probabilities=[], n_frames_analyzed=0,
                processing_time_ms=0.0, fake_frame_ratio=0.0
            )

        frame_probs, _ = predict_probabilities_batch(
            model=self.image_predictor.model,
            images_rgb=frames,
            image_size=self.image_predictor.image_size,
            device=self.image_predictor.device,
            use_tta=self.image_predictor.use_tta,
        )

        # Aggregate
        if self.aggregation == "mean":
            agg_prob = float(np.mean(frame_probs))
        elif self.aggregation == "max":
            agg_prob = float(np.max(frame_probs))
        elif self.aggregation == "vote":
            votes = [1 if p >= self.threshold else 0 for p in frame_probs]
            agg_prob = float(np.mean(votes))
        else:
            agg_prob = float(np.mean(frame_probs))

        is_fake = agg_prob >= self.threshold
        fake_ratio = sum(1 for p in frame_probs if p >= self.threshold) / len(frame_probs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return VideoPredictionResult(
            label="FAKE" if is_fake else "REAL",
            is_fake=is_fake,
            probability=agg_prob,
            confidence=max(agg_prob, 1 - agg_prob),
            frame_probabilities=frame_probs,
            n_frames_analyzed=len(frame_probs),
            processing_time_ms=elapsed_ms,
            fake_frame_ratio=fake_ratio,
        )
