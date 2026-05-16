"""
Inference-oriented video frame extraction.

This module keeps API video processing away from the training/data extraction
helper. It samples a bounded number of frames with OpenCV seek, then crops faces
in batches through the shared FaceDetector instance.
"""
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .face_detector import FaceDetector


CancelCheck = Optional[Callable[[], bool]]
_CROP_CACHE_MAX_ITEMS = 512
_CROP_CACHE: "OrderedDict[Tuple[str, int, int, int], np.ndarray]" = OrderedDict()


def _video_cache_prefix(video_path: str) -> Tuple[str, int, int]:
    path = Path(video_path).resolve()
    stat = path.stat()
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


def _get_cached_crop(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    try:
        key = (*_video_cache_prefix(video_path), int(frame_idx))
    except OSError:
        return None
    crop = _CROP_CACHE.get(key)
    if crop is not None:
        _CROP_CACHE.move_to_end(key)
        return crop.copy()
    return None


def _put_cached_crop(video_path: str, frame_idx: int, crop: np.ndarray) -> None:
    try:
        key = (*_video_cache_prefix(video_path), int(frame_idx))
    except OSError:
        return
    _CROP_CACHE[key] = crop.copy()
    _CROP_CACHE.move_to_end(key)
    while len(_CROP_CACHE) > _CROP_CACHE_MAX_ITEMS:
        _CROP_CACHE.popitem(last=False)


class InferenceVideoProcessor:
    def __init__(
        self,
        face_detector: FaceDetector,
        image_size: int = 224,
        face_batch_size: int = 16,
    ):
        self.face_detector = face_detector
        self.image_size = image_size
        self.face_batch_size = face_batch_size
        self._last_box: Optional[Tuple[int, int, int, int]] = None
        self._last_box_frame: Optional[int] = None

    def get_video_info(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return info

    def _sample_indices(self, total_frames: int, n_frames: int) -> List[int]:
        if total_frames <= 0:
            return []
        n = max(1, min(n_frames, total_frames))
        return sorted({int(i) for i in np.linspace(0, total_frames - 1, n)})

    def _largest_box(self, boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        return boxes[int(np.argmax(areas))]

    def _crop_chunk(
        self,
        video_path: str,
        frame_indices: Sequence[int],
        frames_rgb: Sequence[np.ndarray],
        fallback_full_frame: bool,
        use_box_cache: bool,
        max_cache_gap: int,
    ) -> List[Tuple[int, np.ndarray]]:
        crops: List[Tuple[int, np.ndarray]] = []
        pending_indices: List[int] = []
        pending_frames: List[np.ndarray] = []

        for frame_idx, frame_rgb in zip(frame_indices, frames_rgb):
            cached = _get_cached_crop(video_path, frame_idx)
            if cached is not None:
                crops.append((frame_idx, cached))
            else:
                pending_indices.append(frame_idx)
                pending_frames.append(frame_rgb)

        if not pending_frames:
            return crops

        boxes_batch = self.face_detector.detect_boxes_batch(
            pending_frames,
            batch_size=self.face_batch_size,
        )

        for frame_idx, frame_rgb, boxes in zip(pending_indices, pending_frames, boxes_batch):
            if boxes:
                box = self._largest_box(boxes)
                self._last_box = box
                self._last_box_frame = frame_idx
                crop = self.face_detector._crop(frame_rgb, box)
                _put_cached_crop(video_path, frame_idx, crop)
                crops.append((frame_idx, crop))
                continue

            can_reuse_box = (
                use_box_cache
                and self._last_box is not None
                and self._last_box_frame is not None
                and abs(frame_idx - self._last_box_frame) <= max_cache_gap
            )
            if can_reuse_box:
                crop = self.face_detector._crop(frame_rgb, self._last_box)
                _put_cached_crop(video_path, frame_idx, crop)
                crops.append((frame_idx, crop))
            elif fallback_full_frame:
                crop = cv2.resize(frame_rgb, (self.image_size, self.image_size))
                _put_cached_crop(video_path, frame_idx, crop)
                crops.append((frame_idx, crop))

        return sorted(crops, key=lambda item: item[0])

    def iter_face_crops(
        self,
        video_path: str,
        n_frames: int = 16,
        chunk_size: int = 32,
        timeout_seconds: Optional[float] = None,
        cancel_check: CancelCheck = None,
        fallback_full_frame: bool = False,
        use_box_cache: bool = True,
        max_cache_gap: int = 5,
    ) -> Generator[List[Tuple[int, np.ndarray]], None, None]:
        """
        Yield chunks of (frame_index, face_rgb) without holding the whole video
        sample set in memory.
        """
        started = time.monotonic()
        info = self.get_video_info(video_path)
        total_frames = info["total_frames"]
        cap = cv2.VideoCapture(str(video_path))

        def should_stop() -> bool:
            if cancel_check and cancel_check():
                return True
            return timeout_seconds is not None and (time.monotonic() - started) > timeout_seconds

        def flush(indices: List[int], frames: List[np.ndarray]):
            if not frames:
                return []
            return self._crop_chunk(
                video_path,
                indices,
                frames,
                fallback_full_frame=fallback_full_frame,
                use_box_cache=use_box_cache,
                max_cache_gap=max_cache_gap,
            )

        frame_indices: List[int] = []
        frames_rgb: List[np.ndarray] = []
        try:
            indices = self._sample_indices(total_frames, n_frames)
            if indices:
                for frame_idx in indices:
                    if should_stop():
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame_indices.append(frame_idx)
                    frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if len(frames_rgb) >= chunk_size:
                        crops = flush(frame_indices, frames_rgb)
                        if crops:
                            yield crops
                        frame_indices, frames_rgb = [], []
            else:
                frame_idx = 0
                sampled_count = 0
                while cap.isOpened() and sampled_count < n_frames:
                    if should_stop():
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_indices.append(frame_idx)
                    frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    sampled_count += 1
                    if len(frames_rgb) >= chunk_size:
                        crops = flush(frame_indices, frames_rgb)
                        if crops:
                            yield crops
                        frame_indices, frames_rgb = [], []
                    frame_idx += 1

            crops = flush(frame_indices, frames_rgb)
            if crops and not should_stop():
                yield crops
        finally:
            cap.release()

    def extract_frames_for_inference(
        self,
        video_path: str,
        n_frames: int = 16,
        timeout_seconds: Optional[float] = None,
        cancel_check: CancelCheck = None,
        fallback_full_frame: bool = False,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Return (frame_index, face_rgb) samples for model inference.

        Uses seek + evenly spaced indices when frame count is known. For unusual
        containers without frame count metadata, falls back to streaming until
        enough samples are collected.
        """
        results: List[Tuple[int, np.ndarray]] = []
        for chunk in self.iter_face_crops(
            video_path,
            n_frames=n_frames,
            chunk_size=self.face_batch_size,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
            fallback_full_frame=fallback_full_frame,
        ):
            results.extend(chunk)
        return results


def extract_frames_for_inference(
    video_path: str,
    face_detector: FaceDetector,
    image_size: int = 224,
    n_frames: int = 16,
    timeout_seconds: Optional[float] = None,
    cancel_check: CancelCheck = None,
    fallback_full_frame: bool = False,
) -> List[Tuple[int, np.ndarray]]:
    processor = InferenceVideoProcessor(
        face_detector=face_detector,
        image_size=image_size,
    )
    return processor.extract_frames_for_inference(
        str(Path(video_path)),
        n_frames=n_frames,
        timeout_seconds=timeout_seconds,
        cancel_check=cancel_check,
        fallback_full_frame=fallback_full_frame,
    )
