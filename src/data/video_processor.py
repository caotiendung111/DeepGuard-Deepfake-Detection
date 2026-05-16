"""
DeepGuard — Video Processor
Extract frames from video files, detect and crop faces using MTCNN.
"""
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class VideoProcessor:
    """
    Processes video files to extract face-cropped frames for deepfake analysis.

    Usage:
        processor = VideoProcessor(face_size=224, fps_sample=3)
        frames = processor.extract_frames("video.mp4", output_dir="frames/")
    """

    def __init__(
        self,
        face_size: int = 224,
        fps_sample: int = 3,
        max_frames: Optional[int] = None,
        use_face_detector: bool = True,
        padding: float = 0.3,
    ):
        """
        Args:
            face_size: Output face crop size (square).
            fps_sample: Number of frames to sample per second.
            max_frames: Maximum frames to extract (None = all).
            use_face_detector: Whether to detect and crop faces.
            padding: Padding fraction around detected face bounding box.
        """
        self.face_size = face_size
        self.fps_sample = fps_sample
        self.max_frames = max_frames
        self.use_face_detector = use_face_detector
        self.padding = padding

        self._detector = None

    @property
    def detector(self):
        """Lazy-load MTCNN face detector."""
        if self._detector is None and self.use_face_detector:
            try:
                from facenet_pytorch import MTCNN
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._detector = MTCNN(
                    keep_all=False,
                    device=device,
                    post_process=False,
                )
                logger.info(f"MTCNN loaded on {device}")
            except ImportError:
                logger.warning("facenet-pytorch not installed. Skipping face detection.")
                self.use_face_detector = False
        return self._detector

    def get_video_info(self, video_path: str) -> dict:
        """Return basic metadata about a video file."""
        cap = cv2.VideoCapture(str(video_path))
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_sec": None,
        }
        if info["fps"] > 0:
            info["duration_sec"] = info["total_frames"] / info["fps"]
        cap.release()
        return info

    def iter_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields (frame_idx, BGR frame) at sampled FPS.
        """
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(video_fps / self.fps_sample))

        frame_idx = 0
        extracted = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                yield frame_idx, frame
                extracted += 1
                if self.max_frames and extracted >= self.max_frames:
                    break
            frame_idx += 1

        cap.release()

    def _crop_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect and return the largest face crop from a BGR frame."""
        if self.detector is None:
            return cv2.resize(frame_bgr, (self.face_size, self.face_size))

        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        boxes, _ = self.detector.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            return None

        # Use largest face
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        box = boxes[np.argmax(areas)]
        x1, y1, x2, y2 = [int(c) for c in box]

        h, w = frame_bgr.shape[:2]
        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        face = frame_bgr[y1:y2, x1:x2]
        face = cv2.resize(face, (self.face_size, self.face_size))
        return face

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        save_format: str = "jpg",
        jpeg_quality: int = 95,
    ) -> List[str]:
        """
        Extract face-cropped frames from video and save to output_dir.

        Returns:
            List of saved frame file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for frame_idx, frame in self.iter_frames(video_path):
            if self.use_face_detector:
                face = self._crop_face(frame)
                if face is None:
                    continue
            else:
                face = cv2.resize(frame, (self.face_size, self.face_size))

            filename = f"frame_{frame_idx:06d}.{save_format}"
            out_path = output_dir / filename

            if save_format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(out_path), face, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(str(out_path), face)

            saved_paths.append(str(out_path))

        logger.info(f"Extracted {len(saved_paths)} frames from {Path(video_path).name}")
        return saved_paths

    def extract_frames_for_inference(
        self, video_path: str, n_frames: int = 16
    ) -> List[np.ndarray]:
        """
        Extract n_frames evenly sampled frames for inference (returns RGB numpy arrays).
        """
        info = self.get_video_info(video_path)
        total = info["total_frames"]
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        if total > 0:
            indices = sorted({int(i) for i in np.linspace(0, max(total - 1, 0), n_frames)})
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                if self.use_face_detector:
                    face = self._crop_face(frame)
                    if face is not None:
                        frames.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                else:
                    face = cv2.resize(frame, (self.face_size, self.face_size))
                    frames.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                if len(frames) >= n_frames:
                    break
        else:
            while cap.isOpened() and len(frames) < n_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.use_face_detector:
                    face = self._crop_face(frame)
                    if face is not None:
                        frames.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                else:
                    face = cv2.resize(frame, (self.face_size, self.face_size))
                    frames.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        cap.release()

        # Pad if needed
        while len(frames) < n_frames and len(frames) > 0:
            frames.append(frames[-1])

        return frames
