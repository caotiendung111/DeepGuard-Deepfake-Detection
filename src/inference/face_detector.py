"""
DeepGuard — MTCNN Face Detector Wrapper
"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image


class FaceDetector:
    """
    MTCNN-based face detector with fallback to OpenCV Haar cascades.

    Usage:
        detector = FaceDetector()
        face = detector.detect_and_crop(image_array)
    """

    def __init__(
        self,
        min_face_size: int = 40,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        device: str = "auto",
        face_size: int = 224,
        padding: float = 0.2,
        backend: str = "insightface",
    ):
        self.face_size = face_size
        self.padding = padding
        self.backend = backend
        self.active_backend = "none"
        self._insightface = None
        self._mtcnn = None
        self._haar = None
        self._device = device
        self._min_face_size = min_face_size
        self._thresholds = thresholds

        self._init_backend()

    def _init_backend(self):
        backend = (self.backend or "insightface").lower()
        if backend in {"insightface", "auto"} and self._init_insightface():
            return
        if backend in {"mtcnn", "auto", "insightface"} and self._init_mtcnn():
            return
        self._init_haar()

    def _init_insightface(self) -> bool:
        try:
            import torch
            from insightface.app import FaceAnalysis

            device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
                if self._device == "auto" else str(self._device)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if "cuda" in device_name else ["CPUExecutionProvider"]
            ctx_id = 0 if "cuda" in device_name else -1

            logger.info(
                "Initializing InsightFace detector. The first startup may download "
                "the buffalo_l model package into the InsightFace cache."
            )
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._insightface = app
            self.active_backend = "insightface"
            logger.info(f"InsightFace detector loaded on {device_name}")
            return True
        except Exception as exc:
            logger.warning(f"InsightFace detector unavailable: {exc}")
            return False

    def _init_mtcnn(self) -> bool:
        try:
            import torch
            from facenet_pytorch import MTCNN
            device = ("cuda" if torch.cuda.is_available() else "cpu") \
                if self._device == "auto" else self._device
            self._mtcnn = MTCNN(
                min_face_size=self._min_face_size,
                thresholds=list(self._thresholds),
                keep_all=True,
                device=device,
                post_process=False,
            )
            self.active_backend = "mtcnn"
            logger.info(f"MTCNN face detector loaded on {device}")
            return True
        except Exception as exc:
            logger.warning(f"facenet-pytorch detector unavailable: {exc}")
            return False

    def _init_haar(self):
        """Fallback to OpenCV Haar cascade face detector."""
        import os
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self._haar = cv2.CascadeClassifier(cascade_path)
            self.active_backend = "haar"
            logger.info("OpenCV Haar face detector loaded")
        else:
            logger.warning("No face detector available. Will return full frame.")

    def detect_boxes(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all face bounding boxes in an RGB image.

        Returns:
            List of (x1, y1, x2, y2) bounding boxes.
        """
        if self._insightface is not None:
            faces = self._insightface.get(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            if faces:
                boxes = []
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int).tolist()
                    boxes.append((x1, y1, x2, y2))
                return boxes

        if self._mtcnn is not None:
            pil_img = Image.fromarray(image_rgb)
            boxes, _ = self._mtcnn.detect(pil_img)
            if boxes is not None:
                return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]

        elif self._haar is not None:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                return [(x, y, x + w, y + h) for x, y, w, h in faces]

        return []

    def detect_boxes_batch(
        self,
        images_rgb: Sequence[np.ndarray],
        batch_size: int = 16,
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect face boxes for multiple RGB images.

        facenet-pytorch MTCNN can process a list of PIL images. When the active
        detector does not support batched input, this falls back to the scalar
        path so callers keep one code path.
        """
        if not images_rgb:
            return []

        if self._insightface is not None:
            # InsightFace FaceAnalysis.get is per-image. Keep this method so the
            # caller still benefits from a unified API and can switch backends.
            return [self.detect_boxes(image) for image in images_rgb]

        if self._mtcnn is not None:
            all_boxes: List[List[Tuple[int, int, int, int]]] = []
            for start in range(0, len(images_rgb), batch_size):
                chunk = images_rgb[start:start + batch_size]
                pil_images = [Image.fromarray(image) for image in chunk]
                try:
                    boxes_batch, _ = self._mtcnn.detect(pil_images)
                except Exception as exc:
                    logger.warning(f"Batched MTCNN detect failed; falling back to per-frame detect: {exc}")
                    return [self.detect_boxes(image) for image in images_rgb]

                for boxes in boxes_batch:
                    if boxes is None:
                        all_boxes.append([])
                    else:
                        all_boxes.append([
                            (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                            for b in boxes
                        ])
            return all_boxes

        return [self.detect_boxes(image) for image in images_rgb]

    def crop_largest_batch(
        self,
        images_rgb: Sequence[np.ndarray],
        batch_size: int = 16,
        fallback_full_frame: bool = False,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Crop the largest detected face from each image.

        Returns (input_index, face_rgb) pairs. Missing-face frames are skipped
        unless fallback_full_frame=True.
        """
        boxes_batch = self.detect_boxes_batch(images_rgb, batch_size=batch_size)
        crops: List[Tuple[int, np.ndarray]] = []

        for idx, (image_rgb, boxes) in enumerate(zip(images_rgb, boxes_batch)):
            if boxes:
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
                box = boxes[int(np.argmax(areas))]
                crops.append((idx, self._crop(image_rgb, box)))
            elif fallback_full_frame:
                crops.append((idx, cv2.resize(image_rgb, (self.face_size, self.face_size))))

        return crops

    def detect_and_crop(
        self,
        image_rgb: np.ndarray,
        select: str = "largest",  # "largest" | "all"
    ) -> Optional[np.ndarray]:
        """
        Detect face and return cropped, resized face region.

        Args:
            image_rgb: Input RGB numpy array.
            select: "largest" returns single largest face, "all" returns list.

        Returns:
            Cropped face as RGB numpy array, or None if no face detected.
        """
        boxes = self.detect_boxes(image_rgb)

        if not boxes:
            # Fall back to full frame
            return cv2.resize(image_rgb, (self.face_size, self.face_size))

        if select == "largest":
            # Select face with largest area
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
            box = boxes[np.argmax(areas)]
            return self._crop(image_rgb, box)

        return [self._crop(image_rgb, b) for b in boxes]

    def _crop(self, image_rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply padding and crop face from image."""
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = box

        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        face = image_rgb[y1:y2, x1:x2]
        return cv2.resize(face, (self.face_size, self.face_size))
