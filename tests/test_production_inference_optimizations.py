"""
Runnable tests for production inference optimizations.

These tests use mocks instead of a real model checkpoint or downloaded
InsightFace package. They should run after installing normal test/runtime deps:

    python -m pytest tests/test_production_inference_optimizations.py -v
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient


def _dummy_jpeg_bytes() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_detect_boxes_batch_scalar_fallback():
    from src.inference.face_detector import FaceDetector

    detector = FaceDetector.__new__(FaceDetector)
    detector._insightface = None
    detector._mtcnn = None
    detector._haar = None
    detector.face_size = 224
    detector.padding = 0.2

    calls = []

    def fake_detect_boxes(image_rgb):
        calls.append(image_rgb)
        return [(1, 2, 10, 12)]

    detector.detect_boxes = fake_detect_boxes
    images = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]

    boxes = detector.detect_boxes_batch(images, batch_size=2)

    assert len(calls) == 4
    assert boxes == [[(1, 2, 10, 12)]] * 4


@pytest.mark.parametrize("backend", ["insightface", "mtcnn", "haar", "auto"])
def test_face_detector_backend_fallback(monkeypatch, backend):
    from src.inference.face_detector import FaceDetector

    monkeypatch.setattr(FaceDetector, "_init_insightface", lambda self: False)
    monkeypatch.setattr(FaceDetector, "_init_mtcnn", lambda self: False)

    def fake_init_haar(self):
        self._haar = object()
        self.active_backend = "haar"

    monkeypatch.setattr(FaceDetector, "_init_haar", fake_init_haar)

    detector = FaceDetector(backend=backend, device="cpu")

    assert detector.active_backend == "haar"


def test_image_predictor_adaptive_tta_skips_tta_when_confident(monkeypatch):
    from src.inference import predictor
    from src.inference.predictor import ImagePredictor

    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    forward_calls = []

    monkeypatch.setattr(
        predictor,
        "_cached_val_transforms",
        lambda image_size: lambda image: {"image": torch.zeros(3, image_size, image_size)},
    )

    def tta_should_not_run(image_size):
        raise AssertionError("TTA transforms should not be built for confident adaptive prediction")

    monkeypatch.setattr(predictor, "_cached_tta_transforms", tta_should_not_run)

    def fake_forward(model, tensors, device, batch_size=32):
        forward_calls.append(len(tensors))
        return np.array([0.9], dtype=np.float32)

    monkeypatch.setattr(predictor, "_forward_probabilities", fake_forward)

    image_predictor = ImagePredictor(
        model=model,
        device="cpu",
        image_size=16,
        threshold=0.5,
        use_tta="adaptive",
    )
    result = image_predictor.predict(image)

    assert result.label == "FAKE"
    assert result.probability == pytest.approx(0.9)
    assert result.tta_probabilities == pytest.approx([0.9])
    assert forward_calls == [1]


def test_iter_face_crops_video_no_padding_and_box_cache(tmp_path):
    from src.inference.video_processor import InferenceVideoProcessor

    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (32, 32),
    )
    for value in (20, 80, 140):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()

    class FakeDetector:
        def __init__(self):
            self.detect_calls = 0

        def detect_boxes_batch(self, frames_rgb, batch_size=16):
            self.detect_calls += 1
            # First frame detects a face; later frames miss and should reuse
            # the previous box while within max_cache_gap.
            boxes = [[(4, 4, 24, 24)]]
            boxes.extend([] for _ in range(len(frames_rgb) - 1))
            return boxes

        def _crop(self, image_rgb, box):
            x1, y1, x2, y2 = box
            return cv2.resize(image_rgb[y1:y2, x1:x2], (16, 16))

    processor = InferenceVideoProcessor(
        face_detector=FakeDetector(),
        image_size=16,
        face_batch_size=2,
    )

    chunks = list(processor.iter_face_crops(
        str(video_path),
        n_frames=8,
        chunk_size=2,
        fallback_full_frame=False,
        use_box_cache=True,
        max_cache_gap=5,
    ))
    crops = [item for chunk in chunks for item in chunk]

    assert len(crops) == 3  # short video is not padded to n_frames=8
    assert [frame_idx for frame_idx, _ in crops] == [0, 1, 2]
    assert all(crop.shape == (16, 16, 3) for _, crop in crops)


def test_process_video_background_streaming_aggregation(monkeypatch, tmp_path):
    from api.jobs import video_processor as jobs

    video_path = tmp_path / "job.mp4"
    video_path.write_bytes(b"placeholder")

    chunks = [
        [(0, np.zeros((16, 16, 3), dtype=np.uint8)), (10, np.zeros((16, 16, 3), dtype=np.uint8))],
        [(20, np.zeros((16, 16, 3), dtype=np.uint8))],
    ]
    expected_probs = [0.2, 0.8, 0.5]

    class FakeProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def iter_face_crops(self, *args, **kwargs):
            yield from chunks

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

    call_offsets = {"value": 0}

    def fake_predict_probabilities_batch(**kwargs):
        n = len(kwargs["images_rgb"])
        start = call_offsets["value"]
        call_offsets["value"] += n
        return expected_probs[start:start + n], []

    monkeypatch.setattr(jobs, "InferenceVideoProcessor", FakeProcessor)
    monkeypatch.setattr(jobs, "predict_probabilities_batch", fake_predict_probabilities_batch)
    monkeypatch.setattr(jobs, "record_video_job", lambda *args, **kwargs: None)

    job_id = "streaming-job"
    jobs.video_jobs[job_id] = jobs.VideoResultResponse(job_id=job_id, status="pending")
    jobs.app_state["face_detector"] = object()
    jobs.app_state["model"] = FakeModel()
    jobs.app_state["config"] = SimpleNamespace(
        image_size=16,
        threshold=0.5,
        inference_tta=False,
        inference_batch_size=4,
        video_chunk_size=2,
        face_batch_size=2,
        face_cache_gap=5,
        adaptive_tta_threshold_low=0.4,
        adaptive_tta_threshold_high=0.6,
    )

    jobs.process_video_background(job_id, str(video_path), max_frames=3, timeout_seconds=30)

    result = jobs.video_jobs[job_id]
    assert result.status == "done"
    assert result.frame_probabilities == pytest.approx(expected_probs)
    assert result.probability_fake == pytest.approx(np.mean(expected_probs))
    assert result.fake_frame_ratio == pytest.approx(2 / 3)
    assert not video_path.exists()


def test_delete_cancel_job_endpoint_marks_job_cancelled():
    from api.jobs.video_processor import VideoResultResponse, video_jobs
    from api.main import app

    job_id = "cancel-via-api"
    video_jobs[job_id] = VideoResultResponse(job_id=job_id, status="pending")

    client = TestClient(app)
    response = client.delete(f"/predict/video/{job_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "cancelled"
    assert "cancelled" in payload["error"].lower()


def test_request_use_tta_overrides_config(monkeypatch):
    from api.main import app
    from api.schemas import ImagePredictionResponse
    import api.routers.predict as predict_router

    captured = {}

    def fake_process_image_sync(image_bytes, threshold=None, return_heatmap=True, use_tta=None):
        captured["use_tta"] = use_tta
        return ImagePredictionResponse(
            label="REAL",
            is_fake=False,
            probability_fake=0.1,
            probability_real=0.9,
            confidence=0.9,
            threshold=threshold or 0.5,
            face_detected=True,
            processing_time_ms=1.0,
        )

    monkeypatch.setattr(predict_router, "process_image_sync", fake_process_image_sync)

    client = TestClient(app)
    response = client.post(
        "/predict/image",
        data={"use_tta": "false"},
        files={"file": ("image.jpg", _dummy_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert captured["use_tta"] == "false"
