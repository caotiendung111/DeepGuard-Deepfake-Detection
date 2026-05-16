"""
Async video job runner.

The worker keeps the public in-memory job contract used by the API, but bounds
work per job and moves CSV logging onto a lightweight queue.
"""
import os
import queue
import threading
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Union

import numpy as np
from loguru import logger

from ..dependencies import app_state
from ..schemas import FrameResult, VideoResultResponse
from api.monitoring import record_video_job
from src.inference.predictor import _is_uncertain, predict_probabilities_batch
from src.inference.video_processor import InferenceVideoProcessor


video_jobs: Dict[str, VideoResultResponse] = {}
_cancelled_jobs: Set[str] = set()
_jobs_lock = threading.RLock()
_max_video_jobs = max(1, int(os.getenv("MAX_VIDEO_JOBS", "2")))
_video_slots = threading.BoundedSemaphore(value=_max_video_jobs)
_log_queue: "queue.Queue[dict]" = queue.Queue(maxsize=1024)
_log_writer_started = False
_log_writer_lock = threading.Lock()


def _ensure_log_writer() -> None:
    global _log_writer_started
    with _log_writer_lock:
        if _log_writer_started:
            return
        thread = threading.Thread(target=_log_writer_loop, name="deepguard-video-log-writer", daemon=True)
        thread.start()
        _log_writer_started = True


def _log_writer_loop() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "predictions.csv"

    while True:
        item = _log_queue.get()
        try:
            if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:
                rotated = log_dir / f"predictions_{int(datetime.now().timestamp())}.csv"
                log_file.rename(rotated)

            is_new = not log_file.exists() or log_file.stat().st_size == 0
            with log_file.open("a", encoding="utf-8") as f:
                if is_new:
                    f.write("timestamp,input_type,label,probability_fake,confidence,threshold,processing_time_ms\n")
                f.write(
                    f"{item['timestamp']},video,{item['label']},"
                    f"{item['probability_fake']},{item['confidence']},"
                    f"{item['threshold']},{item['processing_time_ms']}\n"
                )
        finally:
            _log_queue.task_done()


def cancel_video_job(job_id: str) -> bool:
    with _jobs_lock:
        if job_id not in video_jobs:
            return False
        _cancelled_jobs.add(job_id)
        if video_jobs[job_id].status in {"pending", "processing"}:
            video_jobs[job_id].status = "cancelled"
            video_jobs[job_id].error = "Video job cancelled"
        return True


def cancel_all_video_jobs(reason: str = "Server is shutting down") -> int:
    with _jobs_lock:
        job_ids = [
            job_id for job_id, job in video_jobs.items()
            if job.status in {"pending", "processing"}
        ]
    for job_id in job_ids:
        cancel_video_job(job_id)
        with _jobs_lock:
            video_jobs[job_id].error = reason
    return len(job_ids)


def _is_cancelled(job_id: str) -> bool:
    with _jobs_lock:
        return job_id in _cancelled_jobs


def _mark_failed(job_id: str, error: str) -> None:
    with _jobs_lock:
        if job_id in video_jobs:
            video_jobs[job_id].status = "failed"
            video_jobs[job_id].error = error


def process_video_background(
    job_id: str,
    video_path: str,
    threshold: Optional[float] = None,
    max_frames: int = 32,
    timeout_seconds: float = 120.0,
    use_tta: Optional[Union[bool, str]] = None,
) -> None:
    """
    Background worker function to process a video.

    Thread cancellation is cooperative: the router can mark a job as cancelled,
    and this worker checks that flag between expensive stages.
    """
    _ensure_log_writer()
    t0 = time.monotonic()
    acquired = _video_slots.acquire(blocking=False)
    if not acquired:
        _mark_failed(job_id, "Too many video jobs are already running")
        Path(video_path).unlink(missing_ok=True)
        return

    try:
        if _is_cancelled(job_id):
            return

        video_jobs[job_id].status = "processing"

        detector = app_state["face_detector"]
        model = app_state["model"]
        cfg = app_state["config"]
        device = next(model.parameters()).device
        decision_threshold = cfg.threshold if threshold is None else threshold
        effective_tta = cfg.inference_tta if use_tta is None else use_tta
        n_frames = max(1, min(int(max_frames), 128))
        inference_batch_size = getattr(cfg, "inference_batch_size", None)
        use_amp = bool(getattr(cfg, "inference_amp", False))
        video_chunk_size = int(getattr(cfg, "video_chunk_size", 16))
        face_batch_size = int(getattr(cfg, "face_batch_size", 8))
        face_cache_gap = int(getattr(cfg, "face_cache_gap", 5))
        adaptive_low = float(getattr(cfg, "adaptive_tta_threshold_low", 0.4))
        adaptive_high = float(getattr(cfg, "adaptive_tta_threshold_high", 0.6))
        video_chunk_size = max(1, min(video_chunk_size, n_frames))
        tta_mode = effective_tta.lower().strip() if isinstance(effective_tta, str) else effective_tta
        use_video_adaptive_tta = tta_mode in {"adaptive", "auto"}
        decided_tta = None

        processor = InferenceVideoProcessor(
            face_detector=detector,
            image_size=cfg.image_size,
            face_batch_size=face_batch_size,
        )
        timeline = []
        sampled_indices = []
        frame_results = []
        chunks_processed = 0

        logger.info(
            f"Video job {job_id} started | max_frames={n_frames} "
            f"chunk_size={video_chunk_size} tta={effective_tta}"
        )

        for chunk in processor.iter_face_crops(
            video_path,
            n_frames=n_frames,
            chunk_size=video_chunk_size,
            timeout_seconds=timeout_seconds,
            cancel_check=lambda: _is_cancelled(job_id),
            fallback_full_frame=False,
            use_box_cache=True,
            max_cache_gap=face_cache_gap,
        ):
            if _is_cancelled(job_id):
                return
            if (time.monotonic() - t0) > timeout_seconds:
                _mark_failed(job_id, f"Video processing timed out after {timeout_seconds:.0f}s")
                return

            frame_indices = [frame_idx for frame_idx, _ in chunk]
            faces = [face_rgb for _, face_rgb in chunk]
            chunk_tta = decided_tta if decided_tta is not None else effective_tta
            if use_video_adaptive_tta and decided_tta is None:
                base_probs, _ = predict_probabilities_batch(
                    model=model,
                    images_rgb=faces,
                    image_size=cfg.image_size,
                    device=device,
                    use_tta=False,
                    batch_size=inference_batch_size,
                    use_amp=use_amp,
                )
                base_mean = float(np.mean(base_probs))
                decided_tta = _is_uncertain(base_mean, (adaptive_low, adaptive_high))
                chunk_tta = decided_tta
                logger.info(
                    f"Video job {job_id} adaptive TTA decision | "
                    f"first_chunk_mean={base_mean:.4f} use_tta={decided_tta}"
                )

            chunk_probs, _ = predict_probabilities_batch(
                model=model,
                images_rgb=faces,
                image_size=cfg.image_size,
                device=device,
                use_tta=chunk_tta,
                batch_size=inference_batch_size,
                use_amp=use_amp,
            )
            chunks_processed += 1
            sampled_indices.extend(frame_indices)
            timeline.extend(chunk_probs)

            for frame_idx, prob in zip(frame_indices, chunk_probs):
                is_fake = prob >= decision_threshold
                frame_results.append(FrameResult(
                    frame_index=frame_idx,
                    label="FAKE" if is_fake else "REAL",
                    is_fake=is_fake,
                    probability_fake=prob,
                    probability_real=1.0 - prob,
                    confidence=max(prob, 1.0 - prob),
                ))

        if _is_cancelled(job_id):
            return
        if (time.monotonic() - t0) > timeout_seconds:
            _mark_failed(job_id, f"Video processing timed out after {timeout_seconds:.0f}s")
            return
        if not timeline:
            _mark_failed(job_id, "No faces detected in sampled video frames")
            return

        mean_fake_probability = float(np.mean(timeline))
        overall_is_fake = mean_fake_probability >= decision_threshold
        processing_time_ms = (time.monotonic() - t0) * 1000

        result = video_jobs[job_id]
        result.status = "done"
        result.label = "FAKE" if overall_is_fake else "REAL"
        result.is_fake = overall_is_fake
        result.probability_fake = mean_fake_probability
        result.probability_real = 1.0 - mean_fake_probability
        result.confidence = max(mean_fake_probability, 1.0 - mean_fake_probability)
        result.threshold = decision_threshold
        result.frames_analyzed = len(frame_results)
        result.n_frames_analyzed = len(frame_results)
        result.fake_frame_ratio = float(
            sum(1 for value in timeline if value >= decision_threshold) / len(timeline)
        )
        result.frame_results = frame_results
        result.timeline = timeline
        result.frame_probabilities = timeline
        result.processing_time_ms = processing_time_ms

        record_video_job("done", processing_time_ms / 1000, len(frame_results))
        logger.info(
            f"Video job {job_id} done | frames={len(frame_results)} "
            f"chunks={chunks_processed} batch_size={inference_batch_size or 'auto'} "
            f"latency_ms={processing_time_ms:.1f}"
        )

        try:
            _log_queue.put_nowait({
                "timestamp": datetime.now().isoformat(),
                "label": result.label,
                "probability_fake": result.probability_fake,
                "confidence": result.confidence,
                "threshold": result.threshold,
                "processing_time_ms": result.processing_time_ms,
            })
        except queue.Full:
            pass

    except Exception as exc:
        record_video_job("failed", time.monotonic() - t0, 0)
        _mark_failed(job_id, str(exc))
    finally:
        if _is_cancelled(job_id) and video_jobs[job_id].status != "done":
            video_jobs[job_id].status = "cancelled"
            video_jobs[job_id].error = "Video job cancelled"
        with _jobs_lock:
            _cancelled_jobs.discard(job_id)
        _video_slots.release()
        if os.path.exists(video_path):
            os.remove(video_path)
        gc.collect()
