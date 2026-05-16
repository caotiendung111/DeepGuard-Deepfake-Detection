"""
DeepGuard — Prediction Endpoints
POST /api/v1/predict/image
POST /api/v1/predict/video
"""
import io
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from ..schemas.prediction import ImagePredictionResponse, VideoPredictionResponse

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


def _get_predictors():
    """Retrieve predictors from the global model registry."""
    from api.dependencies import app_state
    return (
        app_state.get("image_predictor"),
        app_state.get("video_predictor"),
    )


@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    summary="Classify a single image as REAL or FAKE",
)
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WebP)"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="Decision threshold"),
    return_heatmap: bool = Form(False, description="Return Grad-CAM heatmap as base64"),
):
    """
    Analyze an image and classify it as REAL or FAKE deepfake.

    Returns:
    - **label**: REAL or FAKE
    - **probability_fake**: Probability that image is a deepfake (0-1)
    - **confidence**: Model confidence in its decision
    - **processing_time_ms**: Total inference time
    """
    # Validate content type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {file.content_type}. Allowed: {ALLOWED_IMAGE_TYPES}"
        )

    # Read file
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size: 100MB")

    image_predictor, _ = _get_predictors()

    # Demo mode if model not loaded
    if image_predictor is None:
        logger.warning("Model not loaded — returning demo response")
        import random
        fake_prob = random.uniform(0.1, 0.9)
        return ImagePredictionResponse(
            label="FAKE" if fake_prob >= threshold else "REAL",
            is_fake=fake_prob >= threshold,
            probability_fake=round(fake_prob, 4),
            probability_real=round(1 - fake_prob, 4),
            confidence=round(max(fake_prob, 1 - fake_prob), 4),
            processing_time_ms=12.5,
            model_version="demo",
            filename=file.filename,
        )

    try:
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        result = image_predictor.predict(pil_image, threshold=threshold)

        response = ImagePredictionResponse(
            label=result.label,
            is_fake=result.is_fake,
            probability_fake=result.probability,
            probability_real=1 - result.probability,
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
            filename=file.filename,
        )

        # Optional Grad-CAM heatmap
        if return_heatmap:
            try:
                import base64
                from src.inference.gradcam import GradCAMVisualizer
                viz = GradCAMVisualizer(image_predictor.model)
                heatmap = viz.generate(pil_image)

                # Encode to base64
                img = Image.fromarray(heatmap)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                response.heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")

        logger.info(
            f"Image prediction: {result.label} ({result.probability:.3f}) "
            f"| {result.processing_time_ms:.1f}ms | {file.filename}"
        )
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post(
    "/predict/video",
    response_model=VideoPredictionResponse,
    summary="Classify a video as REAL or FAKE",
)
async def predict_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI)"),
    threshold: float = Form(0.5, ge=0.0, le=1.0),
    n_frames: int = Form(16, ge=4, le=64, description="Number of frames to analyze"),
):
    """
    Analyze a video file and classify it as REAL or FAKE deepfake.

    The system extracts N evenly-spaced frames, detects faces, and classifies each.
    The final verdict is based on mean probability across all analyzed frames.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type: {file.content_type}. Allowed: {ALLOWED_VIDEO_TYPES}"
        )

    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size: 100MB")

    _, video_predictor = _get_predictors()

    if video_predictor is None:
        import random
        fake_prob = random.uniform(0.2, 0.85)
        frame_probs = [random.uniform(0.1, 0.9) for _ in range(n_frames)]
        return VideoPredictionResponse(
            label="FAKE" if fake_prob >= threshold else "REAL",
            is_fake=fake_prob >= threshold,
            probability_fake=round(fake_prob, 4),
            probability_real=round(1 - fake_prob, 4),
            confidence=round(max(fake_prob, 1 - fake_prob), 4),
            n_frames_analyzed=n_frames,
            fake_frame_ratio=round(sum(1 for p in frame_probs if p >= threshold) / n_frames, 4),
            frame_probabilities=frame_probs,
            processing_time_ms=350.0,
            model_version="demo",
            filename=file.filename,
        )

    try:
        # Save to temp file for OpenCV processing
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        video_predictor.n_frames = n_frames
        result = video_predictor.predict(tmp_path, threshold=threshold)

        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)

        logger.info(
            f"Video prediction: {result.label} ({result.probability:.3f}) "
            f"| {result.n_frames_analyzed} frames | {result.processing_time_ms:.0f}ms"
        )

        return VideoPredictionResponse(
            label=result.label,
            is_fake=result.is_fake,
            probability_fake=result.probability,
            probability_real=1 - result.probability,
            confidence=result.confidence,
            n_frames_analyzed=result.n_frames_analyzed,
            fake_frame_ratio=result.fake_frame_ratio,
            frame_probabilities=result.frame_probabilities,
            processing_time_ms=result.processing_time_ms,
            filename=file.filename,
        )

    except Exception as e:
        logger.error(f"Video prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
