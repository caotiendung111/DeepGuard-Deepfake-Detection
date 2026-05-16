"""
DeepGuard — Pydantic Schemas for API request/response models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ImagePredictionResponse(BaseModel):
    """Response schema for image deepfake detection."""
    label: str = Field(..., description="REAL or FAKE", examples=["FAKE"])
    is_fake: bool = Field(..., description="True if classified as deepfake")
    probability_fake: float = Field(..., ge=0.0, le=1.0,
                                    description="Probability that image is a deepfake")
    probability_real: float = Field(..., ge=0.0, le=1.0,
                                    description="Probability that image is real")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Model confidence (max of the two probabilities)")
    processing_time_ms: float = Field(..., description="Inference time in milliseconds")
    filename: Optional[str] = Field(None, description="Original filename")
    model_version: str = Field("1.0.0", description="Model version used for prediction")
    heatmap_base64: Optional[str] = Field(
        None, description="Base64-encoded Grad-CAM heatmap JPEG (if requested)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "label": "FAKE",
            "is_fake": True,
            "probability_fake": 0.9234,
            "probability_real": 0.0766,
            "confidence": 0.9234,
            "processing_time_ms": 45.2,
            "filename": "face_test.jpg",
            "model_version": "1.0.0",
        }
    }}


class VideoPredictionResponse(BaseModel):
    """Response schema for video deepfake detection."""
    label: str = Field(..., description="REAL or FAKE")
    is_fake: bool
    probability_fake: float = Field(..., ge=0.0, le=1.0)
    probability_real: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    n_frames_analyzed: int = Field(..., description="Number of frames analyzed")
    fake_frame_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of frames classified as FAKE"
    )
    frame_probabilities: List[float] = Field(
        ..., description="Per-frame P(fake) probability list"
    )
    processing_time_ms: float
    filename: Optional[str] = None
    model_version: str = "1.0.0"

    model_config = {"json_schema_extra": {
        "example": {
            "label": "FAKE",
            "is_fake": True,
            "probability_fake": 0.871,
            "probability_real": 0.129,
            "confidence": 0.871,
            "n_frames_analyzed": 16,
            "fake_frame_ratio": 0.875,
            "frame_probabilities": [0.91, 0.88, 0.79, 0.92],
            "processing_time_ms": 1240.5,
            "filename": "test_video.mp4",
        }
    }}


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    service: str
    version: str
    uptime_seconds: int
    model_loaded: bool
    device: str
