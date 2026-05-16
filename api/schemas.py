"""
Pydantic Schemas for DeepGuard API
"""
from typing import List, Optional, Any, Union
from pydantic import BaseModel, Field

class ImagePredictionResponse(BaseModel):
    label: str = Field(..., description="Classification result: 'REAL' or 'FAKE'")
    is_fake: bool = Field(..., description="Whether the image crossed the fake threshold")
    probability_fake: float = Field(..., description="Model probability that the face/image is fake")
    probability_real: float = Field(..., description="Model probability that the face/image is real")
    confidence: float = Field(..., description="Decision confidence [0.0 - 1.0]")
    threshold: float = Field(..., description="Decision threshold used for FAKE")
    face_detected: bool = Field(..., description="Whether a face was detected in the image")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    heatmap_base64: Optional[str] = Field(None, description="Base64 encoded Grad-CAM heatmap overlay (if FAKE)")
    tta_probabilities: Optional[List[float]] = Field(None, description="Per-augmentation fake probabilities")
    face_probability_fake: Optional[float] = Field(None, description="Fake probability from the detected face crop")
    full_probability_fake: Optional[float] = Field(None, description="Fake probability from the full uploaded image")
    analysis_note: Optional[str] = Field(None, description="Short diagnostic note about score agreement and image quality")

class VideoJobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current status: 'processing', 'done', 'failed'")
    message: str = Field(..., description="Status message")

class FrameResult(BaseModel):
    frame_index: int
    label: str
    is_fake: Optional[bool] = None
    probability_fake: Optional[float] = None
    probability_real: Optional[float] = None
    confidence: float

class VideoResultResponse(BaseModel):
    job_id: str
    status: str
    label: Optional[str] = None
    is_fake: Optional[bool] = None
    probability_fake: Optional[float] = None
    probability_real: Optional[float] = None
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    frames_analyzed: Optional[int] = None
    n_frames_analyzed: Optional[int] = None
    fake_frame_ratio: Optional[float] = None
    frame_results: Optional[List[FrameResult]] = None
    timeline: Optional[List[float]] = Field(None, description="Timeline of FAKE probabilities")
    frame_probabilities: Optional[List[float]] = Field(None, description="Alias for timeline")
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    uptime_seconds: float
    threshold: Optional[float] = None
    inference_tta: Optional[Union[bool, str]] = None
    model_loaded: bool = False
    device: Optional[str] = None
    cuda_available: bool = False
    cuda_device_name: Optional[str] = None
