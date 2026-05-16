"""
FastAPI application entrypoint for DeepGuard.
"""
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from loguru import logger

# Add project root to python path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.face_detector import FaceDetector
from src.utils.config import load_config
from src.inference.gradcam import GradCAMVisualizer
from src.inference.model_loader import load_detector_checkpoint
from src.inference.predictor import ImagePredictor, VideoPredictor

from api.dependencies import limiter, app_state
from api.middleware.logging_middleware import RequestLoggingMiddleware
from api.monitoring import setup_monitoring
from api.routers import predict
from api.schemas import HealthResponse
from api.jobs.video_processor import cancel_all_video_jobs
from src.utils.logger import setup_logger


setup_logger(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/deepguard.log"),
    rotation=os.getenv("LOG_ROTATION", "100 MB"),
    retention=os.getenv("LOG_RETENTION", "14 days"),
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load machine learning models and heavy resources before the app starts taking requests.
    """
    logger.info("Starting up FastAPI - Loading models...")
    
    cfg = load_config(os.getenv("CONFIG_PATH"))
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto" else cfg.device
    )
    
    # Path to the best model checkpoint
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(f"Model checkpoint not found at {checkpoint_path}. Using untrained weights for API!")
        from src.models import build_model
        model = build_model(
            backbone=cfg.backbone,
            num_classes=cfg.num_classes,
            dropout_rate=cfg.dropout_rate,
            pretrained=False,
        )
    else:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        model, model_metadata = load_detector_checkpoint(str(checkpoint_path), cfg, device)
        cfg.backbone = model_metadata.get("backbone", cfg.backbone)
        
    model.to(device).eval()
    
    face_detector = FaceDetector(
        device=str(device),
        backend=getattr(cfg, "face_detector_backend", "mtcnn"),
        face_size=cfg.image_size,
    )
    
    # Initialize GradCAM
    try:
        gradcam = GradCAMVisualizer(model=model, device=str(device))
    except ImportError:
        logger.warning("Captum/GradCAM not installed. Explanations will not be available.")
        gradcam = None
        
    # Initialize Predictors
    image_predictor = ImagePredictor(
        model=model,
        device=str(device),
        image_size=cfg.image_size,
        threshold=cfg.threshold,
        use_tta=cfg.inference_tta
    )
    video_predictor = VideoPredictor(
        model=model,
        device=str(device),
        image_size=cfg.image_size,
        threshold=cfg.threshold,
        n_frames=cfg.n_frames,
        aggregation=cfg.video_aggregation
    )
    
    # Store in global state
    app_state["model"] = model
    app_state["image_predictor"] = image_predictor
    app_state["video_predictor"] = video_predictor
    app_state["face_detector"] = face_detector
    app_state["gradcam"] = gradcam
    app_state["config"] = cfg
    
    logger.success(f"Models loaded successfully on {device}!")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down - Cleaning up resources...")
    cancelled = cancel_all_video_jobs()
    if cancelled:
        logger.info(f"Marked {cancelled} video job(s) as cancelled during shutdown.")
    app_state["model"] = None
    app_state["face_detector"] = None
    app_state["gradcam"] = None
    torch.cuda.empty_cache()


# Initialize FastAPI app
app = FastAPI(
    title="DeepGuard REST API",
    description="API for Deepfake Detection System",
    version="1.0.0",
    lifespan=lifespan
)

# Setup Rate Limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
setup_monitoring(app)
app.add_middleware(RequestLoggingMiddleware)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handler for generic exceptions to prevent exposing internals
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include Routers
app.include_router(predict.router, tags=["Prediction"])

@app.get("/health", response_model=HealthResponse, tags=["System"])
@limiter.exempt
async def health_check():
    """
    Check system health and model status.
    """
    from api.dependencies import get_uptime
    cfg = app_state["config"]
    
    return HealthResponse(
        status="ok",
        model=cfg.backbone if cfg else "unknown",
        version="1.0.0",
        uptime_seconds=get_uptime(),
        threshold=cfg.threshold if cfg else None,
        inference_tta=cfg.inference_tta if cfg else None,
        model_loaded=app_state["model"] is not None,
        device=str(next(app_state["model"].parameters()).device) if app_state["model"] is not None else None,
        cuda_available=torch.cuda.is_available(),
        cuda_device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
