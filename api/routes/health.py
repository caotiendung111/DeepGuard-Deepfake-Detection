"""
DeepGuard — Health Check Endpoint
"""
import platform
import time
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()
START_TIME = time.time()


@router.get("/health", summary="Health check")
async def health_check():
    """Return service health status and system info."""
    uptime_seconds = int(time.time() - START_TIME)

    from api.main import model_registry
    model_loaded = model_registry.get("image_predictor") is not None

    return {
        "status": "healthy",
        "service": "DeepGuard API",
        "version": "1.0.0",
        "uptime_seconds": uptime_seconds,
        "model_loaded": model_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


@router.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "message": "🛡️ DeepGuard API is running",
        "docs": "/docs",
        "health": "/health",
    })
