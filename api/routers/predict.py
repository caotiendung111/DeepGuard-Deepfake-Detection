"""
Prediction endpoints for image and video
"""
import asyncio
import os
import tempfile
import uuid
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, BackgroundTasks, Form
from loguru import logger

from ..schemas import ImagePredictionResponse, VideoJobResponse, VideoResultResponse
from ..dependencies import limiter
from ..services.inference import process_image_sync
from ..jobs.video_processor import cancel_video_job, process_video_background, video_jobs

router = APIRouter()

# Global threadpool for synchronous OpenCV/PyTorch tasks. Keep this small on CPU;
# use uvicorn --limit-concurrency to bound accepted requests at the server layer.
thread_pool = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("API_INFERENCE_WORKERS", "2"))))

MAX_IMAGE_SIZE = 10 * 1024 * 1024 # 10MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024 # 100MB
TTA_MODES = {None, "true", "false", "adaptive", "auto", "1", "0", "yes", "no", "on", "off"}


def _validate_tta_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().strip()
    if normalized not in TTA_MODES:
        raise HTTPException(status_code=422, detail="use_tta must be true, false, or adaptive")
    return "adaptive" if normalized == "auto" else normalized

@router.post("/predict/image", response_model=ImagePredictionResponse)
@limiter.limit("10/minute")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    threshold: float | None = Form(None),
    return_heatmap: bool = Form(True),
    use_tta: str | None = Form(None),
):
    """
    Synchronous image prediction.
    Detects face, crops, runs through DeepGuard model, and returns probabilities + GradCAM.
    """
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=422, detail="Threshold must be between 0.0 and 1.0")
    use_tta = _validate_tta_mode(use_tta)

    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=422, detail=f"Unsupported file format: {file.content_type}")
        
    # Read bytes and check size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image size exceeds 10MB limit")
        
    loop = asyncio.get_event_loop()
    try:
        # Run inference in threadpool to avoid blocking event loop
        task = partial(
            process_image_sync,
            image_bytes,
            threshold=threshold,
            return_heatmap=return_heatmap,
            use_tta=use_tta,
        )
        result = await asyncio.wait_for(
            loop.run_in_executor(thread_pool, task),
            timeout=60,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Image inference timed out")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Image prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error during inference")
        
    # Log prediction asynchronously to CSV for monitoring
    try:
        from datetime import datetime
        import aiofiles
        import os
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/predictions.csv"
        
        # Simple rotation: rename if > 10MB
        if os.path.exists(log_file) and os.path.getsize(log_file) > 10 * 1024 * 1024:
            os.rename(log_file, f"{log_dir}/predictions_{int(datetime.now().timestamp())}.csv")
            
        async with aiofiles.open(log_file, "a") as f:
            # Write header if new file
            if os.path.getsize(log_file) == 0:
                await f.write("timestamp,input_type,label,probability_fake,confidence,threshold,processing_time_ms\n")
            await f.write(
                f"{datetime.now().isoformat()},image,{result.label},"
                f"{result.probability_fake},{result.confidence},{result.threshold},"
                f"{result.processing_time_ms}\n"
            )
    except Exception as e:
        logger.error(f"Failed to write prediction log: {e}")
        
    return result


@router.post("/predict/video", response_model=VideoJobResponse)
@limiter.limit("5/minute")
async def predict_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float | None = Form(None),
    max_frames: int = Form(32),
    timeout_seconds: float = Form(120.0),
    use_tta: str | None = Form(None),
):
    """
    Asynchronous video prediction.
    Receives video, starts background processing task, and returns a job_id immediately.
    """
    if file.content_type not in ["video/mp4", "video/x-msvideo", "video/avi"]:
        raise HTTPException(status_code=422, detail=f"Unsupported file format: {file.content_type}")

    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=422, detail="Threshold must be between 0.0 and 1.0")
    use_tta = _validate_tta_mode(use_tta)
    if not 1 <= max_frames <= 128:
        raise HTTPException(status_code=422, detail="max_frames must be between 1 and 128")
    if not 5 <= timeout_seconds <= 600:
        raise HTTPException(status_code=422, detail="timeout_seconds must be between 5 and 600")
        
    # Save video to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    
    # Read chunk by chunk to check size
    size = 0
    with os.fdopen(fd, 'wb') as f:
        while True:
            chunk = await file.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_VIDEO_SIZE:
                os.remove(temp_path)
                raise HTTPException(status_code=413, detail="Video size exceeds 100MB limit")
            f.write(chunk)
            
    job_id = str(uuid.uuid4())
    
    # Initialize job in store
    video_jobs[job_id] = VideoResultResponse(
        job_id=job_id,
        status="pending"
    )
    
    task = partial(
        process_video_background,
        job_id,
        temp_path,
        threshold,
        max_frames=max_frames,
        timeout_seconds=timeout_seconds,
        use_tta=use_tta,
    )
    background_tasks.add_task(thread_pool.submit, task)
    
    return VideoJobResponse(
        job_id=job_id,
        status="processing",
        message="Video is being processed. Call GET /predict/video/{job_id} to check status."
    )


@router.get("/predict/video/{job_id}", response_model=VideoResultResponse)
@limiter.limit("30/minute")
async def get_video_status(request: Request, job_id: str):
    """
    Check the status of a video prediction job.
    """
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
        
    return video_jobs[job_id]


@router.delete("/predict/video/{job_id}", response_model=VideoResultResponse)
@limiter.limit("30/minute")
async def cancel_video(request: Request, job_id: str):
    """
    Cooperatively cancel a pending or running video prediction job.
    """
    if not cancel_video_job(job_id):
        raise HTTPException(status_code=404, detail="Job ID not found")
    return video_jobs[job_id]
