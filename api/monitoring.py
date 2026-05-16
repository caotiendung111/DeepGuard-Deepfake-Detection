"""
Prometheus monitoring helpers.

The module is optional: if prometheus_client is not installed, the API still
starts and /metrics returns a short explanatory response.
"""
import time
from typing import Optional

from fastapi import Response

try:
    from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    Counter = Gauge = Histogram = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    generate_latest = None


_REQUEST_COUNT = None
_REQUEST_LATENCY = None
_VIDEO_JOB_COUNT = None
_VIDEO_JOB_LATENCY = None
_VIDEO_FRAMES = None
_GPU_MEMORY_USED = None
_GPU_MEMORY_FREE = None


def _init_metrics() -> None:
    global _REQUEST_COUNT, _REQUEST_LATENCY, _VIDEO_JOB_COUNT
    global _VIDEO_JOB_LATENCY, _VIDEO_FRAMES, _GPU_MEMORY_USED, _GPU_MEMORY_FREE

    if Counter is None or _REQUEST_COUNT is not None:
        return

    _REQUEST_COUNT = Counter(
        "deepguard_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    _REQUEST_LATENCY = Histogram(
        "deepguard_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "path"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    )
    _VIDEO_JOB_COUNT = Counter(
        "deepguard_video_jobs_total",
        "Video job count",
        ["status"],
    )
    _VIDEO_JOB_LATENCY = Histogram(
        "deepguard_video_job_duration_seconds",
        "Video job latency",
        ["status"],
        buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    )
    _VIDEO_FRAMES = Histogram(
        "deepguard_video_frames_analyzed",
        "Frames analyzed per completed video job",
        buckets=(1, 4, 8, 16, 32, 64, 128),
    )
    _GPU_MEMORY_USED = Gauge("deepguard_gpu_memory_used_bytes", "CUDA memory used by this process")
    _GPU_MEMORY_FREE = Gauge("deepguard_gpu_memory_free_bytes", "CUDA free memory reported by PyTorch")


def _update_gpu_metrics() -> None:
    if Gauge is None or _GPU_MEMORY_USED is None:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        _GPU_MEMORY_FREE.set(free_bytes)
        _GPU_MEMORY_USED.set(total_bytes - free_bytes)
    except Exception:
        return


def record_video_job(status: str, duration_seconds: float, frames_analyzed: int) -> None:
    _init_metrics()
    if _VIDEO_JOB_COUNT is None:
        return
    _VIDEO_JOB_COUNT.labels(status=status).inc()
    _VIDEO_JOB_LATENCY.labels(status=status).observe(max(duration_seconds, 0.0))
    if frames_analyzed:
        _VIDEO_FRAMES.observe(frames_analyzed)


def setup_monitoring(app) -> None:
    _init_metrics()

    @app.middleware("http")
    async def prometheus_middleware(request, call_next):
        if _REQUEST_COUNT is None:
            return await call_next(request)

        started = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        _REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status=str(response.status_code),
        ).inc()
        _REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration)
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        if generate_latest is None:
            return Response(
                "prometheus_client is not installed\n",
                media_type="text/plain",
                status_code=503,
            )
        _update_gpu_metrics()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
