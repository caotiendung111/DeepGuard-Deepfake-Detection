FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONFIG_PATH=configs/base.yaml \
    DEVICE=cpu \
    AUTO_CPU_OPTIMIZATIONS=true \
    INFERENCE_BATCH_SIZE=4 \
    INFERENCE_TTA=adaptive \
    FACE_DETECTOR_BACKEND=insightface \
    VIDEO_CHUNK_SIZE=16 \
    MAX_VIDEO_JOBS=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
    && pip install -r requirements.txt \
    && pip install insightface onnxruntime

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT} --workers 1 --limit-concurrency 5"]
