# ============================================================
# DeepGuard — Dockerfile (CPU)
# ============================================================
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY configs/ ./configs/
COPY setup.py .

# Install package
RUN pip install -e . --no-deps

# Create necessary directories
RUN mkdir -p models/checkpoints data/raw data/processed logs

# Expose ports
EXPOSE 8000 7860 8501

# Environment defaults
ENV PYTHONPATH=/app
ENV MODEL_BACKBONE=efficientnet_b4
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Default command: run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
