# DeepGuard Staging Deployment Checklist

## 1. Host Preparation

- Install NVIDIA driver compatible with your GPU.
- Install Docker Engine and Docker Compose plugin.
- Install NVIDIA Container Toolkit.
- Verify GPU visibility:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. Repository and Artifacts

- Clone the repository.
- Place the checkpoint under `models/checkpoints/best_model.pth`.
- Backup the deployed checkpoint and config:

```bash
mkdir -p backups
cp models/checkpoints/best_model.pth backups/best_model.$(date +%Y%m%d).pth
cp configs/base.yaml backups/base.$(date +%Y%m%d).yaml
```

## 3. Validate Checkpoint

```bash
python scripts/validate_checkpoint.py \
  --checkpoint models/checkpoints/best_model.pth \
  --config configs/base.yaml \
  --device cuda
```

Review:

- `status`
- average latency
- probability range
- CUDA allocated/reserved memory

## 4. Environment Variables

Recommended staging values:

```bash
MODEL_CHECKPOINT_PATH=models/checkpoints/best_model.pth
DEVICE=cuda
FACE_DETECTOR_BACKEND=auto
INFERENCE_TTA=adaptive
MAX_VIDEO_JOBS=2
LOG_LEVEL=INFO
LOG_ROTATION=100 MB
LOG_RETENTION=14 days
```

For constrained GPU memory:

```bash
INFERENCE_TTA=false
INFERENCE_BATCH_SIZE=1
VIDEO_CHUNK_SIZE=8
FACE_BATCH_SIZE=4
```

## 5. Build and Run

```bash
docker compose up --build api
```

With monitoring:

```bash
docker compose --profile monitoring up --build
```

## 6. Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Verify:

- `model_loaded: true`
- `device: cuda:*` or `cuda`
- `cuda_available: true`
- `threshold` is expected
- Prometheus can scrape `/metrics`

## 7. Smoke Tests

```bash
pytest tests/ -v
python scripts/benchmark.py --images-dir data/sample --videos-dir data/videos
```

Track:

- image p95 latency
- video p95 latency
- video frames analyzed
- GPU free/used memory
- 5xx errors

## 8. Graceful Shutdown

- The FastAPI lifespan marks pending/running video jobs as cancelled during shutdown.
- Video workers check cancellation cooperatively between extraction and inference chunks.
- Temporary uploaded video files are removed in `finally`.

On staging, verify:

```bash
docker compose stop api
```

Then inspect logs for:

- `Marked N video job(s) as cancelled during shutdown`
- no leaked temp video files

## Troubleshooting

### CUDA out of memory

Reduce memory pressure:

```yaml
inference_tta: false
inference_batch_size: 1
video_chunk_size: 8
face_batch_size: 4
```

Or set env:

```bash
INFERENCE_TTA=false
INFERENCE_BATCH_SIZE=1
VIDEO_CHUNK_SIZE=8
```

Also watch `/metrics`:

- `deepguard_gpu_memory_free_bytes`
- `deepguard_gpu_memory_used_bytes`

### InsightFace model not found or first boot is slow

InsightFace downloads model files on first use. Check:

```bash
ls ~/.insightface/models
```

In Docker, the compose file persists this at `insightface-cache`.

CPU-only hosts should use:

```bash
pip install insightface onnxruntime
```

GPU hosts should use:

```bash
pip install insightface onnxruntime-gpu
```

### MTCNN detects no faces in video

Try:

```yaml
face_detector_backend: insightface
```

or for a lightweight fallback:

```yaml
face_detector_backend: haar
```

If the video is low resolution or side-profile heavy, enable full-frame fallback only after validating false-positive behavior.

### `RuntimeError: expected scalar type float but found half`

Mixed precision is optional and disabled by default:

```yaml
inference_amp: false
```

Only enable after validating the checkpoint:

```bash
python scripts/validate_checkpoint.py --device cuda
```

If AMP is enabled and this error appears, disable it or audit custom layers for dtype assumptions.
