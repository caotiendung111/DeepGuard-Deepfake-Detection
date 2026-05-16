# DeepGuard CPU Deployment

Recommended laptop profile for Intel Core i7-1165G7 / 16 GB RAM:

```powershell
$env:DEVICE="cpu"
$env:AUTO_CPU_OPTIMIZATIONS="true"
$env:FACE_DETECTOR_BACKEND="insightface"
$env:INFERENCE_BATCH_SIZE="4"
$env:INFERENCE_TTA="adaptive"
$env:VIDEO_CHUNK_SIZE="16"
$env:MAX_VIDEO_JOBS="1"
$env:API_INFERENCE_WORKERS="2"
$env:OMP_NUM_THREADS="8"
$env:MKL_NUM_THREADS="8"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 5
```

Linux/macOS equivalent:

```bash
DEVICE=cpu AUTO_CPU_OPTIMIZATIONS=true FACE_DETECTOR_BACKEND=insightface \
INFERENCE_BATCH_SIZE=4 INFERENCE_TTA=adaptive VIDEO_CHUNK_SIZE=16 \
MAX_VIDEO_JOBS=1 API_INFERENCE_WORKERS=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 5
```

## Benchmark

Prepare 100 images as 50 real and 50 fake plus up to 10 short videos:

```powershell
python scripts/benchmark_cpu.py `
  --real-dir data/bench/real `
  --fake-dir data/bench/fake `
  --videos-dir data/bench/videos `
  --image-limit 100 `
  --video-limit 10 `
  --max-frames 32 `
  --chunk-size 16 `
  --batch-size 4
```

Output:

- `reports/benchmark/cpu_benchmark.json`
- `reports/benchmark/cpu_benchmark.md`

The benchmark compares `insightface`, `mtcnn`, and `haar`, plus `false`, `adaptive`, and `true` TTA modes for image inference.

## Docker CPU

```bash
docker compose build api
docker compose up api
```

The default Dockerfile is CPU-only: it installs CPU PyTorch, `onnxruntime`, and `insightface`, and starts Uvicorn with one worker and `--limit-concurrency 5`.

## CPU Troubleshooting

- `MTCNN` is very slow on CPU. Prefer `FACE_DETECTOR_BACKEND=insightface`; use `haar` only when speed matters more than crop quality.
- If ONNXRuntime reports a missing provider, force CPU by using InsightFace with `providers=["CPUExecutionProvider"]`. DeepGuard does this in `FaceDetector`.
- For many videos, keep `MAX_VIDEO_JOBS=1`, `VIDEO_CHUNK_SIZE=16`, and `max_frames=32`. The video processor streams chunks and releases the OpenCV capture in `finally`.
- TTA is expensive on CPU. Use `INFERENCE_TTA=adaptive` for normal serving and `INFERENCE_TTA=false` under high load.
