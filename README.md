# DeepGuard 🛡️
**AI-Powered Deepfake Detection for Images and Videos**

[![GitHub Actions CI](https://img.shields.io/github/actions/workflow/status/caotiendung111/DeepGuard-Deepfake-Detection/ci.yml?branch=main&logo=github&style=flat-square)](https://github.com/caotiendung111/DeepGuard-Deepfake-Detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&style=flat-square)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&style=flat-square)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/Data_Versioning-DVC-945DD6.svg?logo=data-version-control&style=flat-square)](https://dvc.org/)

![DeepGuard Dashboard Preview](dashboard.png)

## 📖 Overview
DeepGuard is an end-to-end production-ready Deep Learning system designed to detect manipulated images and videos (Deepfakes). 
By leveraging EfficientNet architectures and state-of-the-art Explainable AI (XAI) techniques like Grad-CAM, DeepGuard not only flags fake content but also highlights the exact tampered regions, providing interpretable and trustworthy results.

## 🏗️ System Architecture

The following diagram illustrates the workflow of the DeepGuard detection pipeline, showing the relationship between media preprocessing, face detection, neural net inference, Grad-CAM generation, and the serving interfaces:

```mermaid
graph TD
    Input[Upload Video / Image] --> UI[Streamlit / Gradio UI or FastAPI Endpoint]
    UI --> Prep[Frame Extraction & Preprocessing]
    Prep --> MTCNN[Face Detection & Cropping - MTCNN]
    MTCNN --> Model[Deep Learning Inference - EfficientNet / Xception]
    Model --> Post[Post-processing & Softmax Activation]
    Post --> Output[Real/Fake Probability & Grad-CAM Heatmaps]
    Output --> UI
```

## 📊 Performance Benchmarks
Evaluated on a combined test set of FaceForensics++ and Celeb-DF:

| Model | Parameters (M) | AUC-ROC | F1 Score | Latency (ms/img) |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B4 (DeepGuard)** | 17.5 | **0.985** | **0.962** | 24.5 |
| Xception | 20.8 | 0.971 | 0.940 | 31.2 |
| ResNet-50 | 23.5 | 0.945 | 0.912 | **18.4** |

## 🚀 Quick Start
Get DeepGuard running on your local machine in 5 simple steps.

**Step 1: Clone the repository**
```bash
git clone https://github.com/yourusername/DeepGuard.git
cd DeepGuard
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Download sample data & pre-trained weights**
*(Make sure to place `best_model.pth` in `models/checkpoints/`)*

**Step 4: Spin up the system using Docker Compose**
```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

**Step 5: Access the Web UI**
Open your browser and navigate to `http://localhost:8501`. 
The REST API is available at `http://localhost:8000/docs`.

---

## 🛠️ Usage Guide

### 1. Training a new model
To train the model from scratch on your own dataset:
```bash
python scripts/data/build_kaggle_metadata.py --data-root data/raw/deepfake-and-real-images --output-dir data/metadata
python scripts/train.py --config configs/base.yaml --epochs 30
# Or use make shortcut
make data-kaggle-metadata
make train
```

After training, evaluate on the labeled test split to regenerate the calibrated
production threshold in `configs/thresholds.yaml`:
```bash
python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth --test-csv data/metadata/test.csv
```

### 2. Data Version Control (DVC)
To reproduce the entire pipeline (data collection -> preprocessing -> training -> evaluation) using DVC:
```bash
# Pull datasets and models from Google Drive remote
dvc pull

# Run the pipeline
dvc repro
```

### 3. Evaluation & Error Analysis
Evaluate your model and generate a comprehensive PDF report:
```bash
make eval
# Run robustness tests (JPEG compression, Noise)
make eval-robust
```

Run the external real/fake image self-test:
```bash
python scripts/external_self_test.py --mode download
python scripts/external_self_test.py --mode api --api-url http://localhost:8000
```

### 3. Running the REST API locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production inference controls

DeepGuard exposes several inference knobs in `configs/base.yaml`:

- `face_detector_backend`: `mtcnn`, `insightface`, `haar`, or `auto`. `auto`
  tries InsightFace first, then MTCNN, then Haar.
- `inference_tta`: `false`, `true`, or `adaptive`. Request form field
  `use_tta` overrides this per image/video request.
- `video_chunk_size`: number of sampled face crops predicted per streaming
  video chunk. Default: `32`.
- `face_cache_gap`: maximum frame-index gap for reusing the previous detected
  face box when detection temporarily fails. Default: `5`.
- `adaptive_tta_threshold_low` / `adaptive_tta_threshold_high`: low-confidence
  probability band for adaptive TTA. Defaults: `0.4` and `0.6`.

Optional InsightFace backend:

```bash
pip install insightface onnxruntime-gpu
```

Use `onnxruntime` instead of `onnxruntime-gpu` on CPU-only machines. The first
InsightFace startup may download the `buffalo_l` model package into the local
InsightFace cache, so expect a slower first boot.

### Testing and benchmarking

Recommended local environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest httpx
```

For GPU inference, use a Python/PyTorch build matching your CUDA runtime. The
production Dockerfile uses CUDA 11.8.

Run tests:

```bash
pytest tests/ -v
```

Start the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Run benchmark:

```bash
python scripts/benchmark.py --images-dir data/sample --videos-dir data/videos
```

Benchmark outputs:

- `reports/benchmark.json`: raw per-file timings and system snapshots.
- `reports/benchmark.md`: compact Markdown tables.

Important metrics to watch:

- image `avg_s`, `p95_s`: user-facing latency.
- video `frames_analyzed` and `latency_s`: chunk processing cost.
- throughput `requests_per_second`: sustained API capacity.
- `/metrics` GPU memory gauges: whether batch size or TTA is too aggressive.
- 5xx count and request duration histogram in Prometheus.

### Docker production

Build and run API:

```bash
docker compose up --build api
```

Run API with Prometheus and Grafana:

```bash
docker compose --profile monitoring up --build
```

Endpoints:

- API: `http://localhost:8000`
- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin`)

Useful environment variables:

- `CONFIG_PATH`
- `MODEL_CHECKPOINT_PATH`
- `DEVICE`
- `FACE_DETECTOR_BACKEND`
- `INFERENCE_TTA`
- `LOG_LEVEL`
- `LOG_ROTATION`
- `LOG_RETENTION`

## 📁 Repository Structure

```text
DeepGuard/
├── api/                  # FastAPI REST API source code
├── app/                  # Streamlit Web UI source code
├── configs/              # YAML configuration files (hyperparameters)
├── data/                 # Raw datasets, frames, and metadata
├── docker/               # Dockerfiles and docker-compose.yml
├── docs/                 # Detailed technical documentation
├── models/               # Saved checkpoints and weights
├── notebooks/            # Jupyter notebooks for EDA and End-to-End analysis
├── reports/              # Generated PDF reports, ROC curves, Error analysis
├── scripts/              # Training, evaluation, and data pipeline scripts
└── src/                  # Core source code
    ├── data/             # Dataset classes, Datasets, Augmentations
    ├── inference/        # Inference logic and Grad-CAM
    ├── models/           # PyTorch architectures (EfficientNet, Xception)
    ├── training/         # Custom Trainer loop, Losses, Metrics
    └── utils/            # Configurations, Logging, Visualization
```

---

## 📈 Known Limitations & Future Improvements

To guide development and document algorithmic foresight, we outline the primary technical limitations of DeepGuard and the roadmap for scaling to production:

- **Lack of Sequential Temporal Modeling (Frame-by-Frame Processing)**:
  DeepGuard treats video classification as a spatial problem, analyzing frames independently and averaging scores. It cannot capture sequential abnormalities like mouth-to-audio sync errors, sub-frame flickering, or pixel jitter across time. 
  *Improvement Roadmap*: Integrate a hybrid spatial-temporal model by adding a **Video Transformer** (e.g., Vivit) or feeding the frame embedding sequence into a recurrent structure (e.g., **Bi-LSTM** or **GRU**).
- **Domain Generalization & Dataset Bias**:
  Models trained primarily on FaceForensics++ and Celeb-DF exhibit slight accuracy degradation when exposed to raw compression algorithms and compression codecs used on active social media sites (e.g., TikTok, Instagram, Facebook).
  *Improvement Roadmap*: Enrich data preprocessing with adversarial augmentation filters, including variable H.264/H.265 compression artifacts, Gaussian noise, and dynamic lighting shifts.
- **Inference Latency vs. Edge Deployment**:
  The EfficientNet-B4 model has 17.5M parameters and requires substantial computational power to process full 1080p video streams at high FPS.
  *Improvement Roadmap*: Compile and export PyTorch model checkpoints to **ONNX Runtime** format with FP16 mixed precision or utilize **TensorRT** quantization to deploy on low-power edge machines or browser clients.

---

## 📚 Acknowledgements & Citations
This project utilizes the following public datasets for training and evaluation. If you use this work for academic research, please cite the original authors:
- **FaceForensics++**: Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019.
- **Celeb-DF**: Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
