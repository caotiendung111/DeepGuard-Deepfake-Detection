# ============================================================
# DeepGuard — Makefile
# Common development shortcuts
# ============================================================

.PHONY: install install-dev train eval api demo test lint format clean docker-build docker-up data-kaggle-metadata self-test-download self-test-api

# --- Setup ---
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# --- Training ---
train:
	python scripts/train.py --config configs/efficientnet_b4.yaml

train-xception:
	python scripts/train.py --config configs/xception.yaml

# --- Data Pipeline ---
data-download-celebdf:
	python scripts/data/download_datasets.py --dataset celebdf --output-dir data/raw

data-download-ff:
	python scripts/data/download_datasets.py --dataset ff++ --output-dir data/raw --compression c23

data-extract-frames:
	python scripts/data/extract_frames.py --input-dir data/raw --output-dir data/frames --fps 1 --workers 4

data-detect-faces:
	python scripts/data/detect_faces.py --input-dir data/frames --output-dir data/faces --backend mtcnn

data-stats:
	python scripts/data/dataset_stats.py --data-dir data/faces --output-dir reports/dataset

data-kaggle-metadata:
	python scripts/data/build_kaggle_metadata.py --data-root data/raw/deepfake-and-real-images --output-dir data/metadata

data-split:
	python scripts/data/split_dataset.py --data-dir data/faces --output-dir data/metadata

data-pipeline:
	python scripts/data/run_pipeline.py --raw-dir data/raw --fps 1 --face-size 224 --workers 4

data-check-augment:
	python scripts/data/check_augmentations.py --data-dir data/faces --output outputs/aug_check.png --n 16

data-check-quality:
	python scripts/data/check_quality.py --data-dir data/faces --output-dir reports/quality

# --- Preprocessing (legacy) ---
preprocess:
	python scripts/preprocess.py --data-dir data/raw --output-dir data/processed

# --- API Server ---
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# --- Demo UI ---
demo-gradio:
	python app/gradio_demo.py

demo-streamlit:
	streamlit run app/streamlit_demo.py --server.port 8501

# --- MLflow UI ---
mlflow-ui:
	mlflow ui --port 5000

# --- Testing ---
test:
	pytest tests/ -v --cov=src --cov-report=html

test-api:
	pytest tests/test_api.py -v

self-test-download:
	python scripts/external_self_test.py --mode download

self-test-api:
	python scripts/external_self_test.py --mode api --api-url http://localhost:8000

# --- Code Quality ---
lint:
	flake8 src/ api/ scripts/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ api/ scripts/ tests/ --line-length=100
	isort src/ api/ scripts/ tests/

# --- Export ---
export-onnx:
	python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth

# --- Docker ---
docker-build:
	docker build -f docker/Dockerfile -t deepguard:latest .

docker-build-gpu:
	docker build -f docker/Dockerfile.gpu -t deepguard:gpu .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# --- Evaluation ---
eval:
	python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth --test-csv data/metadata/test.csv

eval-error:
	python scripts/evaluation/error_analysis.py

eval-gradcam:
	python scripts/evaluation/run_gradcam.py --checkpoint models/checkpoints/best_model.pth --input-dir reports/evaluation/errors/false_positives

eval-robust:
	python scripts/evaluation/robustness_test.py --checkpoint models/checkpoints/best_model.pth

# --- Cleanup ---
clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf src/*/__pycache__
	rm -rf outputs/*age
