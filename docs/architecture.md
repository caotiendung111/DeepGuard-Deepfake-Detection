# DeepGuard — Kiến trúc Hệ thống

## Tổng quan kiến trúc

```
┌────────────────────────────────────────────────────────────────────┐
│                         DeepGuard System                            │
├────────────────────────┬───────────────────────────────────────────┤
│      Input Layer        │           Processing Pipeline              │
├────────────────────────┼───────────────────────────────────────────┤
│  Image (JPG/PNG)  ──── │──► Face Detection (MTCNN)                 │
│                         │         │                                  │
│  Video (MP4/AVI)  ──── │──► Frame Extraction (OpenCV)              │
│                         │         │                                  │
│                         │    Face Crop + Resize (224×224)           │
│                         │         │                                  │
│                         │    Augmentation (albumentations)           │
├────────────────────────┼───────────────────────────────────────────┤
│     Model Layer         │        Deep Learning Model                 │
├────────────────────────┼───────────────────────────────────────────┤
│                         │  ┌─────────────────────────────────────┐  │
│                         │  │  EfficientNet-B4 (Backbone)          │  │
│                         │  │  • Pretrained on ImageNet           │  │
│                         │  │  • Global Average Pooling           │  │
│                         │  │  • Feature dim: 1,792               │  │
│                         │  └────────────────┬────────────────────┘  │
│                         │                   │                        │
│                         │  ┌────────────────▼────────────────────┐  │
│                         │  │  Classification Head                 │  │
│                         │  │  BN → Dropout(0.3) → Linear(512)   │  │
│                         │  │  → GELU → BN → Dropout → Linear(1) │  │
│                         │  └────────────────┬────────────────────┘  │
│                         │                   │                        │
│                         │              Sigmoid → P(fake)            │
├────────────────────────┼───────────────────────────────────────────┤
│    Output Layer         │          Results & Explainability          │
├────────────────────────┼───────────────────────────────────────────┤
│  Label: REAL/FAKE       │  ← Threshold comparison (default: 0.5)    │
│  Probability (0-1)      │  ← Sigmoid output                         │
│  Grad-CAM Heatmap       │  ← Gradient-weighted class activation map │
│  Frame Timeline         │  ← Per-frame analysis for videos          │
└────────────────────────┴───────────────────────────────────────────┘
```

## Training Pipeline

```
Raw Videos/Images
       │
       ▼
Data Preprocessing (scripts/preprocess.py)
 • MTCNN face detection
 • Frame extraction (fps_sample=3)
 • Resize to 224×224
 • Train/Val/Test split (70/15/15)
       │
       ▼
PyTorch DataLoader
 • Albumentations augmentation (train)
 • Normalize (ImageNet stats)
       │
       ▼
EfficientNet-B4 Fine-tuning
 • Backbone: pretrained ImageNet
 • Loss: Focal Loss (α=0.25, γ=2.0)
 • Optimizer: AdamW (lr=5e-5)
 • Scheduler: CosineAnnealingLR
 • Mixed Precision (AMP)
 • Gradient Clipping (max_norm=1.0)
       │
       ▼
MLflow Experiment Tracking
 • Loss, AUC-ROC, F1 per epoch
 • Best checkpoint saved
 • Hyperparameter logging
       │
       ▼
Best Model Checkpoint
(models/checkpoints/best_model.pth)
```

## Inference Pipeline

```
Input (Image/Video)
       │
       ├── Image Path → PIL.Image.open()
       ├── np.ndarray → direct use
       └── Video File → VideoProcessor.extract_frames()
              │
              ▼
       Face Detection (MTCNN)
       Crop + Resize (224×224)
              │
              ▼
       Albumentations Val Transform
       (Resize + Normalize + ToTensor)
              │
              ▼
       EfficientNet-B4 Forward Pass
              │
              ▼
       Sigmoid → P(fake)
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
    Single Image   Video
    P(fake) ──►  frame_probs[]
    Label          └── mean/max/vote
                   └── P(fake) → Label
              │
              ▼
       GradCAM Heatmap (optional)
       (explains which regions → decision)
```

## Component Overview

| Component | File | Responsibility |
|-----------|------|----------------|
| Dataset | `src/data/dataset.py` | PyTorch Dataset, CSV loading |
| Transforms | `src/data/transforms.py` | albumentations pipelines |
| VideoProcessor | `src/data/video_processor.py` | Frame extraction, face crop |
| EfficientNetDetector | `src/models/efficientnet.py` | Backbone + head |
| XceptionDetector | `src/models/xception.py` | Alternative backbone |
| Trainer | `src/training/trainer.py` | Training loop + MLflow |
| FocalLoss | `src/training/losses.py` | Class imbalance handling |
| MetricTracker | `src/training/metrics.py` | AUC, F1, confusion matrix |
| ImagePredictor | `src/inference/predictor.py` | Single image inference |
| VideoPredictor | `src/inference/predictor.py` | Video inference pipeline |
| GradCAMVisualizer | `src/inference/gradcam.py` | Explainability heatmap |
| FastAPI | `api/main.py` + `routes/` | REST API endpoints |
| Gradio Demo | `app/gradio_demo.py` | Interactive web UI |
| Streamlit Demo | `app/streamlit_demo.py` | Alternative web UI |
