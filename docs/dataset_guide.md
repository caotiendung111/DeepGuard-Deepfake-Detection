# DeepGuard — Hướng dẫn Dataset

## Các Dataset được hỗ trợ

### 1. FaceForensics++ (Khuyến nghị)

**Link:** https://github.com/ondyari/FaceForensics  
**Kích thước:** ~1.5 TB (raw) | ~15 GB (compressed)  
**Labels:** Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter

**Cách tải:**
```bash
# Cài tools
pip install requests
git clone https://github.com/ondyari/FaceForensics

# Yêu cầu Google Form để lấy link tải
# https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform

python download-FaceForensics.py \
    /path/to/output \
    -d all \         # all methods
    -c c23 \         # compression level (c0=raw, c23=light, c40=heavy)
    -t videos        # videos or images
```

**Cấu trúc sau khi tải:**
```
FaceForensics++/
├── original_sequences/actors/raw/videos/   → REAL
├── manipulated_sequences/Deepfakes/raw/videos/  → FAKE
├── manipulated_sequences/Face2Face/raw/videos/  → FAKE
└── ...
```

### 2. DFDC (DeepFake Detection Challenge)

**Link:** https://ai.facebook.com/datasets/dfdc/  
**Kích thước:** ~470 GB  
**Đặc điểm:** Multi-person, high diversity

```bash
# Đăng ký tại Kaggle
kaggle competitions download -c deepfake-detection-challenge
```

### 3. Celeb-DF v2 (Lightweight, khuyến nghị để bắt đầu)

**Link:** https://github.com/yuezunli/celeb-deepfakeforensics  
**Kích thước:** ~2.7 GB  
**Nội dung:** 590 real + 5639 deepfake videos

```bash
# Yêu cầu điền Google Form
# https://docs.google.com/forms/d/e/...
```

---

## Cấu trúc thư mục cần thiết

```
data/
├── raw/
│   ├── real/          ← Video/ảnh thật
│   │   ├── vid001.mp4
│   │   └── ...
│   └── fake/          ← Video/ảnh deepfake
│       ├── vid001.mp4
│       └── ...
└── processed/         ← Tự động tạo bởi preprocess.py
    ├── real/
    └── fake/
```

---

## Tiền xử lý dữ liệu

```bash
# Bước 1: Đặt video vào data/raw/real/ và data/raw/fake/
# Bước 2: Chạy preprocessing
python scripts/preprocess.py \
    --data-dir data/raw \
    --output-dir data/processed \
    --face-size 224 \
    --fps-sample 3 \
    --val-split 0.15 \
    --test-split 0.15
```

Pipeline xử lý:
1. Đọc từng video → trích xuất frames theo FPS sampling
2. MTCNN phát hiện khuôn mặt → crop + resize về 224×224
3. Lưu frames dưới dạng JPEG
4. Tạo file CSV `train.csv`, `val.csv`, `test.csv`

---

## Khuyến nghị tỷ lệ Real/Fake

| Dataset | Real | Fake | Ratio |
|---------|------|------|-------|
| FF++ c23 | 1,000 | 4,000 | 1:4 |
| DFDC | ~20K | ~100K | 1:5 |
| Celeb-DF | 590 | 5,639 | 1:9.5 |

> **Lưu ý:** Dữ liệu mất cân bằng! Sử dụng `FocalLoss` hoặc `WeightedBCELoss`  
> trong `src/training/losses.py` để xử lý class imbalance.
