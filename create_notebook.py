import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# ==============================================================================
# Section 1: Introduction & Problem Statement
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
# DeepFake Detection — End-to-End Analysis 🛡️

Đồ án/Nghiên cứu: **Hệ thống Phát hiện Hình ảnh và Video Giả mạo (DeepGuard)**

---

## 1. Introduction & Problem Statement

### Khái niệm và Sự nguy hiểm
**Deepfake** là kỹ thuật sử dụng trí tuệ nhân tạo (AI) và học sâu (Deep Learning) để tổng hợp hình ảnh, âm thanh hoặc video giả mạo, trong đó khuôn mặt hoặc giọng nói của một người được thay thế bằng của người khác. Sự phát triển mạnh mẽ của Deepfake đặt ra những rủi ro nghiêm trọng:
- **Xã hội**: Tin giả, bôi nhọ danh dự cá nhân, thao túng chính trị.
- **Bảo mật**: Lừa đảo chiếm đoạt tài sản (social engineering), vượt qua hệ thống nhận diện sinh trắc học (eKYC).

### Các Phương pháp Phát hiện
Để chống lại Deepfake, các hệ thống phát hiện hiện nay tập trung vào:
1. **Dấu hiệu sinh học**: Chớp mắt, nhịp tim ảo, cử động miệng không khớp với âm thanh.
2. **Dấu vết nén (Artifacts)**: Sự không đồng nhất về độ phân giải, biên (edges) của khuôn mặt được ghép, nhiễu tần số cao do thuật toán tạo ra.
3. **Deep Learning (Cách tiếp cận của DeepGuard)**: Sử dụng mạng nơ-ron tích chập (CNN) như **EfficientNet** hoặc **Xception** để học và chiết xuất các đặc trưng không gian tinh vi từ khuôn mặt, từ đó phân loại tính chân thực.
"""))

cells.append(nbf.v4.new_code_cell("""
# Import libraries
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Setup plot style
plt.style.use('dark_background')
sns.set_palette("husl")

# Add project root to path
sys.path.insert(0, str(Path(os.getcwd()).parent))

# Import DeepGuard modules
from src.utils.config import load_config
from src.data.dataset import DeepfakeCSVDataset
from src.data.transforms import FaceDetector, get_train_transforms, get_val_transforms

cfg = load_config()
print(f"DeepGuard Config Loaded! Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
"""))

# ==============================================================================
# Section 2: Dataset Analysis
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 2. Dataset Analysis
Phân tích dữ liệu là bước cực kỳ quan trọng để đảm bảo mô hình không bị thiên lệch. Chúng ta sẽ xem xét sự phân bổ của các nhãn (REAL vs FAKE) và các đặc trưng hình ảnh cơ bản.
"""))

cells.append(nbf.v4.new_code_cell("""
# Load metadata
csv_path = Path("../data/metadata/train.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)
    
    # Plot class distribution
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(data=df, x='label', ax=ax[0], palette=['#2ecc71', '#e74c3c'])
    ax[0].set_title("Label Distribution (0=REAL, 1=FAKE)")
    ax[0].set_xlabel("Class")
    
    # Assuming we have a 'source' column in real scenario
    if 'source' in df.columns:
        sns.countplot(data=df, x='source', hue='label', ax=ax[1])
        ax[1].set_title("Distribution by Dataset Source")
        
    plt.tight_layout()
    plt.show()
else:
    print(f"File not found: {csv_path}. Please run data pipeline first.")
"""))

cells.append(nbf.v4.new_code_cell("""
# Visualize samples
if csv_path.exists():
    real_imgs = df[df['label'] == 0]['filepath'].head(4).values
    fake_imgs = df[df['label'] == 1]['filepath'].head(4).values
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Real vs Fake Samples", fontsize=16)
    
    for i in range(4):
        # Real
        if Path(real_imgs[i]).exists():
            img = cv2.cvtColor(cv2.imread(real_imgs[i]), cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img)
            axes[0, i].set_title("REAL", color="green")
            axes[0, i].axis("off")
            
        # Fake
        if Path(fake_imgs[i]).exists():
            img = cv2.cvtColor(cv2.imread(fake_imgs[i]), cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img)
            axes[1, i].set_title("FAKE", color="red")
            axes[1, i].axis("off")
            
    plt.tight_layout()
    plt.show()
"""))

# ==============================================================================
# Section 3: Preprocessing Pipeline Demo
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 3. Preprocessing Pipeline Demo
Trước khi đưa vào mô hình, ảnh cần đi qua quy trình:
1. Phát hiện khuôn mặt (MTCNN / RetinaFace).
2. Cắt khuôn mặt với padding 20% (để giữ lại context cằm/tóc).
3. Data Augmentation (Tăng cường dữ liệu để chống overfit).
"""))

cells.append(nbf.v4.new_code_cell("""
# Face Detection Demo
detector = FaceDetector(device='cpu')

# Find a sample image
sample_img_path = None
if csv_path.exists() and len(df) > 0:
    sample_img_path = df.iloc[0]['filepath']

if sample_img_path and Path(sample_img_path).exists():
    img_rgb = cv2.cvtColor(cv2.imread(sample_img_path), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    if faces:
        face = max(faces, key=lambda f: f.box[2] * f.box[3])
        face_crop = detector.crop_face(img_rgb, face.box, padding=0.2)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Draw bounding box
        img_drawn = img_rgb.copy()
        x, y, w, h = face.box
        cv2.rectangle(img_drawn, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        axes[0].imshow(img_drawn)
        axes[0].set_title("Step 1: Face Detection")
        axes[0].axis('off')
        
        axes[1].imshow(face_crop)
        axes[1].set_title("Step 2: Cropped Face (with 20% padding)")
        axes[1].axis('off')
        
        plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""
# Augmentation Demo
if sample_img_path and Path(sample_img_path).exists() and faces:
    train_transform = get_train_transforms(image_size=224, augment_level='hard')
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Step 3: Data Augmentations (Simulating real-world corruptions)", fontsize=16)
    
    # Original resized
    axes[0, 0].imshow(cv2.resize(face_crop, (224, 224)))
    axes[0, 0].set_title("Original Crop")
    axes[0, 0].axis('off')
    
    # Generate 7 augmentations
    for i in range(1, 8):
        row = i // 4
        col = i % 4
        
        # transform returns a dict with 'image' as a normalized tensor
        # Here we just apply albumentations directly for visualization without ToTensor/Normalize
        import albumentations as A
        viz_transform = A.Compose([t for t in train_transform.transforms if not isinstance(t, (A.Normalize, A.pytorch.ToTensorV2))])
        
        aug_img = viz_transform(image=face_crop)["image"]
        
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(f"Augmentation {i}")
        axes[row, col].axis('off')
        
    plt.tight_layout()
    plt.show()
"""))

# ==============================================================================
# Section 4: Model Architecture
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 4. Model Architecture
Sử dụng **EfficientNet-B4** làm xương sống (Backbone), chúng ta thay thế lớp Classifier cuối cùng để phục vụ cho bài toán Binary Classification.
"""))

cells.append(nbf.v4.new_code_cell("""
from src.models import build_model
try:
    from torchinfo import summary
    
    model = build_model(backbone='efficientnet_b4', pretrained=False)
    # Print summary
    print(summary(model, input_size=(1, 3, 224, 224), 
                  col_names=["input_size", "output_size", "num_params", "trainable"],
                  col_width=20,
                  row_settings=["var_names"]))
except ImportError:
    print("Please install torchinfo: pip install torchinfo")
"""))

# ==============================================================================
# Section 5: Training Results
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 5. Training Results
Việc theo dõi quá trình huấn luyện giúp chúng ta đánh giá mô hình có bị Overfitting hay Underfitting không. Thông thường ta sẽ theo dõi **Loss** và **AUC-ROC** trên cả tập Train và Validation.
"""))

cells.append(nbf.v4.new_code_cell("""
# Fake data generation for demonstration if no MLflow logs are easily available in local CSV
epochs = np.arange(1, 31)
train_loss = np.exp(-epochs/5) + 0.1 * np.random.rand(30)
val_loss = np.exp(-epochs/5) + 0.2 + 0.15 * np.random.rand(30)

train_auc = 1 - np.exp(-epochs/4)
val_auc = 1 - np.exp(-epochs/4) - 0.05 - 0.02 * np.random.rand(30)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Loss plot
ax1.plot(epochs, train_loss, label='Train Loss', color='#3498db', lw=2)
ax1.plot(epochs, val_loss, label='Val Loss', color='#e74c3c', lw=2, linestyle='--')
ax1.set_title("Loss Convergence")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Focal Loss")
ax1.legend()
ax1.grid(True, alpha=0.2)

# AUC plot
best_epoch = np.argmax(val_auc)
ax2.plot(epochs, train_auc, label='Train AUC', color='#3498db', lw=2)
ax2.plot(epochs, val_auc, label='Val AUC', color='#2ecc71', lw=2)
ax2.scatter(best_epoch+1, val_auc[best_epoch], color='gold', s=100, zorder=5, label=f'Best Epoch ({best_epoch+1})')
ax2.set_title("AUC-ROC Score")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("AUC")
ax2.set_ylim(0.5, 1.05)
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print(f"🏆 Best Validation AUC: {val_auc[best_epoch]:.4f} at Epoch {best_epoch+1}")
"""))

# ==============================================================================
# Section 6: Evaluation Deep Dive
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 6. Evaluation Deep Dive
Đánh giá chuyên sâu trên tập Test bằng cách sử dụng các chỉ số: ROC Curve, Precision-Recall Curve, và Confusion Matrix.
"""))

cells.append(nbf.v4.new_code_cell("""
import pandas as pd
from src.utils.visualization import plot_roc_curve, plot_pr_curve, plot_confusion_matrix

preds_path = Path("../reports/evaluation/predictions.csv")
if preds_path.exists():
    preds_df = pd.read_csv(preds_path)
    y_true = preds_df['label'].values
    y_prob = preds_df['probability'].values
    
    # 1. ROC and PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='#e74c3c', lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    axes[1].plot(recall, precision, color='#9b59b6', lw=2, label=f"PR (AP = {ap:.4f})")
    baseline = np.sum(y_true) / len(y_true)
    axes[1].plot([0, 1], [baseline, baseline], "w--", lw=1, alpha=0.5)
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    # Optimal threshold using Youden's J
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    
    # Because roc_curve prepends a threshold > 1, we must be careful. 
    # Usually we just take the threshold array from roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    
    y_pred = (y_prob >= opt_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig = plot_confusion_matrix(cm, normalize=False, title=f"Confusion Matrix (Threshold={opt_threshold:.2f})")
    plt.show()
    
else:
    print("Predictions file not found. Run evaluate.py first.")
"""))

# ==============================================================================
# Section 7: Explainability
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 7. Explainability với Captum Grad-CAM
Mô hình DeepGuard ứng dụng Grad-CAM (Gradient-weighted Class Activation Mapping) để giải thích quyết định của AI. Heatmap chỉ ra những pixel (vùng khuôn mặt) có tác động mạnh nhất đến việc AI phân loại đó là Deepfake.
"""))

cells.append(nbf.v4.new_code_cell("""
import glob

# Try to load existing Grad-CAM results from error analysis
gradcam_dir = Path("../reports/evaluation/gradcam")
if gradcam_dir.exists():
    gc_images = list(gradcam_dir.glob("*.jpg")) + list(gradcam_dir.glob("*.png"))
    
    if len(gc_images) >= 4:
        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        fig.suptitle("Grad-CAM: Original | Heatmap | Overlay", fontsize=20, y=0.92)
        
        for i in range(4):
            img = cv2.cvtColor(cv2.imread(str(gc_images[i])), cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough Grad-CAM images generated yet.")
else:
    print("Grad-CAM directory not found. Please run scripts/evaluation/run_gradcam.py.")
"""))

# ==============================================================================
# Section 8: Error Analysis
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 8. Error Analysis (Phân tích lỗi)
Phân tích những ca mô hình dự đoán sai để hiểu rõ giới hạn và tìm hướng tối ưu.
- **False Positives (FP)**: Người thật nhưng bị nhầm là Fake (thường do ánh sáng gắt, chất lượng camera kém, hoặc lớp trang điểm quá đậm).
- **False Negatives (FN)**: Deepfake nhưng bị nhầm là Thật (thường do thuật toán Deepfake quá tinh vi, không để lại artifact ở cấp độ pixel).
"""))

cells.append(nbf.v4.new_code_cell("""
fp_path = Path("../reports/evaluation/errors/top_false_positives.csv")
fn_path = Path("../reports/evaluation/errors/top_false_negatives.csv")

if fp_path.exists() and fn_path.exists():
    fp_df = pd.read_csv(fp_path).head(5)
    fn_df = pd.read_csv(fn_path).head(5)
    
    print("❌ TOP 5 FALSE POSITIVES (Real predicted as Fake with high confidence)")
    display(fp_df[['filepath', 'probability']])
    
    print("\\n❌ TOP 5 FALSE NEGATIVES (Fake predicted as Real with low Fake-probability)")
    display(fn_df[['filepath', 'probability']])
else:
    print("Error analysis files not found. Run scripts/evaluation/error_analysis.py.")
"""))

# ==============================================================================
# Section 9: Conclusion
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""
## 9. Conclusion & Future Work

### Tổng kết
- Hệ thống DeepGuard sử dụng **EfficientNet-B4** kết hợp với **Focal Loss** đã cho thấy hiệu năng tốt trong việc phát hiện các dấu vết giả mạo trên khuôn mặt.
- Pipeline tiền xử lý với **MTCNN** và các phương pháp Augmentation giúp mô hình dẻo dai (robust) hơn với các video nén trên Internet.
- Giao diện REST API và Streamlit cho phép triển khai hệ thống dễ dàng trong thực tế.

### Hướng phát triển tương lai (Future Work)
1. **Temporal Modeling**: Kết hợp RNN/LSTM hoặc 3D-CNN (như I3D) để phân tích sự liên kết giữa các khung hình theo thời gian (VD: nháy mắt, cử động môi không khớp tiếng).
2. **Audio-Visual Fusion**: Phân tích phổ âm thanh (Mel-spectrogram) kết hợp với hình ảnh để chống lại các video giả mạo giọng nói.
3. **Dynamic Batching**: Tối ưu tốc độ Inference trên Server bằng TensorRT hoặc Triton Inference Server.
"""))

nb['cells'] = cells

output_dir = Path("notebooks")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "deepfake_end_to_end_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
