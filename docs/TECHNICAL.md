# Báo cáo Kỹ thuật - DeepGuard
Tài liệu này giải thích chi tiết về các quyết định thiết kế kiến trúc, tham số siêu việt (hyperparameters) và các bài học rút ra trong quá trình phát triển hệ thống phát hiện Deepfake DeepGuard.

## 1. Kiến trúc Mô hình (Model Architecture)

### Tại sao lại chọn EfficientNet-B4?
Sau nhiều thử nghiệm với ResNet-50, Xception và ViT, chúng tôi quyết định sử dụng **EfficientNet-B4** làm Backbone chính vì các lý do sau:
1. **Hiệu năng không gian**: EfficientNet sử dụng Compound Scaling (cân bằng giữa chiều sâu, chiều rộng và độ phân giải mạng) giúp nó có khả năng trích xuất đặc trưng cực tốt ở mức độ pixel - rất quan trọng để tìm ra các "artifacts" tinh vi do Deepfake để lại.
2. **Kích thước nhẹ**: Tham số chỉ khoảng 17.5M (nhỏ hơn Xception ~20M và ResNet-50 ~23M), giúp tăng tốc độ Inference trên CPU/Edge devices mà vẫn giữ được độ chính xác cao.

### Thiết kế Classifier Head
Đầu ra của EfficientNet (Global Average Pooling) là một vector 1792 chiều. Chúng tôi không nối thẳng vào lớp Linear 1-node vì rất dễ bị Overfitting. Thay vào đó, Classifier Head được thiết kế như sau:
```python
nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(1792, 512),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, 1)
)
```
Thiết kế hình "phễu" này kết hợp với Dropout kép giúp mô hình rèn luyện tính tổng quát hóa (generalization) tốt hơn trên các tệp dữ liệu Deepfake chưa từng gặp (Cross-dataset evaluation).

## 2. Các Quyết định Kỹ thuật Quan trọng

### a. Tiền xử lý Dữ liệu (Preprocessing)
- **Face Detection (MTCNN)**: Không đưa cả bức ảnh vào mô hình vì background sẽ làm nhiễu. Hệ thống chỉ detect và cắt khuôn mặt.
- **Padding 20%**: Một số kỹ thuật Deepfake chỉ làm giả vùng "mũi, miệng, mắt" (face swapping) và để lại vết khâu ở vùng cằm và trán. Việc padding 20% giúp mô hình "nhìn thấy" viền khuôn mặt để phát hiện các vết khâu này.

### b. Data Augmentation
Dữ liệu trên mạng thường bị nén (Facebook/Tiktok) dẫn đến mất chi tiết. Việc sử dụng `Albumentations` với `ImageCompression`, `GaussNoise` và `CoarseDropout` giúp mô hình học cách không phụ thuộc quá nhiều vào các pixel sắc nét hoàn hảo, mô phỏng lại môi trường thực tế.

### c. Xử lý Video (Inference Async)
Để xử lý video dài (100MB), việc chờ đợi Synchronous là không khả thi (nguy cơ HTTP Timeout). Chúng tôi đã chuyển sang kiến trúc Async Job:
1. Client gửi Video -> Nhận `job_id`.
2. Server tạo Background Task chạy cắt khung hình (1 FPS) và dự đoán.
3. Client dùng Polling `GET /predict/video/{job_id}` để vẽ biểu đồ diễn biến.
Phương pháp Aggregation được áp dụng là: **Mean Confidence** + **Majority Voting**.

## 3. Nhật ký tinh chỉnh Hyperparameters (Tuning Log)

| Thử nghiệm | Loss Function | Optimizer | Lr Scheduler | AUC-ROC | Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | BCEWithLogits | Adam | StepLR | 0.920 | Mô hình học nhanh nhưng chững lại sớm, overfit. |
| Exp 2 | BCEWithLogits | AdamW | CosineAnnealing | 0.945 | Thay đổi Lr Scheduler giúp hội tụ mượt mà hơn. |
| Exp 3 | Focal Loss (γ=2) | AdamW | CosineAnnealing | 0.970 | **Đột phá**: Focal Loss xử lý cực tốt dữ liệu mất cân bằng và tập trung vào các ca "Hard-negatives". |
| Final | Focal Loss (γ=2) | AdamW | Linear Warmup + Cosine | 0.985 | Thêm 5 epoch Warmup giúp tránh "sốc" gradient đầu kỳ, model hội tụ ổn định. |

## 4. Bài học kinh nghiệm (Lessons Learned)

1. **Đừng log mô hình sau mỗi epoch**: Model nặng ~80MB. Việc dùng `mlflow.pytorch.log_model` sau mỗi epoch sẽ làm đầy ổ cứng cực nhanh. Giải pháp: Chỉ lưu đè file `best_model.pth` cục bộ, và gọi lệnh upload lên MLflow 1 lần duy nhất ở cuối pipeline.
2. **Kích thước Batch vs Kích thước Ảnh**: Tăng kích thước ảnh lên `299x299` có thể giúp Xception tăng độ chính xác nhưng lại giảm Batch Size làm thời gian train lâu gấp đôi. `224x224` là điểm cân bằng ngọt ngào (sweet spot).
3. **Giải thích được (XAI) là điểm ăn tiền**: Phân loại đúng/sai chỉ là những con số. Việc ứng dụng `Captum LayerGradCam` để vẽ ra một Heatmap chính xác nơi mô hình phát hiện lỗi bóp méo giúp tăng tính thuyết phục của hệ thống lên gấp nhiều lần.

## 5. Hướng nâng cấp tương lai (Future Enhancements)

Để hệ thống hoàn thiện hơn nữa và sẵn sàng chống lại các kỹ thuật Deepfake "thế hệ mới" (GenAI), chúng tôi đề xuất 4 hướng phát triển chính:

1. **Temporal Modeling cho Video (Mô hình thời gian)**
   - Hiện tại, video đang được xử lý theo kiểu "Frame-by-frame" (từng khung hình riêng lẻ) rồi Vote. Điều này bỏ qua sự liên kết thời gian (VD: nháy mắt không tự nhiên, viền mép bị trượt khi nói).
   - *Giải pháp*: Kết hợp `LSTM` hoặc `GRU` (Recurrent Neural Networks) đè lên trên (on top of) các vector đặc trưng của EfficientNet, hoặc dùng mạng `3D-CNN (I3D)` để phân tích sự thay đổi không gian-thời gian cùng lúc.
   
2. **Multimodal Detection (Nhận diện đa phương thức)**
   - Deepfake audio (giả mạo giọng nói) đang ngày càng tinh vi. Việc chỉ nhìn hình ảnh là không đủ nếu video bị chèn giọng nói ghép bởi AI.
   - *Giải pháp*: Xây dựng một luồng phụ xử lý phổ âm thanh (Mel-spectrogram) kết hợp với luồng hình ảnh. Kiểm tra độ lệch pha giữa chuyển động môi (Lip-sync) và luồng âm thanh.

3. **Self-supervised Pre-training**
   - Dữ liệu Deepfake gán nhãn rất tốn kém để thu thập.
   - *Giải pháp*: Dùng hàng triệu video không nhãn trên YouTube/Tiktok để pre-train mô hình bằng kỹ thuật Self-Supervised Learning (VD: Masked Autoencoders), giúp mô hình tự học "thế nào là một khuôn mặt thật và tự nhiên" trước khi fine-tune vào bài toán phát hiện giả mạo.

4. **Adversarial Training (Huấn luyện đối kháng)**
   - Kẻ tấn công có thể cố tình chèn nhiễu vô hình (Adversarial attacks) vào ảnh/video để đánh lừa mô hình của chúng ta.
   - *Giải pháp*: Áp dụng kỹ thuật Adversarial Training - chủ động tạo ra các mẫu bị tấn công và ép mô hình phải nhận diện đúng, giúp tăng tính bền vững (Robustness) lên mức tối đa.
