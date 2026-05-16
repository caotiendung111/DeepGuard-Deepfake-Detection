# Changelog

Tất cả những thay đổi đáng chú ý của dự án DeepGuard sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án tuân thủ theo [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-05-12
### Đã thêm (Added)
- **Explainable AI (XAI)**: Tích hợp thư viện `captum` thay thế cho thư viện cũ. Thêm class `GradCAMVisualizer` sử dụng `LayerGradCam` để xuất ra ảnh Heatmap 3-Panel giải thích kết quả của mô hình.
- **REST API Nâng cao**: Xây dựng toàn bộ hệ thống API bằng FastAPI với kiến trúc Đồng bộ cho Ảnh và Bất đồng bộ (Async Job Polling) cho Video. Tích hợp Rate Limiting (`slowapi`) và xử lý lỗi (Exception Handling) chuẩn chỉnh.
- **Web UI (Streamlit)**: Xây dựng giao diện Demo tương tác trực quan 100% tiếng Việt với 4 chức năng: Quét ảnh tĩnh (kèm Heatmap), Quét video (kèm Plotly Timeline rà soát frame), Xử lý lô (Batch) và Tab Hiểu Mô hình.
- **Jupyter Notebook Report**: Sinh file notebook chuẩn khoa học `deepfake_end_to_end_analysis.ipynb` phục vụ cho việc thuyết trình và bảo vệ đồ án.
- **Documentation**: Cập nhật toàn bộ tài liệu dự án bao gồm `README.md`, `TECHNICAL.md` và `API.md`.

## [1.1.0] - 2026-05-11
### Đã thêm (Added)
- **Hỗ trợ xử lý Video**: Script trích xuất khung hình tự động từ Video MP4/AVI bằng OpenCV.
- **Evaluation Module**: Xây dựng công cụ phân tích tự động `evaluate.py`. Tính toán Youden's J threshold, xuất báo cáo PDF gồm ROC, PR Curve, và Confusion Matrix.
- **Error Miner**: Hệ thống trích xuất Top False Positives và False Negatives (`error_analysis.py`) giúp đánh giá điểm mù của mô hình.
- **Robustness Testing**: Tích hợp các module tự động phá hoại dữ liệu (JPEG compress, resize, noise) để đo độ sụt giảm AUC.
- **Benchmark**: Script so sánh đa mô hình đồng thời (EfficientNet vs Xception) về mặt thông số, AUC và Latency (ms).

## [1.0.0] - 2026-05-10
### Đã thêm (Added)
- Khởi tạo kiến trúc dự án DeepGuard ban đầu.
- **Data Pipeline**: Xây dựng luồng xử lý dữ liệu chuẩn từ nguồn tải (FF++, Celeb-DF) -> Extract -> MTCNN Face Detect -> Filter (lọc ảnh mờ/tối).
- **Model Architecture**: Tích hợp EfficientNet-B4 với Custom Classifier Head (Dropout -> Dense -> ReLU -> Dense).
- **Training Pipeline**: Vòng lặp huấn luyện tối ưu hỗ trợ Mixed Precision (AMP), Gradient Clipping, Focal Loss và Cosine Annealing Learning Rate.
- **Tracking**: Tích hợp MLflow theo dõi Hyperparameters và metrics. Lưu giữ mô hình tốt nhất dựa trên Validation AUC.
