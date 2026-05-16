# Tài Liệu REST API - DeepGuard

Hệ thống DeepGuard cung cấp một RESTful API hiệu năng cao được xây dựng trên FastAPI, hỗ trợ xử lý ảnh tĩnh (đồng bộ) và video (bất đồng bộ).

## 1. Thông tin chung
- **Base URL**: `http://localhost:8000` (hoặc URL server của bạn).
- **Swagger UI**: `http://localhost:8000/docs` (Tự động sinh ra tài liệu tương tác).
- **Rate Limiting Policy**: Áp dụng giới hạn số lượng request dựa trên IP để chống Spam/DDoS. Nếu vượt ngưỡng, hệ thống trả về mã lỗi `429 Too Many Requests`.

---

## 2. API Endpoints

### 2.1. System Health Check
Kiểm tra trạng thái hệ thống và cấu hình mô hình đang chạy.
- **Endpoint**: `GET /health`
- **Rate Limit**: 60 requests / phút
- **Ví dụ CURL**:
```bash
curl -X GET "http://localhost:8000/health"
```
- **Response (200 OK)**:
```json
{
  "status": "ok",
  "model": "efficientnet_b4",
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

### 2.2. Nhận diện Ảnh Tĩnh (Image Prediction)
Tải lên một bức ảnh để hệ thống dự đoán và sinh ra ảnh Heatmap giải thích (nếu có).
- **Endpoint**: `POST /predict/image`
- **Rate Limit**: 10 requests / phút
- **Input Rules**: File upload (form-data), hỗ trợ định dạng `.jpg, .png, .webp`. Giới hạn dung lượng: **10MB**.
- **Ví dụ CURL**:
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```
- **Response (200 OK)**:
```json
{
  "label": "FAKE",
  "confidence": 0.985,
  "face_detected": true,
  "processing_time_ms": 145.2,
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### 2.3. Nhận diện Video (Bất đồng bộ)
Do việc quét từng khung hình video mất rất nhiều thời gian, API này áp dụng mô hình Asynchronous. Nó trả về `job_id` lập tức để client tự Polling.
- **Endpoint**: `POST /predict/video`
- **Rate Limit**: 5 requests / phút
- **Input Rules**: File upload (`.mp4, .avi`). Giới hạn dung lượng: **100MB**.
- **Ví dụ CURL**:
```bash
curl -X POST "http://localhost:8000/predict/video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/video.mp4"
```
- **Response (200 OK)**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Video is being processed. Call GET /predict/video/{job_id} to check status."
}
```

### 2.4. Theo dõi trạng thái Video (Polling)
Kiểm tra tiến độ và lấy kết quả xử lý của một Video Job.
- **Endpoint**: `GET /predict/video/{job_id}`
- **Rate Limit**: 30 requests / phút
- **Ví dụ CURL**:
```bash
curl -X GET "http://localhost:8000/predict/video/550e8400-e29b-41d4-a716-446655440000"
```
- **Response (Khi chưa xong - 200 OK)**:
```json
{
  "job_id": "550e8400-...",
  "status": "processing"
}
```
- **Response (Khi hoàn thành - 200 OK)**:
```json
{
  "job_id": "550e8400-...",
  "status": "done",
  "label": "FAKE",
  "confidence": 0.87,
  "frames_analyzed": 42,
  "frame_results": [
    {"frame_index": 0, "label": "REAL", "confidence": 0.12},
    {"frame_index": 1, "label": "FAKE", "confidence": 0.95}
  ],
  "timeline": [0.12, 0.95, ...],
  "processing_time_ms": 15000.0,
  "error": null
}
```

---

## 3. Bảng Mã Lỗi (Error Codes)

Hệ thống xử lý lỗi đồng nhất trả về định dạng JSON: `{"detail": "Error message here"}`.

| Mã lỗi (HTTP Status) | Tên Lỗi | Mô tả & Cách xử lý |
| :---: | :--- | :--- |
| **400** | Bad Request | Logic nghiệp vụ thất bại. (VD: "No face detected in the image"). Bạn cần upload một bức ảnh rõ mặt hơn. |
| **404** | Not Found | Không tìm thấy `job_id` khi polling kết quả video. |
| **413** | Payload Too Large | File upload vượt quá quy định (Ảnh > 10MB hoặc Video > 100MB). Cần nén file trước khi gửi. |
| **422** | Unprocessable Entity | Định dạng file không được hỗ trợ (VD: Upload `.txt` vào endpoint xử lý ảnh). |
| **429** | Too Many Requests | Bạn đã spam API quá mức cho phép của Rate Limiting. Vui lòng chờ 1 phút rồi thử lại. |
| **500** | Internal Server Error | Lỗi xảy ra trong quá trình Inference (Pytorch / Cuda). Hãy kiểm tra log server. |

---
*Tài liệu được tạo tự động cho hệ thống DeepGuard v1.0.0*
