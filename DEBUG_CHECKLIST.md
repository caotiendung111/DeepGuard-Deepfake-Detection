# Checklist Debug Thường Gặp - DeepGuard

Nếu pipeline gặp lỗi trong quá trình chạy, bạn có thể dựa vào danh sách này để khắc phục nhanh chóng:

### 1. Lỗi Huấn luyện / GPU (CUDA Out Of Memory)
- **Triệu chứng:** Xuất hiện lỗi `RuntimeError: CUDA out of memory` hoặc chương trình crash đột ngột khi bắt đầu train/epoch đầu tiên.
- **Khắc phục:** 
  - Mở file cấu hình (vd: `configs/train_config.yaml` hoặc tham số trong `dvc.yaml`).
  - **Giảm `batch_size`** (ví dụ: từ `32` xuống `16` hoặc `8`).
  - Nếu vẫn gặp lỗi OOM, có thể thử giảm kích thước ảnh đầu vào (`image_size`) (ví dụ từ `224` xuống `128` hoặc `112`), hoặc đóng các chương trình đang sử dụng GPU khác.

### 2. Lỗi Không Phát Hiện Được Khuôn Mặt (Face Not Detected)
- **Triệu chứng:** Model trả về kết quả sai khác kỳ vọng, hoặc log ghi nhận cảnh báo liên tục `Face not detected`.
- **Khắc phục:**
  - Kiểm tra lại các hàm liên quan tới module nhận diện khuôn mặt (MTCNN, RetinaFace).
  - **Điều chỉnh `threshold` của MTCNN**: giảm các ngưỡng confidence xuống (ví dụ: `[0.6, 0.7, 0.7]` xuống `[0.5, 0.6, 0.6]`) để nhận diện được khuôn mặt mờ, bị khuất hoặc ở xa.
  - Kiểm tra `min_face_size` trong cấu hình MTCNN.

### 3. Lỗi API Timeout (Đặc biệt cho Video)
- **Triệu chứng:** Request tới `POST /predict/video` bị timeout hoặc Client bị ngắt kết nối `Connection aborted`.
- **Khắc phục:**
  - Hãy chắc chắn rằng endpoint `/predict/video` đang sử dụng cơ chế **Background Tasks** (Async Job). API nên trả về mã `202 Accepted` kèm `job_id` thay vì bắt Client chờ xử lý xong.
  - Nếu đã dùng Background Tasks nhưng job vẫn kẹt, hãy kiểm tra Celery/Worker xem có đang chạy hay bị treo không.
  - Sử dụng video với độ dài nhỏ hơn (ví dụ: cắt video còn 10 giây).

### 4. Lỗi DVC Pull Thất Bại (Thiếu Dữ Liệu)
- **Triệu chứng:** Lỗi `ERROR: unexpected error - không tìm thấy raw data`, hoặc lệnh `dvc pull` bị treo / fail.
- **Khắc phục:**
  - Chạy `dvc remote list` để kiểm tra remote.
  - Nếu sử dụng Google Drive làm remote storage: **Credentials có thể đã hết hạn**. Chạy lệnh xác thực lại để lấy token mới.
  - Thử chạy độc lập `dvc pull` ngoài terminal để kiểm tra xem có cửa sổ pop-up trình duyệt yêu cầu xác thực không.

### 5. Lỗi Thiếu Môi Trường Hoặc Dependency Conflict
- **Triệu chứng:** Gặp lỗi `ModuleNotFoundError`, `ImportError`, hoặc `Version Conflict` trong lúc chạy.
- **Khắc phục:**
  - Kiểm tra kết quả báo cáo của `scripts/check_env.py` khi bắt đầu pipeline.
  - Cài đặt đúng và đủ các thư viện với phiên bản tương ứng: `pip install -r requirements.txt`.
