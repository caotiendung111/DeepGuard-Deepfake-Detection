#!/bin/bash

# Thiết lập cờ để dừng script ngay lập tức nếu bất kỳ lệnh nào fail
set -e

echo "=========================================="
echo "    DEEPGUARD - END-TO-END PIPELINE       "
echo "=========================================="

echo -e "\n[1/4] Kiểm tra môi trường..."
python scripts/check_env.py

echo -e "\n[2/4] Chạy DVC pipeline (Data -> Train -> Evaluate)..."
# Chạy toàn bộ pipeline qua DVC
dvc repro

echo -e "\n[3/4] Chạy toàn bộ Unit Tests..."
# Chạy pytest và hiển thị verbose output
pytest tests/ -v

echo -e "\n[4/4] Khởi động API và chạy Smoke Test..."

# Khởi động uvicorn server ở background
echo "Khởi động FastAPI server ở cổng 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
# Lấy Process ID của uvicorn để tắt sau khi test xong
API_PID=$!

# Đợi vài giây để API khởi động hoàn toàn
sleep 5

# Tạo thư mục và dữ liệu giả (nếu chưa có) để test không bị lỗi file not found
echo "Chuẩn bị test data..."
mkdir -p test_data
python -c "
import cv2, numpy as np, os
if not os.path.exists('test_data/real.jpg'):
    cv2.imwrite('test_data/real.jpg', np.zeros((224, 224, 3), dtype=np.uint8))
if not os.path.exists('test_data/fake.jpg'):
    cv2.imwrite('test_data/fake.jpg', np.zeros((224, 224, 3), dtype=np.uint8))
if not os.path.exists('test_data/sample.mp4'):
    out = cv2.VideoWriter('test_data/sample.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (224,224))
    [out.write(np.zeros((224,224,3), dtype=np.uint8)) for _ in range(300)]
    out.release()
" || echo "Không thể tạo file OpenCV dummy (bỏ qua nếu file đã có thực tế)."

echo "Chạy smoke tests (API Endpoints Check)..."

# Dùng set +e tạm thời để dù smoke test fail cũng vẫn có thể kill API Server
set +e
python scripts/smoke_test.py \
    --real-img test_data/real.jpg \
    --fake-img test_data/fake.jpg \
    --video test_data/sample.mp4
TEST_EXIT_CODE=$?
set -e

# Tắt server FastAPI sau khi test xong
echo "Tắt FastAPI server (PID: $API_PID)..."
kill $API_PID || true

echo "=========================================="
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "    [FAIL] SMOKE TEST THẤT BẠI!           "
    echo "=========================================="
    exit 1
else
    echo "    [PASS] TOÀN BỘ PIPELINE THÀNH CÔNG!   "
    echo "=========================================="
fi
