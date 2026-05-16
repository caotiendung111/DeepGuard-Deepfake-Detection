import argparse
import requests
import time
import sys
import os

API_URL = "http://127.0.0.1:8000"

def test_health():
    print("Testing GET /health...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            print("  [PASS] Health check ok.")
            return True
        else:
            print(f"  [FAIL] Health check failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"  [FAIL] Health check exception: {e}")
    return False

def test_image(img_path, expected_label):
    print(f"Testing POST /predict/image with {img_path}...")
    if not os.path.exists(img_path):
        print(f"  [SKIP] Image not found: {img_path}")
        return False
        
    try:
        with open(img_path, "rb") as f:
            resp = requests.post(f"{API_URL}/predict/image", files={"file": f}, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            # Trích xuất label từ response. Có thể nằm ở trường 'label' hoặc 'prediction'
            label = data.get("label", data.get("prediction", "UNKNOWN"))
            
            # So sánh (chấp nhận chữ hoa chữ thường)
            if str(label).upper() == expected_label.upper():
                print(f"  [PASS] Image predicted correctly as {label}.")
                return True
            else:
                print(f"  [FAIL] Expected {expected_label}, got {label}. Full response: {data}")
        else:
            print(f"  [FAIL] HTTP Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
    return False

def test_video(vid_path):
    print(f"Testing POST /predict/video with {vid_path}...")
    if not os.path.exists(vid_path):
        print(f"  [SKIP] Video not found: {vid_path}")
        return False
        
    try:
        with open(vid_path, "rb") as f:
            resp = requests.post(f"{API_URL}/predict/video", files={"file": f}, timeout=10)
        
        # Nếu API sử dụng Async Jobs, status sẽ là 202 (Accepted)
        if resp.status_code == 202: 
            data = resp.json()
            job_id = data.get("job_id")
            print(f"  [PASS] Video submitted successfully. Job ID: {job_id}")
            
            # Polling status của job video
            print(f"  Polling job status...")
            for i in range(20):
                time.sleep(2)
                status_resp = requests.get(f"{API_URL}/predict/video/{job_id}", timeout=5)
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    status = status_data.get("status")
                    print(f"    - Attempt {i+1}: {status}")
                    
                    if status == "completed":
                        print(f"  [PASS] Video processing completed! Result: {status_data.get('result')}")
                        return True
                    elif status == "failed":
                        print(f"  [FAIL] Video processing failed! Error: {status_data.get('error')}")
                        return False
                else:
                    print(f"  [FAIL] Failed to get job status. HTTP {status_resp.status_code}")
                    return False
            print("  [FAIL] Timeout waiting for video processing.")
            
        # Nếu API xử lý trực tiếp luôn, trả về 200 OK
        elif resp.status_code == 200:
            print(f"  [PASS] Video processed synchronously. Response: {resp.json()}")
            return True
        else:
            print(f"  [FAIL] HTTP Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-img", default="test_data/real.jpg", help="Path to a real face image")
    parser.add_argument("--fake-img", default="test_data/fake.jpg", help="Path to a fake face image")
    parser.add_argument("--video", default="test_data/sample.mp4", help="Path to a sample 10s video")
    args = parser.parse_args()

    print("========================================")
    print("          SMOKE TEST PIPELINE           ")
    print("========================================\n")
    
    health_ok = test_health()
    if not health_ok:
        print("\n[!] API is not healthy. Aborting tests.")
        sys.exit(1)
        
    print()
    test_image(args.real_img, "REAL")
    print()
    test_image(args.fake_img, "FAKE")
    print()
    test_video(args.video)
    
    print("\n========================================")
    print("             SMOKE TEST END             ")
    print("========================================\n")
