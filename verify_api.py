
import requests
import json
import os

API_URL = "http://127.0.0.1:8000/predict/image"

def test_prediction(image_path, label):
    print(f"\n--- Testing {label} Image: {os.path.basename(image_path)} ---")
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                print(json.dumps(result, indent=2))
                pred_label = result.get("prediction", "Unknown")
                conf = result.get("probability", 0)
                print(f"RESULT: Model says this is {pred_label} (Conf: {conf:.4f})")
            else:
                print(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"Connection Error: {e}")

if __name__ == "__main__":
    # Test Real Image
    test_prediction("data/external_test/real/real_tomas.jpg", "REAL")
    
    # Test Fake Image
    test_prediction("data/external_test/fake/fake_tpdne.jpg", "FAKE")
