import pytest
import io
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data

def test_predict_image_invalid_format():
    # Test uploading a text file instead of image
    file_content = b"This is not an image"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/predict/image", files=files)
    assert response.status_code == 422
    assert "Unsupported file format" in response.json()["detail"]

def test_predict_video_invalid_format():
    # Test uploading a non-video
    file_content = b"This is not a video"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    
    response = client.post("/predict/video", files=files)
    assert response.status_code == 422
    assert "Unsupported file format" in response.json()["detail"]

def test_get_invalid_job_id():
    response = client.get("/predict/video/invalid-job-id")
    assert response.status_code == 404
