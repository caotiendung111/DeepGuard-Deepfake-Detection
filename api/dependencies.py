"""
Dependencies and shared state for the API
"""
import time
from slowapi import Limiter
from slowapi.util import get_remote_address

# Global Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Global variables to store model state (loaded in lifespan)
app_state = {
    "model": None,
    "image_predictor": None,
    "video_predictor": None,
    "face_detector": None,
    "gradcam": None,
    "start_time": time.time(),
    "config": None
}

def get_model():
    return app_state["model"]

def get_face_detector():
    return app_state["face_detector"]

def get_gradcam():
    return app_state["gradcam"]

def get_app_config():
    return app_state["config"]

def get_uptime():
    return time.time() - app_state["start_time"]
