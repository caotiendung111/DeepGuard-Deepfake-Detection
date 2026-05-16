
print(">>> [DEBUG] Bat dau nap thu vien...")
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print(">>> [DEBUG] Nap Torch...")
import torch
print(f">>> [DEBUG] Torch OK (Device: {'cuda' if torch.cuda.is_available() else 'cpu'})")

print(">>> [DEBUG] Nap FastAPI...")
from fastapi import FastAPI
print(">>> [DEBUG] FastAPI OK")

print(">>> [DEBUG] Nap Config...")
from src.utils.config import load_config
print(">>> [DEBUG] Config OK")

print(">>> [DEBUG] Nap Predictors...")
from src.inference.predictor import ImagePredictor
print(">>> [DEBUG] Predictors OK")

print(">>> [DEBUG] Nap app tu api.main...")
try:
    from api.main import app
    print(">>> [DEBUG] APP IMPORT OK!")
except Exception as e:
    print(f">>> [DEBUG] APP IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    print(">>> [DEBUG] Dang khoi dong server...")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print(f">>> [DEBUG] SERVER START FAILED: {e}")
