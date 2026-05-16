import pytest
import numpy as np
from src.data.transforms import get_train_transforms
from src.inference.face_detector import FaceDetector

def test_face_detector_no_face():
    detector = FaceDetector(device='cpu')
    
    # Create an image of pure noise (no face)
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    
    faces = detector.detect_boxes(img)
    assert len(faces) == 0

def test_face_detector_cropping():
    detector = FaceDetector(device='cpu')
    
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Mock a box [x, y, w, h]
    box = [100, 100, 50, 50]
    
    crop = detector._crop(img, (100, 100, 150, 150))
    assert crop.shape == (224, 224, 3)

def test_augmentations():
    transform = get_train_transforms(image_size=224, augment_level='light')
    
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    augmented = transform(image=img)["image"]
    
    # After normalization and ToTensor
    assert augmented.shape == (3, 224, 224)
