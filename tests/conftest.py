"""
DeepGuard — Test Configuration & Fixtures
"""
import pytest
import numpy as np
import torch
from PIL import Image


@pytest.fixture
def dummy_image_rgb():
    """224x224 random RGB numpy array."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_pil_image():
    """224x224 random PIL Image."""
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


@pytest.fixture
def dummy_tensor():
    """Single image tensor (1, 3, 224, 224)."""
    return torch.rand(1, 3, 224, 224)


@pytest.fixture
def dummy_batch():
    """Batch of 4 images."""
    return torch.rand(4, 3, 224, 224)


@pytest.fixture
def sample_labels():
    return torch.tensor([0, 1, 0, 1])


@pytest.fixture
def device():
    return torch.device("cpu")
