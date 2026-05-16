import pytest
import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from src.data.dataset import DeepfakeCSVDataset
from torch.utils.data import DataLoader

@pytest.fixture
def mock_csv(tmp_path):
    # Create a dummy CSV for testing
    csv_path = tmp_path / "test.csv"
    data = {
        "filepath": [str(tmp_path / f"{i}.jpg") for i in range(4)],
        "label": [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    # Create dummy images
    for p in data["filepath"]:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(p, img)
        
    return csv_path

def test_dataset_loading(mock_csv):
    dataset = DeepfakeCSVDataset(csv_file=mock_csv)
    assert len(dataset) == 4
    
    img_tensor, label, path = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert isinstance(path, str)
    assert label.item() in [0, 1]

def test_dataloader(mock_csv):
    dataset = DeepfakeCSVDataset(csv_file=mock_csv)
    loader = DataLoader(dataset, batch_size=2)
    
    batch = next(iter(loader))
    images, labels, paths = batch
    
    assert images.shape[0] == 2 # batch size
    assert images.shape[1] == 3 # channels
    assert labels.shape[0] == 2
