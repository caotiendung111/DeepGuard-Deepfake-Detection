import pytest
import torch
from src.models import build_model

def test_efficientnet_output_shape():
    model = build_model(backbone="efficientnet_b4", pretrained=False)
    model.eval()
    
    # Batch size 2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 1)

def test_xception_output_shape():
    model = build_model(backbone="xception", pretrained=False)
    model.eval()
    
    # Xception normally uses 299x299, but we test 224x224 as well
    dummy_input = torch.randn(2, 3, 299, 299)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 1)
