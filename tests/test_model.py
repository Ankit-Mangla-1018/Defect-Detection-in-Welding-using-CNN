import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import WeldCNN


def test_output_shape():
    model = WeldCNN(num_classes=4)
    model.eval()
    x = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 4), f"Expected (8, 4), got {out.shape}"


def test_different_image_sizes():
    """GAP makes the model resolution-agnostic."""
    model = WeldCNN(num_classes=4)
    model.eval()
    for size in [128, 224, 320]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4), f"Failed at size {size}"


def test_trainable_params():
    model = WeldCNN(num_classes=4)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Sanity: should be a few hundred thousand, not millions
    assert 100_000 < params < 5_000_000, f"Unexpected param count: {params}"


def test_num_classes_config():
    for n in [2, 4, 6]:
        model = WeldCNN(num_classes=n)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == n
