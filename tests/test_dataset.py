import os
import sys
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import WeldDataset, CLASS_NAMES


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a minimal fake dataset structure."""
    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(d / f"{cls}_{i}.jpg")
    return tmp_path


def test_dataset_len(fake_dataset):
    ds = WeldDataset(str(fake_dataset), split="train")
    # 4 classes × 4 images each = 16
    assert len(ds) == 16


def test_dataset_item_shape(fake_dataset):
    import torchvision.transforms as T
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    ds = WeldDataset(str(fake_dataset), split="train", transform=transform)
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert label in range(len(CLASS_NAMES))


def test_weighted_sampler_length(fake_dataset):
    ds = WeldDataset(str(fake_dataset), split="train")
    sampler = ds.make_weighted_sampler()
    assert sampler.num_samples == len(ds)


def test_missing_folder_raises():
    with pytest.raises(RuntimeError):
        WeldDataset("/nonexistent/path", split="train")
