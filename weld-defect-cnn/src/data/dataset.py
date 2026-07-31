import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


CLASS_NAMES = ["good", "crack", "porosity", "spatters"]


class WeldDataset(Dataset):
    """
    Expects the following folder layout (produced by scripts/download_data.py):
        data/processed/{split}/{class_name}/{image}.jpg
    where split ∈ {train, val, test}.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root) / split
        self.transform = transform
        self.class_names = CLASS_NAMES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: list[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. "
                "Run `python scripts/download_data.py` first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Return a sampler that up-weights underrepresented defect classes."""
        labels = [lbl for _, lbl in self.samples]
        class_counts = np.bincount(labels, minlength=len(self.class_names))
        class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
        sample_weights = [class_weights[lbl] for lbl in labels]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
