"""
Download the Welding Defect dataset from Kaggle and create stratified train/val/test splits.

Requires Kaggle API credentials (~/.kaggle/kaggle.json).
Get yours at: https://www.kaggle.com/settings → API → Create New Token

Dataset: https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection
License: CC BY-SA 4.0
"""

import os
import shutil
import argparse
import subprocess
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

DATASET_SLUG = "sukmaadhiwijaya/welding-defect-object-detection"

CLASS_MAP = {
    "good":     "good",
    "crack":    "crack",
    "porosity": "porosity",
    "spatters": "spatters",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download(raw_dir: str) -> None:
    print(f"Downloading {DATASET_SLUG} → {raw_dir}")
    os.makedirs(raw_dir, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", raw_dir, "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print(
            "\nERROR: kaggle download failed. Make sure:\n"
            "  1. kaggle is installed: pip install kaggle\n"
            "  2. ~/.kaggle/kaggle.json exists with your API token\n"
            "  3. You have accepted the dataset license on Kaggle\n",
            file=sys.stderr,
        )
        sys.exit(1)

    source_md = Path(raw_dir) / "SOURCE.md"
    source_md.write_text(
        "# Dataset Source\n\n"
        f"- **Kaggle slug**: `{DATASET_SLUG}`\n"
        "- **URL**: https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection\n"
        "- **License**: CC BY-SA 4.0\n"
        "- **Citation**: Sukma Adhiwijaya (2023). Welding Defect Object Detection Dataset. Kaggle.\n"
    )
    print(f"Source metadata written → {source_md}")


def split_dataset(raw_dir: str, proc_dir: str,
                  val_split: float, test_split: float, seed: int) -> None:
    raw_path  = Path(raw_dir)
    proc_path = Path(proc_dir)

    for split in ("train", "val", "test"):
        for cls in CLASS_MAP.values():
            (proc_path / split / cls).mkdir(parents=True, exist_ok=True)

    total_moved = 0
    for raw_cls, canon_cls in CLASS_MAP.items():
        candidates = [raw_path / raw_cls, raw_path / raw_cls.capitalize()]
        src_dir = next((p for p in candidates if p.exists()), None)
        if src_dir is None:
            print(f"  [WARN] No folder found for class '{raw_cls}' — skipping.")
            continue

        images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if len(images) < 3:
            print(f"  [WARN] Only {len(images)} images for '{raw_cls}' — skipping.")
            continue

        # Stratified split: first carve out test, then val from remaining
        train_val, test_imgs = train_test_split(
            images, test_size=test_split, random_state=seed
        )
        # val_split is fraction of total, adjust for remaining pool
        val_fraction = val_split / (1 - test_split)
        train_imgs, val_imgs = train_test_split(
            train_val, test_size=val_fraction, random_state=seed
        )

        split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
        for split_name, paths in split_map.items():
            dst_dir = proc_path / split_name / canon_cls
            for p in paths:
                shutil.copy2(p, dst_dir / p.name)
            print(f"  {canon_cls}/{split_name}: {len(paths)} images")
            total_moved += len(paths)

    (proc_path / ".gitkeep").touch()
    print(f"\nDone — {total_moved} images split into {proc_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split weld defect dataset")
    parser.add_argument("--raw_dir",       default="data/raw")
    parser.add_argument("--proc_dir",      default="data/processed")
    parser.add_argument("--val_split",     type=float, default=0.15)
    parser.add_argument("--test_split",    type=float, default=0.15)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip Kaggle download if raw data is already present")
    args = parser.parse_args()

    if not args.skip_download:
        download(args.raw_dir)
    split_dataset(args.raw_dir, args.proc_dir, args.val_split, args.test_split, args.seed)
