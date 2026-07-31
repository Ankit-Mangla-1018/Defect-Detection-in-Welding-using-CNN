"""
Run inference on a single weld image.

Usage:
    python scripts/predict.py --image path/to/weld.jpg --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
import os
import yaml
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import CLASS_NAMES, get_eval_transforms
from src.models import build_model


def predict(image_path: str, ckpt_path: str, cfg_path: str = "configs/baseline.yaml") -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    transform = get_eval_transforms(cfg["data"]["image_size"], cfg)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)   # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu()

    pred_idx = probs.argmax().item()
    print(f"\nImage     : {image_path}")
    print(f"Prediction: {CLASS_NAMES[pred_idx].upper()}  ({probs[pred_idx]:.2%} confidence)\n")
    print("All class probabilities:")
    for cls, p in zip(CLASS_NAMES, probs):
        bar = "█" * int(p * 30)
        print(f"  {cls:<12} {p:.2%}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config",     default="configs/baseline.yaml")
    args = parser.parse_args()
    predict(args.image, args.checkpoint, args.config)
