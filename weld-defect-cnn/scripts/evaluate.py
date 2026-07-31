"""
Evaluate a trained model on the held-out test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint checkpoints/resnet18/best_model.pt --config configs/resnet18_finetune.yaml
"""

import argparse
import sys
import os
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, compute_metrics, plot_confusion_matrix
from src.data import build_dataloaders
from src.models import build_model_from_cfg


def main(ckpt_path: str, cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = build_dataloaders(cfg)

    model = build_model_from_cfg(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_preds, all_labels, class_names)

    print(f"\n{'='*52}")
    print(f"Test Accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"Macro F1      : {metrics['f1_macro']:.4f}")
    print(f"\nPer-class Report:\n{metrics['report']}")

    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)
    cm_path = os.path.join(cfg["training"]["log_dir"], "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path=cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config",     default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.checkpoint, args.config)
