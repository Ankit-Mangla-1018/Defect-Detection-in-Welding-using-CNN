"""
Train WeldCNN or ResNet18 (fine-tuned) on the weld defect dataset.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/resnet18_finetune.yaml
"""

import argparse
import sys
import os
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, plot_training_curves
from src.data import build_dataloaders
from src.models import build_model_from_cfg
from src.training import Trainer


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config : {cfg_path}")
    print(f"Device : {device}\n")

    train_loader, val_loader, _, class_names = build_dataloaders(cfg)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    model = build_model_from_cfg(cfg)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model     : {cfg['model']['name']}")
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable\n")

    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    history = trainer.fit()

    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)
    curves_path = os.path.join(cfg["training"]["log_dir"], "training_curves.png")
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        save_path=curves_path,
    )
    print(f"\nDone. Checkpoint: {cfg['training']['checkpoint_dir']}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)
