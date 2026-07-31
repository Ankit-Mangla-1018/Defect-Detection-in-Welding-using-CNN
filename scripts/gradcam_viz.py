"""
Generate Grad-CAM visualisations for one sample per defect class.

Usage:
    python scripts/gradcam_viz.py --checkpoint checkpoints/best_model.pt --config configs/baseline.yaml

Saves a grid to assets/gradcam_examples.png showing:
  - Original image
  - Grad-CAM heatmap
  - Overlay (original + heatmap)
for each class (good, crack, porosity, spatters).
"""

import argparse
import sys
import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.transforms as T
from src.models import build_model_from_cfg
from src.data.dataset import CLASS_NAMES
from src.utils.gradcam import GradCAM, get_gradcam_target_layer


def run(ckpt_path: str, cfg_path: str, proc_dir: str, out_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    sz     = cfg["data"]["image_size"]
    device = torch.device("cpu")

    model = build_model_from_cfg(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    target_layer = get_gradcam_target_layer(model)
    cam_engine   = GradCAM(model, target_layer)

    eval_tf = T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Denormalize helper
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denorm(t: torch.Tensor) -> np.ndarray:
        img = t.squeeze().permute(1, 2, 0).numpy()
        img = (img * std + mean).clip(0, 1)
        return np.uint8(img * 255)

    fig, axes = plt.subplots(
        len(CLASS_NAMES), 3,
        figsize=(9, 12),
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
    )
    col_labels = ["Original", "Grad-CAM heatmap", "Overlay"]
    for ax, lbl in zip(axes[0], col_labels):
        ax.set_title(lbl, fontsize=11, pad=6)

    for row, cls in enumerate(CLASS_NAMES):
        cls_dir = Path(proc_dir) / "test" / cls
        img_path = sorted(cls_dir.iterdir())[0]

        pil_img = Image.open(img_path).convert("RGB").resize((sz, sz))
        orig_np = np.array(pil_img)

        tensor = eval_tf(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
        pred_idx = logits.argmax(1).item()
        conf     = torch.softmax(logits, dim=1)[0, pred_idx].item()

        # Grad-CAM for the predicted class
        heatmap = cam_engine(tensor.clone(), class_idx=pred_idx)
        overlay = GradCAM.overlay(heatmap, orig_np)

        row_label = f"{cls}\n(pred: {CLASS_NAMES[pred_idx]}, {conf:.0%})"
        axes[row][0].set_ylabel(row_label, fontsize=9, rotation=0,
                                 labelpad=70, va="center")

        axes[row][0].imshow(orig_np)
        axes[row][1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[row][2].imshow(overlay)

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(
        "Grad-CAM — Model Attention per Defect Class",
        fontsize=13, y=1.01,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    cam_engine.remove_hooks()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config",     default="configs/baseline.yaml")
    parser.add_argument("--proc_dir",   default="data/processed")
    parser.add_argument("--out",        default="assets/gradcam_examples.png")
    args = parser.parse_args()
    run(args.checkpoint, args.config, args.proc_dir, args.out)
