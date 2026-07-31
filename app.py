"""
Weld Defect Detector — Interactive Streamlit Demo

Run with:
    streamlit run app.py

Requires a trained checkpoint at checkpoints/best_model.pt.
Train one with: python scripts/train.py --config configs/baseline.yaml
"""

import sys
import os
import io

import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import build_model_from_cfg
from src.data.dataset import CLASS_NAMES
from src.utils.gradcam import GradCAM, get_gradcam_target_layer

import yaml

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weld Defect Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────────
CFG_PATH  = "configs/baseline.yaml"
CKPT_PATH = "checkpoints/best_model.pt"

CLASS_INFO = {
    "good":     {"emoji": "✅", "color": "#2ecc71", "desc": "No defect detected. Weld meets quality criteria."},
    "crack":    {"emoji": "⚠️", "color": "#e74c3c", "desc": "Linear crack detected. Safety-critical — requires immediate inspection."},
    "porosity": {"emoji": "🔶", "color": "#f39c12", "desc": "Gas pore(s) detected. Weakens weld integrity under stress."},
    "spatters": {"emoji": "🔵", "color": "#3498db", "desc": "Spatter deposits detected. Cosmetic issue; may indicate parameter drift."},
}

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_cfg(cfg)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()
    return model, cfg


@st.cache_resource
def get_transforms(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def run_inference(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()
    return probs


def generate_gradcam(model, tensor, pred_idx):
    target_layer = get_gradcam_target_layer(model)
    cam = GradCAM(model, target_layer)
    heatmap = cam(tensor.clone(), class_idx=pred_idx)
    cam.remove_hooks()
    return heatmap


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Weld Defect Detector")
st.markdown(
    "Upload a weld image to classify it as **good**, **crack**, **porosity**, or **spatters**. "
    "Grad-CAM highlights the regions the model used to make its decision."
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Model:** WeldCNN — custom 4-block CNN  \n"
        "**Training:** ~800 images, 4 classes  \n"
        "**Test accuracy:** 97.0%  \n"
        "**Macro F1:** 0.970  \n\n"
        "---\n"
        "**Classes:**"
    )
    for cls, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} **{cls.capitalize()}** — {info['desc']}")

    st.divider()
    show_gradcam = st.toggle("Show Grad-CAM overlay", value=True)
    st.markdown(
        "[GitHub](https://github.com/Ankit-Mangla-1018/weld-defect-cnn) · "
        "[Dataset](https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection)"
    )

# Check checkpoint
if not os.path.exists(CKPT_PATH):
    st.error(
        f"No checkpoint found at `{CKPT_PATH}`.  \n"
        "Train the model first:\n```bash\npython scripts/train.py --config configs/baseline.yaml\n```"
    )
    st.stop()

# Load model
try:
    model, cfg = load_model()
    image_size = cfg["data"]["image_size"]
    transform  = get_transforms(image_size)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Upload
uploaded = st.file_uploader(
    "Upload a weld image (JPG, PNG)",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Works best with greyscale or near-greyscale weld radiograph images.",
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    tensor  = transform(pil_img).unsqueeze(0)
    probs   = run_inference(model, tensor)
    pred_idx = probs.argmax().item()
    pred_cls = CLASS_NAMES[pred_idx]
    info     = CLASS_INFO[pred_cls]

    # Result banner
    st.markdown(
        f"""
        <div style="background:{info['color']}22; border-left:5px solid {info['color']};
                    padding:1rem 1.25rem; border-radius:6px; margin-bottom:1rem;">
            <span style="font-size:1.6rem">{info['emoji']}</span>
            <span style="font-size:1.3rem; font-weight:700; margin-left:0.5rem;">
                {pred_cls.upper()}
            </span>
            &nbsp;&nbsp;
            <span style="font-size:1rem; color:#444;">
                {probs[pred_idx].item():.1%} confidence
            </span>
            <br><span style="color:#555; font-size:0.9rem;">{info['desc']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    # Original image
    with col1:
        st.subheader("Input image")
        st.image(pil_img, use_container_width=True)

    # Grad-CAM overlay
    with col2:
        if show_gradcam:
            st.subheader("Grad-CAM — model attention")
            with st.spinner("Computing Grad-CAM..."):
                try:
                    heatmap   = generate_gradcam(model, tensor, pred_idx)
                    orig_np   = np.array(pil_img.resize((image_size, image_size)))
                    from src.utils.gradcam import GradCAM as _GC
                    overlay   = _GC.overlay(heatmap, orig_np, alpha=0.5)
                    st.image(overlay, use_container_width=True)
                    st.caption("Warmer colours (red/yellow) = regions with highest influence on prediction.")
                except Exception as e:
                    st.warning(f"Grad-CAM failed: {e}")
        else:
            st.subheader("Class probabilities")

    # Probability bars
    st.subheader("All class probabilities")
    for i, cls in enumerate(CLASS_NAMES):
        p = probs[i].item()
        bar_color = CLASS_INFO[cls]["color"] if i == pred_idx else "#ccc"
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; margin-bottom:6px;">
                <span style="width:90px; font-weight:{'700' if i==pred_idx else '400'}">{cls}</span>
                <div style="flex:1; background:#eee; border-radius:4px; height:18px; margin:0 10px;">
                    <div style="width:{p*100:.1f}%; background:{bar_color};
                                height:100%; border-radius:4px;"></div>
                </div>
                <span style="width:50px; text-align:right; font-weight:{'700' if i==pred_idx else '400'}">
                    {p:.1%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # Placeholder / instructions
    st.info("👆 Upload a weld image above to get started.")
    st.markdown("#### What this model detects")
    cols = st.columns(len(CLASS_NAMES))
    for col, (cls, info) in zip(cols, CLASS_INFO.items()):
        col.markdown(
            f"**{info['emoji']} {cls.capitalize()}**  \n{info['desc']}"
        )
