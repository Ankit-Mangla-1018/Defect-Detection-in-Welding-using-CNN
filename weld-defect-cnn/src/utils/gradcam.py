"""
Gradient-weighted Class Activation Mapping (Grad-CAM).

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

Highlights which spatial regions the model attended to when making a prediction,
useful for debugging and building trust in defect detection systems.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    Hook-based Grad-CAM for any CNN with a named target layer.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(input_tensor, class_idx=None)   # None → predicted class
        overlay = cam.overlay(heatmap, original_image)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,   # (1, C, H, W)
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Return Grad-CAM heatmap in [0, 1], shape (H, W)."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool gradients over spatial dims → (C,)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay(
        heatmap: np.ndarray,
        image: np.ndarray,          # (H, W, 3) uint8 RGB
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Superimpose coloured heatmap on the original image.

        Returns uint8 RGB array same size as image.
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        colormap        = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb    = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        blended = np.uint8(alpha * colormap_rgb + (1 - alpha) * image)
        return blended

    def remove_hooks(self) -> None:
        """Call when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Return the last convolutional layer for supported architectures.
    Used as the default Grad-CAM target.
    """
    from src.models.cnn import WeldCNN
    try:
        from src.models.resnet import FineTunedResNet18
        if isinstance(model, FineTunedResNet18):
            return model.model.layer4[-1].conv2
    except ImportError:
        pass

    if isinstance(model, WeldCNN):
        # Last ConvBlock's conv layer inside features[-1].block[0]
        return model.features[-1].block[0]

    raise ValueError(
        f"No default Grad-CAM layer for {type(model).__name__}. "
        "Pass target_layer explicitly."
    )
