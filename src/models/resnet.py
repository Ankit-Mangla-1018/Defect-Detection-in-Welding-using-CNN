"""
Fine-tuned ResNet18 for weld defect classification.

Uses ImageNet pretrained weights and replaces the classification head.
Two fine-tuning strategies:
  - feature_extract=True  → freeze backbone, train head only (fast, needs less data)
  - feature_extract=False → unfreeze all layers, full fine-tune (better with more data)
"""

import torch
import torch.nn as nn
from torchvision import models


class FineTunedResNet18(nn.Module):
    """
    ResNet18 with pretrained ImageNet weights + custom classification head.

    Args:
        num_classes:      number of output classes
        feature_extract:  if True, freeze backbone and only train the head
        dropout:          dropout rate before the final linear layer
        pretrained:       if False, use random initialisation (useful offline)
    """

    def __init__(
        self,
        num_classes: int = 4,
        feature_extract: bool = False,
        dropout: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if pretrained:
            try:
                weights  = models.ResNet18_Weights.DEFAULT
                backbone = models.resnet18(weights=weights)
            except Exception:
                print(
                    "[WARN] Could not download ResNet18 pretrained weights "
                    "(no internet or blocked URL). Falling back to random init."
                )
                backbone = models.resnet18(weights=None)
        else:
            backbone = models.resnet18(weights=None)

        if feature_extract:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_resnet(cfg: dict) -> FineTunedResNet18:
    mcfg = cfg["model"]
    return FineTunedResNet18(
        num_classes=mcfg["num_classes"],
        feature_extract=mcfg.get("feature_extract", False),
        dropout=mcfg.get("dropout", 0.4),
        pretrained=mcfg.get("pretrained", True),
    )
