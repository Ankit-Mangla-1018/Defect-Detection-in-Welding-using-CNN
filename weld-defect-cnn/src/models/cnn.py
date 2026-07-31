import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool"""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WeldCNN(nn.Module):
    """
    Four convolutional blocks + Global Average Pooling + FC head.

    Input:  (B, 3, H, W)  — expects 224×224 by default
    Output: (B, num_classes)  — raw logits
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),    # → 112×112
            ConvBlock(32,  64),    # →  56×56
            ConvBlock(64,  128),   # →  28×28
            ConvBlock(128, 256),   # →  14×14
        )
        # Global Average Pooling collapses spatial dims to 1×1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_model(cfg: dict) -> WeldCNN:
    return WeldCNN(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
