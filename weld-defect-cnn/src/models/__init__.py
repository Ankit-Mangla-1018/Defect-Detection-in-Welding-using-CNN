from .cnn import WeldCNN, build_model
from .resnet import FineTunedResNet18, build_resnet


def build_model_from_cfg(cfg: dict):
    """Factory: returns WeldCNN or FineTunedResNet18 based on cfg['model']['name']."""
    name = cfg["model"].get("name", "WeldCNN")
    if name == "WeldCNN":
        return build_model(cfg)
    elif name == "ResNet18":
        return build_resnet(cfg)
    else:
        raise ValueError(f"Unknown model name: '{name}'. Choose 'WeldCNN' or 'ResNet18'.")
