import torchvision.transforms as T


def get_train_transforms(image_size: int, cfg: dict) -> T.Compose:
    aug = cfg.get("augmentation", {})
    ops = [T.Resize((image_size, image_size))]

    if aug.get("horizontal_flip", True):
        ops.append(T.RandomHorizontalFlip())
    if aug.get("vertical_flip", False):
        ops.append(T.RandomVerticalFlip())
    if aug.get("rotation_degrees", 0):
        ops.append(T.RandomRotation(aug["rotation_degrees"]))
    if aug.get("color_jitter", False):
        ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))

    ops += [
        T.ToTensor(),
        T.Normalize(mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
                    std=aug.get("normalize_std",  [0.229, 0.224, 0.225])),
    ]
    return T.Compose(ops)


def get_eval_transforms(image_size: int, cfg: dict) -> T.Compose:
    aug = cfg.get("augmentation", {})
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=aug.get("normalize_mean", [0.485, 0.456, 0.406]),
                    std=aug.get("normalize_std",  [0.229, 0.224, 0.225])),
    ])
