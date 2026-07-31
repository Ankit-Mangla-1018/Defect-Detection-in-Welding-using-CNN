from torch.utils.data import DataLoader

from .dataset import WeldDataset
from .transforms import get_train_transforms, get_eval_transforms


def build_dataloaders(cfg: dict):
    img_size   = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]
    workers    = cfg["data"]["num_workers"]
    proc_dir   = cfg["data"]["processed_dir"]

    train_ds = WeldDataset(proc_dir, split="train", transform=get_train_transforms(img_size, cfg))
    val_ds   = WeldDataset(proc_dir, split="val",   transform=get_eval_transforms(img_size, cfg))
    test_ds  = WeldDataset(proc_dir, split="test",  transform=get_eval_transforms(img_size, cfg))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_ds.make_weighted_sampler(),
        num_workers=workers,
        pin_memory=True,
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.class_names
