import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device

        tcfg = cfg["training"]
        self.epochs    = tcfg["epochs"]
        self.grad_clip = tcfg.get("grad_clip", 1.0)
        self.ckpt_dir  = tcfg["checkpoint_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
        )

        if tcfg["scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )

        self.early_stopping = EarlyStopping(patience=tcfg["early_stopping_patience"])
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }
        self.best_val_loss = float("inf")

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping — prevents exploding gradients on noisy batches
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct    += (logits.argmax(1) == labels).sum().item()
                total      += images.size(0)

        return total_loss / total, correct / total

    def fit(self) -> dict:
        print(f"Training on {self.device} for up to {self.epochs} epochs\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   train=False)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc: {train_acc:.3f}/{val_acc:.3f} | "
                f"{elapsed:.1f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                ckpt = os.path.join(self.ckpt_dir, "best_model.pt")
                torch.save(self.model.state_dict(), ckpt)
                print(f"  ✓ Saved best model → {ckpt}")

            if self.early_stopping.step(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        return self.history
