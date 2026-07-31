from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import numpy as np


def compute_metrics(preds: list, labels: list, class_names: list) -> dict:
    """Return accuracy, macro-F1, and per-class report."""
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(labels, preds)
    return {"accuracy": acc, "f1_macro": f1, "report": report, "confusion_matrix": cm}
