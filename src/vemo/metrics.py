"""Scores reported for every model and protocol."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def summarize(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Accuracy, macro-F1, per-class recall and the confusion matrix.

    Macro-F1 is the headline: RAVDESS has half as many neutral clips as any other
    emotion, so plain accuracy rewards ignoring it.
    """
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": {label: float(r) for label, r in zip(labels, recall)},
        "labels": list(labels),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def across_folds(fold_scores: list[dict], key: str = "macro_f1") -> dict:
    """Mean and standard deviation of one score over folds."""
    values = np.array([s[key] for s in fold_scores], dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
