"""The logistic-regression head as plain NumPy, importable without scikit-learn.

``models/head.json`` stores a standardised multinomial logistic regression as
numbers (see ``serve.head_from_pipeline``). This module only needs NumPy, so the
deployed API (``api/index.py``) can use it without pulling in sklearn or librosa.
"""

from __future__ import annotations

import numpy as np

#: Keys a head.json must have.
HEAD_KEYS = ("classes", "mean", "scale", "coef", "intercept")


def head_proba(head: dict, X: np.ndarray) -> np.ndarray:
    """Class probabilities, matching ``pipeline.predict_proba`` for multinomial regression."""
    z = (X - np.asarray(head["mean"])) / np.asarray(head["scale"])
    logits = z @ np.asarray(head["coef"]).T + np.asarray(head["intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def validate_head(head: dict) -> int:
    """Raise ``ValueError`` unless the head is internally consistent; return its input width."""
    missing = [k for k in HEAD_KEYS if k not in head]
    if missing:
        raise ValueError(f"head is missing {missing}")
    n_classes, dim = len(head["classes"]), len(head["mean"])
    coef = np.asarray(head["coef"], dtype=float)
    if (len(head["scale"]) != dim or coef.shape != (n_classes, dim)
            or len(head["intercept"]) != n_classes or n_classes < 2):
        raise ValueError("head shapes are inconsistent")
    return dim
