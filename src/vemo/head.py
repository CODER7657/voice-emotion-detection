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


def _finite(head: dict, key: str, ndim: int) -> np.ndarray:
    try:
        raw = np.asarray(head[key], dtype=object)
    except (TypeError, ValueError):  # ragged nesting
        raise ValueError(f"head {key!r} must be numbers") from None
    # JSON numbers only: no strings such as "1.0", no booleans, no nested surprises.
    if raw.ndim != ndim or not all(type(v) in (int, float) for v in raw.ravel()):
        raise ValueError(f"head {key!r} must be a {ndim}-D array of numbers")
    arr = raw.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"head {key!r} contains NaN or infinity")
    return arr


def validate_head(head: dict) -> int:
    """Raise ``ValueError`` unless the head is usable; return its input width.

    Usable means: all keys present, classes a list of distinct names, every
    number finite, every scale strictly positive, and consistent shapes, so
    ``head_proba`` can only return finite probabilities for finite input.
    """
    if not isinstance(head, dict):
        raise ValueError("head must be a JSON object")
    missing = [k for k in HEAD_KEYS if k not in head]
    if missing:
        raise ValueError(f"head is missing {missing}")
    classes = head["classes"]
    if (not isinstance(classes, list) or len(classes) < 2
            or not all(isinstance(c, str) and c for c in classes) or len(set(classes)) != len(classes)):
        raise ValueError("head 'classes' must be a list of at least two distinct names")
    mean, scale = _finite(head, "mean", 1), _finite(head, "scale", 1)
    coef, intercept = _finite(head, "coef", 2), _finite(head, "intercept", 1)
    n_classes, dim = len(classes), len(mean)
    if dim == 0 or scale.shape != (dim,) or coef.shape != (n_classes, dim) or intercept.shape != (n_classes,):
        raise ValueError("head shapes are inconsistent")
    if not np.all(scale > 0):
        raise ValueError("head 'scale' must be > 0 everywhere")
    return dim
