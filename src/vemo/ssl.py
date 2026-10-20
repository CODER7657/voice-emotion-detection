"""Frozen self-supervised speech features (WavLM).

The pretrained model is never trained here. Each clip is passed through it once,
hidden states are averaged over time and over layers, and a linear classifier is
trained on top. Run this on a GPU (Colab); on a laptop CPU it is slow.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "microsoft/wavlm-base-plus"


def embed(
    clips: list[np.ndarray],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    sr: int = 16_000,
) -> np.ndarray:
    """Return one (hidden_size,) vector per clip, mean-pooled over time and layers."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            inputs = extractor(batch, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = model(**inputs, output_hidden_states=True).hidden_states
            # (layers, batch, time, dim) -> mean over time, then over layers.
            pooled = torch.stack(hidden).mean(dim=2).mean(dim=0)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)
