"""Silence trimming and fixed-length cropping in NumPy alone.

The served model must not import librosa: it pulls in numba and scipy, which do
not fit a Vercel Python function. This is a port of the two steps the model was
trained behind (``features.trim_waveform``):

1. ``librosa.effects.trim(y, top_db=30)``: frame RMS (2048 samples, hop 512,
   zero-padded so frames are centred), in dB relative to the loudest frame, and
   keep the span from the first to one past the last frame above ``-top_db``.
2. Crop or zero-pad to 3 s.

It mirrors librosa's arithmetic, float32 power and the same frame layout and
reduction axis, so frames sitting right at the threshold fall the same way.
``tests/test_trim.py`` checks the result against librosa itself.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
DURATION_S = 3.0
FRAME_LENGTH = 2048
HOP_LENGTH = 512
#: librosa.amplitude_to_db's default floor, applied to power as amin**2.
AMIN = 1e-5


def fix_length(y: np.ndarray, n: int) -> np.ndarray:
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def _frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    pad = frame_length // 2
    padded = np.pad(y, (pad, pad), mode="constant")
    # (frame_length, n_frames), the layout librosa.util.frame returns, reduced
    # over axis 0 as librosa.feature.rms does, so float32 sums match.
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length].T
    power = np.mean(np.square(np.abs(frames), dtype=np.float32), axis=0)
    return np.sqrt(power)


def trim_bounds(
    y: np.ndarray,
    top_db: float = 30.0,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> tuple[int, int]:
    """(start, end) sample indices of the non-silent span, as librosa returns."""
    if y.ndim != 1:
        raise ValueError(f"expected mono audio (1-D), got shape {y.shape}")
    if len(y) == 0:
        return 0, 0
    if not np.all(np.isfinite(y)):
        raise ValueError("audio contains NaN or infinity")

    rms = _frame_rms(y.astype(np.float32, copy=False), frame_length, hop_length)
    power = np.square(rms)
    ref = np.max(power)
    amin = np.float32(AMIN**2)
    db = 10.0 * np.log10(np.maximum(amin, power)) - 10.0 * np.log10(np.maximum(amin, ref))

    loud = np.flatnonzero(db > -top_db)
    if loud.size == 0:
        return 0, 0
    start = int(loud[0]) * hop_length
    end = min(len(y), (int(loud[-1]) + 1) * hop_length)
    return start, end


def trim(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    start, end = trim_bounds(y, top_db)
    return y[start:end]


def trim_waveform(
    y: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 30.0
) -> np.ndarray:
    """Drop-in for ``features.trim_waveform``: trim, then exactly 3 s of float32.

    All-zero or empty input is not trimmed (there is no loudest frame to be
    relative to), matching the guard in the librosa version.
    """
    y = np.asarray(y, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        raise ValueError("audio contains NaN or infinity")
    if len(y) and np.abs(y).max() > 0:
        y = trim(y, top_db)
    return fix_length(y, int(sr * DURATION_S)).astype(np.float32, copy=False)
