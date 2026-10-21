"""Audio loading, augmentation and hand-crafted features.

Clips are resampled to 16 kHz (what pretrained speech models expect), the first
0.5 s is skipped because every RAVDESS clip opens with silence, and the next 3 s
are kept, zero-padded if the clip is shorter.
"""

from __future__ import annotations

import librosa
import numpy as np

SAMPLE_RATE = 16_000
OFFSET_S = 0.5
DURATION_S = 3.0
N_MELS = 128
N_FFT = 1024
HOP = 256
N_MFCC = 40


def fix_length(y: np.ndarray, n: int) -> np.ndarray:
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def load_clip(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, offset=OFFSET_S, duration=DURATION_S)
    return fix_length(y.astype(np.float32), int(sr * DURATION_S))


def load_trimmed(path: str, sr: int = SAMPLE_RATE, top_db: float = 30.0) -> np.ndarray:
    """Trim leading and trailing silence by energy, then keep 3 s.

    The fixed 0.5 s offset in ``load_clip`` suits RAVDESS, whose clips all open with
    silence, but cuts speech from datasets and recordings that don't. This loader
    is used wherever clips from outside RAVDESS are involved: the cross-dataset test
    and the served model.
    """
    y, _ = librosa.load(path, sr=sr)
    return trim_waveform(y.astype(np.float32), sr, top_db)


def trim_waveform(y: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 30.0) -> np.ndarray:
    if len(y) and np.abs(y).max() > 0:
        y, _ = librosa.effects.trim(y, top_db=top_db)
    return fix_length(y.astype(np.float32), int(sr * DURATION_S))


def augment(y: np.ndarray, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Light noise always, then pitch shift and time stretch each with p = 0.5.

    Only ever applied to training clips. The original pipeline augmented before
    splitting, which let a noisy copy of a test clip into training.
    """
    n = len(y)
    out = y + rng.normal(0.0, 0.005 * (np.abs(y).max() + 1e-8), n).astype(np.float32)
    if rng.random() < 0.5:
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=float(rng.uniform(-2, 2)))
    if rng.random() < 0.5:
        out = librosa.effects.time_stretch(out, rate=float(rng.uniform(0.85, 1.15)))
    return fix_length(out.astype(np.float32), n)


def log_mel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """(N_MELS, frames) log-mel spectrogram in dB, the CNN's input."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def mfcc_stats(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mean and std over time of MFCCs and their first and second deltas (240 values)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    stacked = np.concatenate(
        [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)], axis=0
    )
    return np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)]).astype(np.float32)
