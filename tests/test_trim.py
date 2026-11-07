"""The NumPy trim must agree with librosa, which the model was trained behind.

Done-when (roadmap voice #1): start and end within one sample of
``librosa.effects.trim(top_db=30)`` on 50 clips, plus silent, short and long
cases. CI uses 50 deterministic synthetic clips (RAVDESS is CC BY-NC-SA and is
not committed); set ``VEMO_CLIPS_DIR`` to a folder of .wav files to run the same
comparison on real speech locally.
"""

from __future__ import annotations

import os
import pathlib

import librosa
import numpy as np
import pytest

from vemo import features, trim

SR = 16_000
THREE_S = 3 * SR


def _speechlike(rng: np.random.Generator, seconds: float, level: float) -> np.ndarray:
    """Amplitude-modulated harmonics plus noise: syllable-rate energy changes,
    which is what makes the frame threshold interesting."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = rng.uniform(90, 260)
    voice = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(2, 6) * t + rng.uniform(0, np.pi)))
    noise = rng.normal(0, 0.3, len(t))
    y = (voice + noise) * envelope
    return (level * y / (np.abs(y).max() + 1e-12)).astype(np.float32)


def _synthetic_clips(n: int = 50) -> list[np.ndarray]:
    rng = np.random.default_rng(1188976)
    clips = []
    for i in range(n):
        lead = np.zeros(int(rng.uniform(0, 1.5) * SR), np.float32)
        tail = np.zeros(int(rng.uniform(0, 1.5) * SR), np.float32)
        speech = _speechlike(rng, rng.uniform(0.05, 5.0), 10 ** rng.uniform(-3, 0))
        y = np.concatenate([lead, speech, tail])
        floor = [0.0, 1e-5, 1e-4, 1e-3][i % 4]  # some clips have room noise
        if floor:
            y = y + rng.normal(0, floor, len(y)).astype(np.float32)
        clips.append(y.astype(np.float32))
    return clips


SYNTHETIC = _synthetic_clips()


def _librosa_bounds(y: np.ndarray) -> tuple[int, int]:
    _, (start, end) = librosa.effects.trim(y, top_db=30)
    return int(start), int(end)


def _assert_matches(y: np.ndarray) -> None:
    start, end = trim.trim_bounds(y)
    ref_start, ref_end = _librosa_bounds(y)
    assert abs(start - ref_start) <= 1 and abs(end - ref_end) <= 1, (
        f"numpy ({start}, {end}) vs librosa ({ref_start}, {ref_end})"
    )


@pytest.mark.parametrize("index", range(len(SYNTHETIC)))
def test_bounds_match_librosa_on_synthetic_clips(index):
    _assert_matches(SYNTHETIC[index])


def test_the_synthetic_set_actually_trims_something():
    """A set where nothing is trimmed would make the comparison vacuous."""
    trimmed = sum(trim.trim_bounds(y) != (0, len(y)) for y in SYNTHETIC)
    assert trimmed >= 30


@pytest.mark.parametrize("index", range(len(SYNTHETIC)))
def test_waveform_matches_the_librosa_pipeline(index):
    y = SYNTHETIC[index]
    np.testing.assert_allclose(trim.trim_waveform(y), features.trim_waveform(y), atol=1e-6)


_CLIPS_DIR = os.environ.get("VEMO_CLIPS_DIR")
_REAL = sorted(pathlib.Path(_CLIPS_DIR).glob("*.wav"))[:50] if _CLIPS_DIR else []


@pytest.mark.skipif(not _REAL, reason="set VEMO_CLIPS_DIR to compare on real speech")
@pytest.mark.parametrize("path", _REAL, ids=lambda p: p.name)
def test_bounds_match_librosa_on_real_speech(path):
    y, _ = librosa.load(path, sr=SR)
    _assert_matches(y.astype(np.float32))


# --- Edge cases ---------------------------------------------------------------


def test_silence_is_returned_whole_then_padded():
    y = np.zeros(SR, np.float32)
    out = trim.trim_waveform(y)
    assert out.shape == (THREE_S,) and not out.any()


def test_empty_input_becomes_three_seconds_of_silence():
    out = trim.trim_waveform(np.zeros(0, np.float32))
    assert out.shape == (THREE_S,) and out.dtype == np.float32 and not out.any()
    assert trim.trim_bounds(np.zeros(0, np.float32)) == (0, 0)


@pytest.mark.parametrize("n", [1, 100, 511, 512, 2047, 2048, 2049])
def test_shorter_than_a_frame_matches_librosa(n):
    y = _speechlike(np.random.default_rng(n), n / SR + 1e-9, 0.5)[:n]
    if not np.abs(y).max():
        y[0] = 0.1
    _assert_matches(y)
    assert trim.trim_waveform(y).shape == (THREE_S,)


def test_long_clip_is_cropped_to_three_seconds_from_the_speech_onset():
    rng = np.random.default_rng(7)
    y = np.concatenate([np.zeros(SR, np.float32), _speechlike(rng, 6.0, 0.8)])
    start, _ = trim.trim_bounds(y)
    out = trim.trim_waveform(y)
    assert out.shape == (THREE_S,)
    np.testing.assert_array_equal(out, y[start : start + THREE_S])


def test_exactly_three_seconds_of_speech_is_unchanged_in_length():
    y = _speechlike(np.random.default_rng(3), 3.0, 0.5)
    assert trim.trim_waveform(y).shape == (THREE_S,)


def test_a_single_click_in_silence_keeps_only_the_frames_around_it():
    y = np.zeros(2 * SR, np.float32)
    y[SR] = 1.0
    _assert_matches(y)
    start, end = trim.trim_bounds(y)
    assert start < SR < end and end - start <= 3 * trim.FRAME_LENGTH


def test_constant_signal_is_not_trimmed():
    y = np.full(SR, 0.25, np.float32)
    _assert_matches(y)


def test_output_is_float32_whatever_comes_in():
    y = (_speechlike(np.random.default_rng(1), 1.0, 0.5)).astype(np.float64)
    assert trim.trim_waveform(y).dtype == np.float32


def test_stereo_is_rejected():
    with pytest.raises(ValueError, match="mono"):
        trim.trim_bounds(np.zeros((2, SR), np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("fn", [trim.trim_waveform, trim.trim_bounds])
def test_non_finite_audio_is_rejected(bad, fn):
    y = np.zeros(SR, np.float32)
    y[10] = bad
    with pytest.raises(ValueError, match="NaN or infinity"):
        fn(y)


def test_the_module_does_not_import_librosa():
    """The point of the port: the served model can drop librosa."""
    import ast

    tree = ast.parse(pathlib.Path(trim.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert imported <= {"__future__", "numpy"}
