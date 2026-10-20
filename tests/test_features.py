import numpy as np
import soundfile as sf

from vemo import features as F


def tone(seconds: float, sr: int = F.SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_load_clip_skips_the_opening_and_pads_to_three_seconds(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, tone(1.0), F.SAMPLE_RATE)
    y = F.load_clip(str(path))
    assert y.shape == (int(F.SAMPLE_RATE * F.DURATION_S),)
    assert np.abs(y[: F.SAMPLE_RATE // 2]).max() > 0.1  # the 0.5 s left after the offset
    assert np.abs(y[F.SAMPLE_RATE:]).max() == 0  # the rest is padding


def test_feature_shapes():
    y = F.fix_length(tone(3.0), int(F.SAMPLE_RATE * F.DURATION_S))
    assert F.log_mel(y).shape[0] == F.N_MELS
    assert F.mfcc_stats(y).shape == (F.N_MFCC * 3 * 2,)


def test_augment_keeps_length_and_is_reproducible():
    y = tone(3.0)
    a = F.augment(y, np.random.default_rng([0, 5]))
    b = F.augment(y, np.random.default_rng([0, 5]))
    assert a.shape == y.shape
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, y)
