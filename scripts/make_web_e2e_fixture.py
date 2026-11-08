"""Write the web e2e audio fixtures (synthetic: no dataset audio is redistributed).

* clip.wav: 44.1 kHz stereo 16-bit, 0.8 s silence, 2.2 s of a speech-like voiced sound
  (harmonics of a gliding 140-190 Hz pitch with a syllable-rate envelope), 0.6 s silence.
* clip.expected.f32: what the training pipeline makes of it (librosa.load at 16 kHz mono,
  then vemo.trim.trim_waveform): the e2e test checks the browser sends the same window.
* silence.wav: 2 s of digital silence; garbage.wav: not audio.

    python scripts/make_web_e2e_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vemo.trim import trim_waveform  # noqa: E402

OUT = ROOT / "web" / "e2e" / "fixtures"
SR = 44_100


def voiced(seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 50 * np.sin(2 * np.pi * 0.7 * t)  # gliding pitch
    phase = 2 * np.pi * np.cumsum(f0) / sr
    tone = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = 0.5 * (1 - np.cos(2 * np.pi * 4 * t)) * np.minimum(1, t / 0.05)  # 4 syllables/s
    return (0.25 * tone * envelope).astype(np.float32)


def main() -> int:
    import librosa

    OUT.mkdir(parents=True, exist_ok=True)
    mono = np.concatenate([np.zeros(int(0.8 * SR), np.float32), voiced(2.2), np.zeros(int(0.6 * SR), np.float32)])
    stereo = np.stack([mono, 0.9 * mono], axis=1)
    sf.write(OUT / "clip.wav", stereo, SR, subtype="PCM_16")
    y, _ = librosa.load(OUT / "clip.wav", sr=16_000, mono=True)
    trim_waveform(y).astype("<f4").tofile(OUT / "clip.expected.f32")
    sf.write(OUT / "silence.wav", np.zeros(2 * 16_000, np.float32), 16_000, subtype="PCM_16")
    (OUT / "garbage.wav").write_bytes(b"this is not audio\n" * 64)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
