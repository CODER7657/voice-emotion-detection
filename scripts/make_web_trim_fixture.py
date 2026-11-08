"""Write web/src/trim-fixture.json: the expected trim bounds from src/vemo/trim.py.

The web client ports the trim to TypeScript; its unit test rebuilds each signal from
these segment recipes and must find the same (start, end). tests/test_web_trim_fixture.py
fails if this file goes stale.

    python scripts/make_web_trim_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vemo.trim import trim_bounds  # noqa: E402

OUT = ROOT / "web" / "src" / "trim-fixture.json"

# Each segment: [samples, amplitude, frequency Hz]; amplitude 0 is silence.
CASES = {
    "speech_between_silences": [[8000, 0, 0], [24000, 0.5, 220], [6000, 0, 0]],
    "quiet_lead_in": [[5000, 0.001, 100], [20000, 0.4, 330], [3000, 0, 0]],
    "two_bursts_with_a_gap": [[4000, 0, 0], [6000, 0.3, 440], [8000, 0, 0], [6000, 0.6, 180]],
    "longer_than_three_seconds": [[2000, 0, 0], [60000, 0.2, 250], [2000, 0, 0]],
    "all_loud": [[30000, 0.5, 200]],
    "silence": [[16000, 0, 0]],
    "shorter_than_a_frame": [[1000, 0.5, 300]],
}


def build(segments: list[list[float]], sr: int = 16_000) -> np.ndarray:
    parts = [
        (amp * np.sin(2 * np.pi * freq * np.arange(int(n)) / sr)).astype(np.float32)
        for n, amp, freq in segments
    ]
    return np.concatenate(parts)


def fixture() -> dict:
    return {
        name: {"segments": segs, "bounds": list(trim_bounds(build(segs)))}
        for name, segs in CASES.items()
    }


def main() -> int:
    OUT.write_text(json.dumps(fixture(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
