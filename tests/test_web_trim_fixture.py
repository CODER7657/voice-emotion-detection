"""The web client's trim fixture matches src/vemo/trim.py (regenerate with the script)."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("make_web_trim_fixture", ROOT / "scripts" / "make_web_trim_fixture.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixture_is_current():
    module = _script()
    committed = json.loads(module.OUT.read_text(encoding="utf-8"))
    assert committed == module.fixture(), "run: python scripts/make_web_trim_fixture.py"


def test_fixture_covers_the_edge_cases():
    bounds = {name: tuple(case["bounds"]) for name, case in _script().fixture().items()}
    # librosa: on all zeros every frame is 0 dB relative to the (zero) reference, so all
    # of it is kept; the web client rejects silence by its peak instead.
    assert bounds["silence"] == (0, 16000)
    assert bounds["all_loud"] == (0, 30000)
    start, end = bounds["speech_between_silences"]
    assert 0 < start < 8000 and 32000 < end < 38000
