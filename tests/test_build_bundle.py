"""scripts/build_vercel_bundle.py builds a deploy dir that runs on its own."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("fastapi")

from conftest import write_standin_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_vercel_bundle as bvb  # noqa: E402


@pytest.fixture(scope="module")
def standin(tmp_path_factory):
    return write_standin_model(tmp_path_factory.mktemp("bundle_model")) / "wavlm_pooled_fp16w.onnx"


def test_bundle_layout_and_config(tmp_path, standin):
    out = bvb.build(standin, tmp_path / "bundle", region="iad1")
    files = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())
    assert files == [
        ".python-version", "api/index.py", "models/head.json", "models/wavlm_pooled_fp16w.onnx",
        "pyproject.toml", "requirements.txt", "src/vemo/__init__.py", "src/vemo/head.py",
        "src/vemo/trim.py", "src/vemo/weights.py", "vercel.json",
    ]
    cfg = json.loads((out / "vercel.json").read_text())
    assert cfg["regions"] == ["iad1"]
    fn = cfg["functions"]["api/index.py"]
    assert 10 <= fn["maxDuration"] <= 300 and "models/**" in fn["includeFiles"]
    assert "memory" not in fn  # not settable in vercel.json under Fluid compute
    pyproject = (out / "pyproject.toml").read_text()
    assert 'entrypoint = "api.index:app"' in pyproject and ">=3.12,<3.13" in pyproject
    assert (out / ".python-version").read_text().strip() == "3.12"
    assert (out / "requirements.txt").read_text() == (ROOT / "api" / "requirements.txt").read_text()
    for heavy in ("librosa", "torch", "scikit-learn", "sklearn", "pandas"):
        assert heavy not in pyproject


def test_bundle_app_serves_from_its_own_files(tmp_path, standin):
    """Run the bundled app in a fresh interpreter: it must find vemo and models/ inside the bundle."""
    out = bvb.build(standin, tmp_path / "bundle", head=standin.parent / "head.json")
    code = (
        "import sys, importlib.util, numpy as np\n"
        "spec = importlib.util.spec_from_file_location('idx', 'api/index.py')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['idx'] = m; spec.loader.exec_module(m)\n"
        "import vemo, pathlib\n"
        "assert pathlib.Path(vemo.__file__).resolve().is_relative_to(pathlib.Path.cwd().resolve()), vemo.__file__\n"
        "from fastapi.testclient import TestClient\n"
        "c = TestClient(m.app)\n"
        "assert c.get('/ready').status_code == 200, c.get('/ready').text\n"
        "r = c.post('/predict', content=np.zeros(48000, '<f4').tobytes(),"
        " headers={'Content-Type': 'application/octet-stream'})\n"
        "assert r.status_code == 200, r.text\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("VEMO_MODEL_DIR", "MODEL_SHA256", "PYTHONPATH")}
    res = subprocess.run([sys.executable, "-c", code], cwd=out, env=env,
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


def test_sha_mismatch_refuses_to_build(tmp_path, standin):
    with pytest.raises(ValueError, match="does not match"):
        bvb.build(standin, tmp_path / "bundle", sha256="0" * 64)
    assert not (tmp_path / "bundle").exists()


def test_missing_model_refuses_to_build(tmp_path):
    with pytest.raises(FileNotFoundError):
        bvb.build(tmp_path / "nope.onnx", tmp_path / "bundle")


def test_will_not_delete_an_unrelated_directory(tmp_path, standin):
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        bvb.build(standin, precious)
    assert (precious / "keep.txt").exists()


def test_rebuild_replaces_previous_bundle(tmp_path, standin):
    out = bvb.build(standin, tmp_path / "bundle")
    (out / "stale.txt").write_text("old")
    bvb.build(standin, out)
    assert not (out / "stale.txt").exists()


def test_main_reports_size_and_passes_under_limit(tmp_path, standin, capsys):
    assert bvb.main(["--model", str(standin), "--out", str(tmp_path / "b")]) == 0
    assert "of 500 MB Vercel Python limit" in capsys.readouterr().out


def test_main_fails_over_limit(tmp_path, standin, monkeypatch):
    monkeypatch.setattr(bvb, "LIMIT_MB", 0.001)
    assert bvb.main(["--model", str(standin), "--out", str(tmp_path / "b")]) == 1


def test_missing_head_refuses_to_build(tmp_path, standin):
    with pytest.raises(FileNotFoundError, match="head"):
        bvb.build(standin, tmp_path / "bundle", head=tmp_path / "nope.json")


# ---------------------------------------------------------------- fetch mode (Hobby: 100 MB upload cap)

URL = "https://github.com/CODER7657/voice-emotion-detection/releases/download/model-v1/wavlm_pooled_fp16w.onnx"


def _load_generated(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("generated_vercel_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, data):
        import io

        self._buf = io.BytesIO(data)

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_mode_leaves_model_out_and_wires_build_step(tmp_path, standin):
    from vemo.weights import sha256_file

    out = bvb.build(standin, tmp_path / "bundle", fetch_url=URL, head=standin.parent / "head.json")
    assert not (out / "models" / "wavlm_pooled_fp16w.onnx").exists()
    assert 'build = "python vercel_build.py"' in (out / "pyproject.toml").read_text()
    gen = _load_generated(out / "vercel_build.py")
    assert gen.MODEL_URL == URL and gen.MODEL_SHA256 == sha256_file(standin)
    assert gen.DEST == (out / "models" / "wavlm_pooled_fp16w.onnx").resolve()
    assert bvb.dir_size(out) < 1e6  # tiny upload


def test_fetch_mode_end_to_end_build_then_serve(tmp_path, standin):
    """Simulate Vercel's build step with a fake download, then serve from the bundle."""
    out = bvb.build(standin, tmp_path / "bundle", fetch_url=URL, head=standin.parent / "head.json")
    gen = _load_generated(out / "vercel_build.py")
    data = standin.read_bytes()
    gen.fetch(gen.MODEL_URL, gen.MODEL_SHA256, gen.DEST, opener=lambda url, timeout: _FakeResponse(data))
    assert gen.DEST.read_bytes() == data
    code = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('idx', 'api/index.py')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['idx'] = m; spec.loader.exec_module(m)\n"
        "from fastapi.testclient import TestClient\n"
        "r = TestClient(m.app).get('/ready'); assert r.status_code == 200, r.text; print('ok')\n"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("VEMO_MODEL_DIR", "MODEL_SHA256")}
    res = subprocess.run([sys.executable, "-c", code], cwd=out, env=env, capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, res.stderr


def test_fetch_url_must_be_https(tmp_path, standin):
    with pytest.raises(ValueError, match="https"):
        bvb.build(standin, tmp_path / "bundle", fetch_url="http://example.com/m.onnx")


def test_main_fetch_mode_counts_the_model(tmp_path, standin, capsys):
    assert bvb.main(["--model", str(standin), "--out", str(tmp_path / "b"), "--fetch-url", URL,
                     "--head", str(standin.parent / "head.json")]) == 0
    assert "fetched during the Vercel build" in capsys.readouterr().out


import vercel_fetch_model as vfm  # noqa: E402


def test_fetch_rejects_wrong_hash_and_leaves_nothing(tmp_path):
    dest = tmp_path / "models" / "m.onnx"
    with pytest.raises(ValueError, match="pinned"):
        vfm.fetch(URL, "0" * 64, dest, opener=lambda url, timeout: _FakeResponse(b"tampered"))
    assert not dest.exists() and not dest.with_suffix(".part").exists()


def test_fetch_refuses_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="https"):
        vfm.fetch("http://x/m.onnx", "0" * 64, tmp_path / "m")
    with pytest.raises(ValueError, match="sha256"):
        vfm.fetch(URL, "abc", tmp_path / "m")


def test_fetch_reuses_a_verified_file(tmp_path):
    import hashlib

    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"model")
    sha = hashlib.sha256(b"model").hexdigest()
    assert vfm.fetch(URL, sha.upper(), dest, opener=lambda *a, **k: pytest.fail("downloaded again")) == sha


def test_fetch_replaces_a_stale_file(tmp_path):
    import hashlib

    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"old")
    sha = hashlib.sha256(b"new").hexdigest()
    vfm.fetch(URL, sha, dest, opener=lambda url, timeout: _FakeResponse(b"new"))
    assert dest.read_bytes() == b"new"


def test_fetch_network_error_propagates_and_cleans_up(tmp_path):
    dest = tmp_path / "m.onnx"

    def boom(url, timeout):
        raise OSError("network down")

    with pytest.raises(OSError):
        vfm.fetch(URL, "0" * 64, dest, opener=boom)
    assert not dest.with_suffix(".part").exists()
