"""The deployed API (api/index.py) against a tiny stand-in model.

The stand-in (see conftest.write_standin_model) has the real model's interface:
48,000 float32 samples in, a pooled embedding out, an 8-class head on top, and
fp16-stored weights that the loader upcasts. The real model is checked locally
by scripts/check_api_real_model.py.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import API_CLASSES

N = 48_000
ROOT = Path(__file__).resolve().parents[1]


def pcm(n: int = N, seed: int = 0) -> bytes:
    t = np.arange(n) / 16_000
    g = np.random.default_rng(seed)
    y = 0.3 * np.sin(2 * np.pi * 220 * t) * (t > 0.4) + 0.01 * g.normal(size=n)
    return y.astype("<f4").tobytes()


OCTET = {"Content-Type": "application/octet-stream"}


@pytest.fixture
def make_client(api_env):
    from fastapi.testclient import TestClient

    def make():
        return TestClient(api_env.create_app())

    return make


@pytest.fixture
def client(make_client):
    return make_client()


def post(client, body, headers=None):
    return client.post("/predict", content=body, headers={**OCTET, **(headers or {})})


# ---------------------------------------------------------------- happy path


def test_predict_returns_all_classes_sorted(client, standin_model_dir, monkeypatch):
    from vemo.weights import sha256_file

    monkeypatch.setenv("MODEL_SHA256", sha256_file(standin_model_dir / "wavlm_pooled_fp16w.onnx"))
    r = post(client, pcm())
    assert r.status_code == 200, r.text
    body = r.json()
    probs = body["probabilities"]
    assert sorted(probs) == sorted(API_CLASSES)
    values = list(probs.values())
    assert values == sorted(values, reverse=True)
    assert math.isclose(sum(values), 1.0, abs_tol=1e-3)
    assert all(v == round(v, 4) for v in values)
    assert body["emotion"] == next(iter(probs))
    assert isinstance(body["inference_ms"], (int, float)) and body["inference_ms"] >= 0
    assert body["model_sha256"] == sha256_file(standin_model_dir / "wavlm_pooled_fp16w.onnx")
    assert body["note"] == "Trained on acted English speech (RAVDESS). Not for decisions about people."


def test_prediction_matches_trim_session_head_by_hand(client, api_env, standin_model_dir):
    """The handler is exactly trim -> session -> head, on the upcast fp16 weights."""
    import onnxruntime as ort

    from vemo.head import head_proba
    from vemo.trim import trim_waveform
    from vemo.weights import load_model_bytes

    raw = pcm(seed=3)
    session = ort.InferenceSession(load_model_bytes(standin_model_dir / "wavlm_pooled_fp16w.onnx"))
    x = trim_waveform(np.frombuffer(raw, dtype="<f4"))
    head = json.loads((standin_model_dir / "head.json").read_text())
    p = head_proba(head, session.run(None, {"input_values": x[None]})[0])[0]
    got = post(client, raw).json()["probabilities"]
    for i, name in enumerate(head["classes"]):
        assert got[name] == pytest.approx(round(float(p[i]), 4), abs=1e-4)


def test_silence_is_accepted(client):
    assert post(client, bytes(N * 4)).status_code == 200


def test_content_type_with_parameters_is_accepted(client):
    r = client.post("/predict", content=pcm(), headers={"Content-Type": "application/octet-stream; x=1"})
    assert r.status_code == 200


def test_model_sha256_is_null_when_no_pin_is_set(client, api_env, monkeypatch):
    """Without MODEL_SHA256 the 190 MB hash is skipped entirely."""
    monkeypatch.setattr(api_env, "sha256_bytes", lambda data: pytest.fail("hashed without a pin"))
    r = post(client, pcm())
    assert r.status_code == 200 and r.json()["model_sha256"] is None


def test_missing_content_type_is_415(client):
    r = client.post("/predict", content=pcm())
    assert r.request.headers.get("content-type") is None
    assert r.status_code == 415


@pytest.mark.parametrize("ctype", ["audio/wav", "text/plain", "application/json",
                                   "application/x-www-form-urlencoded", "multipart/form-data; boundary=x",
                                   "application/octet-streamx"])
def test_other_content_types_are_415(client, ctype):
    r = client.post("/predict", content=pcm(), headers={"Content-Type": ctype})
    assert r.status_code == 415


# ---------------------------------------------------------------- bad bodies


@pytest.mark.parametrize("n_bytes", [N * 4 - 4, N * 4 + 4, N * 4 - 1, N * 4 + 1, 1, 3, 4])
def test_wrong_length_is_400(client, n_bytes):
    r = post(client, pcm(N + 1)[:n_bytes])
    assert r.status_code == 400
    assert "48000" in r.json()["detail"]


def test_empty_body_is_400(client):
    assert post(client, b"").status_code == 400


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_is_400(client, bad):
    y = np.frombuffer(pcm(), dtype="<f4").copy()
    y[12_345] = bad
    r = post(client, y.tobytes())
    assert r.status_code == 400 and "NaN" in r.json()["detail"]


def test_big_endian_nan_pattern_is_caught(client):
    # bytes that are finite big-endian but NaN little-endian must still be rejected
    y = np.zeros(N, dtype="<f4")
    y.view(np.uint32)[0] = 0x7FC00000
    assert post(client, y.tobytes()).status_code == 400


def test_oversized_body_with_content_length_is_413(client):
    assert post(client, b"\0" * (256 * 1024 + 4)).status_code == 413


def test_oversized_declared_length_is_413_before_reading(client, api_env, monkeypatch):
    called = []
    monkeypatch.setattr(api_env, "get_model", lambda: called.append(1))
    r = post(client, b"\0" * 8, headers={"Content-Length": str(10 * 1024 * 1024)})
    assert r.status_code == 413 and not called


def test_oversized_body_without_content_length_is_413(client):
    def chunks():
        for _ in range(40):
            yield b"\0" * 8192  # 320 KB, streamed with chunked transfer encoding

    r = client.post("/predict", content=chunks(), headers=OCTET)
    assert r.request.headers.get("content-length") is None
    assert r.status_code == 413


def test_correct_body_without_content_length_is_accepted(client):
    raw = pcm()

    def chunks():
        for i in range(0, len(raw), 50_000):
            yield raw[i:i + 50_000]

    r = client.post("/predict", content=chunks(), headers=OCTET)
    assert r.request.headers.get("content-length") is None
    assert r.status_code == 200


def test_body_at_limit_but_wrong_length_is_400(client):
    assert post(client, b"\0" * (256 * 1024)).status_code == 400


# ---------------------------------------------------------------- health / ready


def test_health_does_not_load_the_model(client, api_env, monkeypatch):
    monkeypatch.setattr(api_env, "_load", lambda *a: pytest.fail("health loaded the model"))
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert api_env._cache == {}


def test_ready_when_model_present(client, standin_model_dir, monkeypatch):
    from vemo.weights import sha256_file

    monkeypatch.setenv("MODEL_SHA256", sha256_file(standin_model_dir / "wavlm_pooled_fp16w.onnx"))
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_sha256"] == sha256_file(standin_model_dir / "wavlm_pooled_fp16w.onnx")


def test_ready_accepts_matching_pinned_sha(client, monkeypatch, standin_model_dir):
    from vemo.weights import sha256_file

    pinned = sha256_file(standin_model_dir / "wavlm_pooled_fp16w.onnx").upper()
    monkeypatch.setenv("MODEL_SHA256", pinned)
    assert client.get("/ready").status_code == 200


def test_ready_503_on_sha_mismatch_and_predict_refuses(client, monkeypatch):
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["missing"] == ["model"] and "sha256" in r.json()["reason"]
    assert post(client, pcm()).status_code == 503


@pytest.mark.parametrize("remove,missing", [
    (["wavlm_pooled_fp16w.onnx"], ["model"]),
    (["head.json"], ["head"]),
    (["wavlm_pooled_fp16w.onnx", "head.json"], ["model", "head"]),
])
def test_ready_503_names_what_is_missing(client, monkeypatch, tmp_path, standin_model_dir, remove, missing):
    import shutil

    for f in ("wavlm_pooled_fp16w.onnx", "head.json"):
        if f not in remove:
            shutil.copy(standin_model_dir / f, tmp_path / f)
    monkeypatch.setenv("VEMO_MODEL_DIR", str(tmp_path))
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["missing"] == missing and body["ready"] is False
    assert str(tmp_path) not in r.text  # no filesystem paths leaked


def test_ready_503_on_corrupt_model(client, monkeypatch, tmp_path, standin_model_dir):
    import shutil

    shutil.copy(standin_model_dir / "head.json", tmp_path / "head.json")
    (tmp_path / "wavlm_pooled_fp16w.onnx").write_bytes(b"not an onnx file")
    monkeypatch.setenv("VEMO_MODEL_DIR", str(tmp_path))
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["missing"] == ["session"]


def test_ready_503_on_head_that_does_not_fit(client, monkeypatch, tmp_path, standin_model_dir):
    import shutil

    shutil.copy(standin_model_dir / "wavlm_pooled_fp16w.onnx", tmp_path / "wavlm_pooled_fp16w.onnx")
    head = json.loads((standin_model_dir / "head.json").read_text())
    head["mean"], head["scale"] = head["mean"][:4], head["scale"][:4]
    head["coef"] = [row[:4] for row in head["coef"]]
    (tmp_path / "head.json").write_text(json.dumps(head))
    monkeypatch.setenv("VEMO_MODEL_DIR", str(tmp_path))
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["missing"] == ["head"]


def test_model_loads_once_per_instance(client, api_env, monkeypatch):
    calls = []
    real = api_env._load
    monkeypatch.setattr(api_env, "_load", lambda *a: calls.append(1) or real(*a))
    for _ in range(3):
        assert post(client, pcm()).status_code == 200
    assert client.get("/ready").status_code == 200
    assert len(calls) == 1


def test_failure_is_cached_until_files_change(client, api_env, monkeypatch, tmp_path, standin_model_dir):
    import shutil

    monkeypatch.setenv("VEMO_MODEL_DIR", str(tmp_path))
    calls = []
    real = api_env._load
    monkeypatch.setattr(api_env, "_load", lambda *a: calls.append(1) or real(*a))
    assert client.get("/ready").status_code == 503
    assert client.get("/ready").status_code == 503
    assert len(calls) == 1
    for f in ("wavlm_pooled_fp16w.onnx", "head.json"):
        shutil.copy(standin_model_dir / f, tmp_path / f)
    assert client.get("/ready").status_code == 200
    assert len(calls) == 2


def test_model_file_is_read_once_per_load(client, api_env, monkeypatch, standin_model_dir):
    from vemo.weights import sha256_file

    monkeypatch.setenv("MODEL_SHA256", sha256_file(standin_model_dir / "wavlm_pooled_fp16w.onnx"))
    reads = []
    real = api_env._read_bytes
    monkeypatch.setattr(api_env, "_read_bytes", lambda p: reads.append(p.name) or real(p))
    assert client.get("/ready").status_code == 200
    assert post(client, pcm()).status_code == 200
    assert reads == ["head.json", "wavlm_pooled_fp16w.onnx"]


def test_transient_session_failure_is_retried(client, api_env, monkeypatch):
    calls = []
    real = api_env._make_session

    def flaky(data):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("simulated")
        return real(data)

    monkeypatch.setattr(api_env, "_make_session", flaky)
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["missing"] == ["session"]
    assert "MemoryError" in r.json()["reason"]
    assert client.get("/ready").status_code == 200  # not cached: retried and succeeded
    assert post(client, pcm()).status_code == 200
    assert len(calls) == 2  # and then cached


def test_onnxruntime_error_on_predict_is_503_then_recovers(client, api_env, monkeypatch):
    calls = []
    real = api_env._make_session

    def flaky(data):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("onnxruntime: simulated allocation failure")
        return real(data)

    monkeypatch.setattr(api_env, "_make_session", flaky)
    assert post(client, pcm()).status_code == 503
    assert post(client, pcm()).status_code == 200


@pytest.mark.parametrize("which", ["wavlm_pooled_fp16w.onnx", "head.json"])
def test_read_error_is_503_and_retried(client, api_env, monkeypatch, which):
    calls = []
    real = api_env._read_bytes

    def flaky(path):
        if path.name == which and not calls:
            calls.append(1)
            raise PermissionError(13, "simulated EACCES")
        return real(path)

    monkeypatch.setattr(api_env, "_read_bytes", flaky)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["missing"] == (["model"] if which.endswith(".onnx") else ["head"])
    assert "PermissionError" in r.json()["reason"] and "EACCES" not in r.text
    assert client.get("/ready").status_code == 200


def test_read_error_on_predict_is_503_not_500(client, api_env, monkeypatch):
    def broken(path):
        raise OSError(5, "simulated EIO")

    monkeypatch.setattr(api_env, "_read_bytes", broken)
    assert post(client, pcm()).status_code == 503


def _with_head(tmp_path, standin_model_dir, mutate):
    import shutil

    shutil.copy(standin_model_dir / "wavlm_pooled_fp16w.onnx", tmp_path / "wavlm_pooled_fp16w.onnx")
    head = json.loads((standin_model_dir / "head.json").read_text())
    mutate(head)
    # allow_nan so Infinity/NaN reach the file, as a hand-edited head.json could
    (tmp_path / "head.json").write_text(json.dumps(head, allow_nan=True))
    return tmp_path


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda h: h["scale"].__setitem__(0, 0.0), id="zero-scale"),
    pytest.param(lambda h: h["mean"].__setitem__(3, "abc"), id="non-numeric-mean"),
    pytest.param(lambda h: h["coef"][2].__setitem__(1, float("inf")), id="inf-coef"),
    pytest.param(lambda h: h["intercept"].__setitem__(0, float("nan")), id="nan-intercept"),
])
def test_bad_head_values_are_503_never_500(client, monkeypatch, tmp_path, standin_model_dir, mutate):
    monkeypatch.setenv("VEMO_MODEL_DIR", str(_with_head(tmp_path, standin_model_dir, mutate)))
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["missing"] == ["head"]
    p = post(client, pcm())
    assert p.status_code == 503


def test_non_utf8_head_is_503(client, monkeypatch, tmp_path, standin_model_dir):
    import shutil

    shutil.copy(standin_model_dir / "wavlm_pooled_fp16w.onnx", tmp_path / "wavlm_pooled_fp16w.onnx")
    (tmp_path / "head.json").write_bytes(bytes([0xFF, 0xFE]) + b"{")
    monkeypatch.setenv("VEMO_MODEL_DIR", str(tmp_path))
    assert client.get("/ready").status_code == 503


def test_non_finite_model_output_is_503(client, api_env, monkeypatch):
    class NaNSession:
        def run(self, outputs, feeds):
            return [np.full((1, 16), np.nan, dtype=np.float32)]

    real = api_env.get_model
    monkeypatch.setattr(api_env, "get_model", lambda: real()._replace(session=NaNSession()))
    r = post(client, pcm())
    assert r.status_code == 503


# ---------------------------------------------------------------- rate limit


def test_21st_request_in_a_minute_is_429(client):
    for _ in range(20):
        assert post(client, pcm()).status_code == 200
    r = post(client, pcm())
    assert r.status_code == 429
    assert 1 <= int(r.headers["Retry-After"]) <= 60


def test_ready_shares_the_limit_but_health_does_not(client):
    for _ in range(20):
        assert client.get("/ready").status_code == 200
    assert client.get("/ready").status_code == 429
    assert post(client, pcm()).status_code == 429
    assert client.get("/health").status_code == 200


def test_spoofed_x_real_ip_is_ignored_off_vercel(client):
    for i in range(20):
        assert post(client, pcm(), {"x-real-ip": f"10.0.0.{i}"}).status_code == 200
    assert post(client, pcm(), {"x-real-ip": "10.0.1.1"}).status_code == 429


def test_x_real_ip_is_the_key_on_vercel(make_client, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    client = make_client()
    for _ in range(20):
        assert post(client, pcm(), {"x-real-ip": "203.0.113.7"}).status_code == 200
    assert post(client, pcm(), {"x-real-ip": "203.0.113.7"}).status_code == 429
    assert post(client, pcm(), {"x-real-ip": "203.0.113.8"}).status_code == 200


def test_limiter_window_slides(api_module):
    now = [1000.0]
    lim = api_module.SlidingWindowLimiter(window=60, clock=lambda: now[0])
    for _ in range(3):
        assert lim.hit("a", 3) == 0
    assert lim.hit("a", 3) == 60
    now[0] += 30
    assert lim.hit("a", 3) == 30
    now[0] += 30.001
    assert lim.hit("a", 3) == 0
    assert lim.hit("b", 3) == 0


def test_limiter_memory_is_bounded(api_module, monkeypatch):
    now = [0.0]
    lim = api_module.SlidingWindowLimiter(window=60, clock=lambda: now[0])
    monkeypatch.setattr(lim, "MAX_KEYS", 100)
    for i in range(1000):
        now[0] += 0.001
        lim.hit(f"ip{i}", 20)
    assert len(lim.hits) <= 100


def test_bad_rate_limit_env_falls_back(api_module, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "lots")
    assert api_module.rate_limit() == 20
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "-5")
    assert api_module.rate_limit() == 1


# ---------------------------------------------------------------- CORS


SITE = "https://emotion.example.app"


def test_no_cors_by_default(client):
    r = client.get("/health", headers={"Origin": SITE})
    assert "access-control-allow-origin" not in r.headers


def test_allowed_origin_is_echoed(make_client, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", f"{SITE}, http://localhost:5173")
    client = make_client()
    r = post(client, pcm(), {"Origin": SITE})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == SITE
    pre = client.options("/predict", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_other_origins_are_not_echoed(make_client, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", SITE)
    client = make_client()
    for origin in ("https://evil.example", SITE + ".evil.example", "http://emotion.example.app"):
        r = client.get("/health", headers={"Origin": origin})
        assert "access-control-allow-origin" not in r.headers
        pre = client.options("/predict", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in pre.headers


def test_429_still_carries_cors_and_exposes_retry_after(make_client, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", SITE)
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    client = make_client()
    client.get("/ready", headers={"Origin": SITE})
    r = client.get("/ready", headers={"Origin": SITE})
    assert r.status_code == 429 and r.headers["access-control-allow-origin"] == SITE
    assert "retry-after" in r.headers["access-control-expose-headers"].lower()


@pytest.mark.parametrize("raw,expected", [
    ("", []),
    ("https://a.example", ["https://a.example"]),
    ("https://a.example,https://b.example:8443", ["https://a.example", "https://b.example:8443"]),
    ("http://localhost:3000, http://127.0.0.1", ["http://localhost:3000", "http://127.0.0.1"]),
    ("http://a.example", []),          # plain http only for localhost
    ("https://a.example/", []),        # exact origins: no path
    ("*", []),
    ("https://a.example/path", []),
    ("null", []),
])
def test_allowed_origins_parsing(api_module, raw, expected):
    assert api_module.allowed_origins(raw) == expected


# ---------------------------------------------------------------- dependencies


def test_api_imports_without_heavy_libraries():
    """The deployed function must not need librosa, sklearn, pandas, scipy or torch."""
    code = (
        "import sys, importlib.abc\n"
        "BLOCK = {'librosa', 'sklearn', 'pandas', 'scipy', 'torch', 'numba', 'soundfile', 'matplotlib'}\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in BLOCK: raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('idx', r'{ROOT / 'api' / 'index.py'}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "import vemo.trim, vemo.weights, vemo.head\n"
        "print('ok', sorted(BLOCK & set(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok []"


def test_runtime_requirements_are_minimal_and_pinned():
    lines = [ln.strip() for ln in (ROOT / "api" / "requirements.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    names = {ln.split("==")[0].lower() for ln in lines}
    assert names == {"numpy", "onnxruntime", "onnx", "fastapi"}
    assert all("==" in ln for ln in lines)


def test_serve_head_proba_is_the_same_function():
    import os

    os.environ.setdefault("VEMO_NO_APP", "1")
    from vemo import head, serve

    assert serve.head_proba is head.head_proba


@pytest.mark.parametrize("value", ["abc", "-1", "1e3", "\u00b2"])
def test_invalid_content_length_is_400(api_env, value):
    """Driven at the ASGI level: HTTP clients refuse to send a malformed Content-Length."""
    import asyncio

    app = api_env.create_app()
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/predict", "raw_path": b"/predict", "query_string": b"",
             "root_path": "", "client": ("127.0.0.1", 1), "server": ("testserver", 80),
             "headers": [(b"content-type", b"application/octet-stream"),
                         (b"content-length", value.encode("utf-8"))]}
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 400
