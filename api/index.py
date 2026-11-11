"""Speech-emotion API for the Vercel Python runtime (roadmap voice 5d).

The browser records, resamples to 16 kHz mono and sends exactly 3 s as raw
float32 little-endian PCM, so the function needs no audio decoder: only NumPy,
ONNX and ONNX Runtime (no librosa, scikit-learn, pandas or torch).

    POST /predict   body: 48,000 float32 LE samples (192,000 bytes),
                    Content-Type: application/octet-stream (required; 415 otherwise)
    GET  /health    liveness; never loads the model
    GET  /ready     200 once the model file and head are present and the session loads

Configuration (environment):
    VEMO_MODEL_DIR     directory holding wavlm_pooled_fp16w.onnx and head.json
                       (default: ../models next to this file)
    MODEL_SHA256       if set, the model bytes must hash to this or /ready is 503; unset,
                       no hash is computed and model_sha256 is null
    ALLOWED_ORIGINS    comma-separated exact origins allowed by CORS (https only,
                       except http://localhost / 127.0.0.1 for development); default none
    RATE_LIMIT_PER_MIN requests per client IP per minute on /predict and /ready (default 20)
    VERCEL             set to 1 by Vercel; only then is x-real-ip trusted as the client IP
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "vemo").is_dir() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vemo.head import head_proba, validate_head  # noqa: E402
from vemo.trim import DURATION_S, SAMPLE_RATE, trim_waveform  # noqa: E402
from vemo.weights import load_model_bytes_from, sha256_bytes  # noqa: E402

log = logging.getLogger("vemo.api")

ONNX_FILE = "wavlm_pooled_fp16w.onnx"
HEAD_FILE = "head.json"
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)  # 48,000
BODY_BYTES = N_SAMPLES * 4  # 192,000
MAX_BODY = 256 * 1024
WINDOW_S = 60.0
NOTE = "Trained on acted English speech (RAVDESS). Not for decisions about people."


def model_dir() -> Path:
    return Path(os.getenv("VEMO_MODEL_DIR") or ROOT / "models")


# --------------------------------------------------------------------------- model


class Loaded(NamedTuple):
    session: object
    input_name: str
    head: dict
    #: sha256 of the model bytes, verified against MODEL_SHA256; None when no pin is set
    #: (hashing is skipped then; production deploys always set the pin).
    sha256: str | None


class ModelUnavailable(Exception):
    """The model cannot serve. ``permanent`` failures are cached until the files or pin change."""

    def __init__(self, missing: list[str], reason: str, permanent: bool = True):
        super().__init__(reason)
        self.missing, self.reason, self.permanent = missing, reason, permanent


_lock = threading.Lock()
#: One slot per warm instance: (key, Loaded | ModelUnavailable). The key holds the
#: files' size and mtime and the pinned hash. Only permanent states are stored
#: (loaded, file missing, sha mismatch, bad head); transient failures (I/O errors,
#: session creation, out of memory) are retried by the next request.
_cache: dict[str, tuple[tuple, object]] = {}


def _stat_key(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _make_session(data: bytes):
    """ONNX Runtime session from the model bytes (fp16-stored weights upcast in memory)."""
    import onnxruntime as ort

    return ort.InferenceSession(load_model_bytes_from(data), providers=["CPUExecutionProvider"])


def _load(onnx: Path, head_path: Path, expected: str) -> Loaded:
    missing = [name for name, p in (("model", onnx), ("head", head_path)) if not p.is_file()]
    if missing:
        raise ModelUnavailable(missing, f"missing: {', '.join(missing)}")
    try:
        head_text = _read_bytes(head_path).decode("utf-8")
    except OSError as exc:
        raise ModelUnavailable(["head"], f"head.json could not be read ({type(exc).__name__})", False) from None
    except UnicodeDecodeError:
        raise ModelUnavailable(["head"], "head.json is not UTF-8") from None
    try:
        head = json.loads(head_text)
        dim = validate_head(head)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ModelUnavailable(["head"], f"head.json is invalid: {exc}") from None
    try:
        data = _read_bytes(onnx)  # read once: hashed and parsed from the same bytes
    except OSError as exc:
        raise ModelUnavailable(["model"], f"model could not be read ({type(exc).__name__})", False) from None
    digest = None
    if expected:
        digest = sha256_bytes(data)
        if digest != expected:
            raise ModelUnavailable(["model"], "model sha256 does not match MODEL_SHA256")
    try:
        session = _make_session(data)
    except Exception as exc:  # MemoryError, onnx / onnxruntime errors: retried next request
        log.exception("model session failed to load")
        raise ModelUnavailable(["session"], f"model session failed to load ({type(exc).__name__})",
                               False) from None
    finally:
        del data
    out_dim = session.get_outputs()[0].shape[-1]
    if isinstance(out_dim, int) and out_dim != dim:
        raise ModelUnavailable(["head"], f"head expects {dim} features, model gives {out_dim}")
    return Loaded(session, session.get_inputs()[0].name, head, digest)


def get_model() -> Loaded:
    """The cached model, loading it on first use; raises ``ModelUnavailable``."""
    directory = model_dir()
    onnx, head_path = directory / ONNX_FILE, directory / HEAD_FILE
    expected = os.getenv("MODEL_SHA256", "").strip().lower()
    key = (str(directory), _stat_key(onnx), _stat_key(head_path), expected)
    with _lock:
        hit = _cache.get("model")
        if hit is not None and hit[0] == key:
            value = hit[1]
        else:
            try:
                value = _load(onnx, head_path, expected)
            except ModelUnavailable as exc:
                if not exc.permanent:
                    _cache.pop("model", None)
                    raise
                value = exc
            _cache["model"] = (key, value)
    if isinstance(value, ModelUnavailable):
        raise value
    return value


def reset_model_cache() -> None:
    with _lock:
        _cache.clear()


# --------------------------------------------------------------------------- rate limit


class SlidingWindowLimiter:
    """At most ``limit`` hits per ``window`` seconds per key, in memory (per warm instance)."""

    MAX_KEYS = 10_000

    def __init__(self, window: float = WINDOW_S, clock=time.monotonic):
        self.window, self.clock = window, clock
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def hit(self, key: str, limit: int) -> int:
        """Record a hit; return 0 if allowed, else seconds until the next slot frees."""
        now = self.clock()
        with self.lock:
            if len(self.hits) >= self.MAX_KEYS:
                self._sweep(now)
            q = self.hits.setdefault(key, deque())
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= limit:
                return max(1, math.ceil(q[0] + self.window - now))
            q.append(now)
            return 0

    def _sweep(self, now: float) -> None:
        for k in [k for k, q in self.hits.items() if not q or q[-1] <= now - self.window]:
            del self.hits[k]
        if len(self.hits) >= self.MAX_KEYS:  # all active: drop the oldest to bound memory
            for k in sorted(self.hits, key=lambda k: self.hits[k][-1])[: len(self.hits) // 2]:
                del self.hits[k]

    def reset(self) -> None:
        with self.lock:
            self.hits.clear()


def client_ip(request: Request) -> str:
    """x-real-ip is set by Vercel's edge; anywhere else it is client-controlled, so ignore it."""
    if os.getenv("VERCEL") == "1":
        real = request.headers.get("x-real-ip", "").strip()
        if real:
            return real
    return request.client.host if request.client else "unknown"


def rate_limit() -> int:
    try:
        return max(1, int(os.getenv("RATE_LIMIT_PER_MIN", "20")))
    except ValueError:
        return 20


# --------------------------------------------------------------------------- CORS

_ORIGIN = re.compile(r"^(https://[a-z0-9.-]+(:\d{1,5})?|http://(localhost|127\.0\.0\.1)(:\d{1,5})?)$")


def allowed_origins(raw: str | None = None) -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "") if raw is None else raw
    good = []
    for origin in (o.strip() for o in raw.split(",")):
        if not origin:
            continue
        if _ORIGIN.match(origin):
            good.append(origin)
        else:
            log.warning("ignoring ALLOWED_ORIGINS entry %r (exact https origins only)", origin)
    return good


# --------------------------------------------------------------------------- app


def _error(status: int, detail: str, **headers: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers or None)


_ERR = {"content": {"application/json": {"schema": {
    "type": "object", "properties": {"detail": {"type": "string"}}}}}}

PREDICT_OPENAPI = {
    "requestBody": {
        "required": True,
        "description": f"{N_SAMPLES} float32 little-endian samples (16 kHz mono, 3 s), {BODY_BYTES} bytes.",
        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
    }
}


def _infer(wave: np.ndarray) -> dict:
    m = get_model()
    t = time.perf_counter()
    x = trim_waveform(wave, sr=SAMPLE_RATE)
    emb = m.session.run(None, {m.input_name: x[None]})[0]
    p = head_proba(m.head, np.asarray(emb, dtype=np.float64).reshape(1, -1))[0]
    ms = (time.perf_counter() - t) * 1000
    if not np.all(np.isfinite(p)):
        log.error("model produced non-finite probabilities")
        raise ModelUnavailable(["session"], "model produced non-finite output", False)
    classes = m.head["classes"]
    order = np.argsort(-p, kind="stable")
    return {
        "emotion": classes[order[0]],
        "probabilities": {classes[i]: round(float(p[i]), 4) for i in order},
        "inference_ms": round(ms, 1),
        "model_sha256": m.sha256,
        "note": NOTE,
    }


SILENCE_PEAK = 1e-3  # same floor as the web client
TONE_BAND_HZ = 30.0  # half-width of the band around the strongest frequency
TONE_FRACTION = 0.9  # more of the energy than this in that one band is a tone, not a voice


def tone_fraction(wave: np.ndarray) -> float:
    """Share of the signal's energy within TONE_BAND_HZ of its strongest frequency.

    A voice spreads its energy over harmonics and noise, so even a held vowel stays well
    under 0.9. A test tone, a hum or a whistle sits near 1.0, and so does a constant (DC)
    signal, which has no energy at all once its mean is removed.
    """
    x = wave.astype(np.float64)
    x -= x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    total = float(power.sum())
    if total <= 1e-12:
        return 1.0
    peak = int(np.argmax(power))
    half = max(1, round(TONE_BAND_HZ * len(x) / SAMPLE_RATE))
    return float(power[max(0, peak - half) : peak + half + 1].sum()) / total


def create_app() -> FastAPI:
    app = FastAPI(
        title="Speech emotion API (RAVDESS-trained, research use)",
        version="1.0.0",
        description=NOTE,
        docs_url=None,
        redoc_url=None,
    )
    origins = allowed_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            expose_headers=["Retry-After"],
            max_age=600,
        )
    limiter = SlidingWindowLimiter()
    app.state.limiter = limiter

    def limited(request: Request) -> JSONResponse | None:
        wait = limiter.hit(client_ip(request), rate_limit())
        if wait:
            return _error(429, "Too many requests; try again shortly.", **{"Retry-After": str(wait)})
        return None

    @app.get("/health", responses={200: {"description": "The function is up."}})
    def health():
        return {"status": "ok"}

    @app.get(
        "/ready",
        responses={200: {"description": "Model loaded."}, 429: _ERR,
                   503: {"description": "Model file, head or session unavailable."}},
    )
    def ready(request: Request):
        if (blocked := limited(request)) is not None:
            return blocked
        try:
            m = get_model()
        except ModelUnavailable as exc:
            return JSONResponse({"ready": False, "missing": exc.missing, "reason": exc.reason},
                                status_code=503)
        return {"ready": True, "model_sha256": m.sha256, "classes": len(m.head["classes"])}

    @app.post(
        "/predict",
        openapi_extra=PREDICT_OPENAPI,
        responses={200: {"description": "Prediction."}, 400: _ERR, 413: _ERR, 415: _ERR,
                   429: _ERR, 503: _ERR},
    )
    async def predict(request: Request):
        if (blocked := limited(request)) is not None:
            return blocked
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/octet-stream":
            return _error(415, "Send Content-Type: application/octet-stream.")
        declared = request.headers.get("content-length")
        if declared is not None:
            declared = declared.strip()
            if not (declared.isascii() and declared.isdigit()):
                return _error(400, "Invalid Content-Length.")
            if int(declared) > MAX_BODY:
                return _error(413, "Body exceeds 256 KB.")
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY:
                return _error(413, "Body exceeds 256 KB.")
        if len(body) != BODY_BYTES:
            return _error(400, f"Send exactly {N_SAMPLES} float32 little-endian samples ({BODY_BYTES} bytes); "
                               f"got {len(body)} bytes.")
        wave = np.frombuffer(bytes(body), dtype="<f4").astype(np.float32)  # native, writable
        if not np.all(np.isfinite(wave)):
            return _error(400, "Audio contains NaN or infinity.")
        # The model always names an emotion, silence included ("sad", probability 1.0). The web
        # page refuses silence before sending (isSilent in web/src/audio.ts); other callers get
        # the same answer here. The trim is relative to the loudest frame, so a floor is needed.
        if float(np.max(np.abs(wave))) < SILENCE_PEAK:
            return _error(400, "No speech in the audio: it is silent or too quiet.")
        # A steady tone is not speech either, and the model would still name an emotion for it.
        if tone_fraction(wave) > TONE_FRACTION:
            return _error(400, "No speech in the audio: it is a single steady tone.")
        try:
            # Loading (first request) and inference are CPU-bound: keep them off the event loop.
            return await run_in_threadpool(_infer, wave)
        except ModelUnavailable:
            return _error(503, "Model is not available.")

    return app


app = create_app()
