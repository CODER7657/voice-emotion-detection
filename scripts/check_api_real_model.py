"""Local check of the live API with the REAL model on RAVDESS clips (not run in CI).

    python scripts/check_api_real_model.py --ravdess D:/datasets/ravdess [--n 20]

Sends what the browser sends: the first 3 s of each clip as 16 kHz mono float32,
zero-padded to 48,000 samples, raw in the body; the server trims. Audio is read
and resampled here in the harness only (librosa), never in the API. Reports
agreement with the filename label (the served head was fitted on all 1,440
RAVDESS clips, so this is an end-to-end smoke check, not held-out accuracy),
p50/p95 of the server's inference_ms, and the
cold-start time of the first request (model load + upcast + first inference).
Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import librosa

    from vemo.ravdess import EMOTIONS
    from vemo.weights import sha256_file

    clips = sorted(args.ravdess.glob("Actor_*/03-01-*.wav"))
    if not clips:
        raise SystemExit(f"no RAVDESS speech clips under {args.ravdess}")
    rng = np.random.default_rng(args.seed)
    # balanced: n/8 clips per emotion, random actors
    by_emotion: dict[str, list[Path]] = {}
    for c in clips:
        by_emotion.setdefault(EMOTIONS[c.name.split("-")[2]], []).append(c)
    per = max(1, args.n // len(by_emotion))
    chosen = [p for e in sorted(by_emotion) for p in rng.choice(by_emotion[e], per, replace=False)]
    chosen += list(rng.choice([c for c in clips if c not in chosen], args.n - len(chosen), replace=False))

    os.environ["VEMO_MODEL_DIR"] = str(args.models)
    os.environ["MODEL_SHA256"] = sha256_file(args.models / "wavlm_pooled_fp16w.onnx")
    os.environ["RATE_LIMIT_PER_MIN"] = "100000"
    spec = importlib.util.spec_from_file_location("api_index", ROOT / "api" / "index.py")
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    from fastapi.testclient import TestClient

    client = TestClient(api.create_app())

    hits, ms, rows = 0, [], []
    cold = None
    for i, path in enumerate(chosen):
        y, _ = librosa.load(path, sr=16_000, mono=True)
        y = y[:48_000]
        y = np.pad(y, (0, 48_000 - len(y))).astype("<f4")
        t = time.perf_counter()
        r = client.post("/predict", content=y.tobytes(), headers={"Content-Type": "application/octet-stream"})
        wall = (time.perf_counter() - t) * 1000
        r.raise_for_status()
        body = r.json()
        label = EMOTIONS[path.name.split("-")[2]]
        if i == 0:
            cold = wall
        else:
            ms.append(body["inference_ms"])
        hits += body["emotion"] == label
        rows.append(f"{path.name}  {label:>9} -> {body['emotion']:<9} {body['inference_ms']:7.1f} ms")
    print("\n".join(rows))
    print(f"\nmodel_sha256 {body['model_sha256']}")
    print(f"agreement with label: {hits}/{len(chosen)} = {hits / len(chosen):.0%}")
    print(f"inference_ms (warm, n={len(ms)}): p50 {np.percentile(ms, 50):.1f}  p95 {np.percentile(ms, 95):.1f}")
    print(f"cold start (first request, wall incl. load+upcast+sha256): {cold:.0f} ms")
    print(f"/ready -> {client.get('/ready').status_code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
