"""Serve the WavLM + logistic-regression model over HTTP, on CPU, without PyTorch.

    pip install -e ".[serve]"
    python scripts/export_onnx.py --ravdess <RAVDESS dir> --out models   # once; writes models/
    uvicorn vemo.serve:app --port 8000
    curl -F "file=@clip.wav" localhost:8000/predict

The embedding model is WavLM-base-plus exported to ONNX in fp32 (``scripts/export_onnx.py``).
An int8 copy was measured and rejected: it lost 9 points of macro-F1 and ran slower on CPU
(``results/export/export_report.json``). The classifier is a standardised logistic regression
stored as plain numbers in ``head.json``. Clips are trimmed of silence and cut or
padded to 3 s, as in training (``features.load_trimmed``).
"""

import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(os.getenv("VEMO_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
ONNX_FILE = "wavlm_pooled_fp32.onnx"
HEAD_FILE = "head.json"
MAX_BYTES = 10 * 1024 * 1024


def head_from_pipeline(pipeline) -> dict:
    """The numbers needed to reproduce a StandardScaler + LogisticRegression pipeline."""
    scaler, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
    return {"classes": [str(c) for c in lr.classes_], "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}


def head_proba(head: dict, X: np.ndarray) -> np.ndarray:
    """Class probabilities, matching ``pipeline.predict_proba`` for multinomial regression."""
    z = (X - np.asarray(head["mean"])) / np.asarray(head["scale"])
    logits = z @ np.asarray(head["coef"]).T + np.asarray(head["intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


class Model:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort

        self.head = json.loads((model_dir / HEAD_FILE).read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(model_dir / ONNX_FILE), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def embed(self, wave: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input: wave[None].astype(np.float32)})[0]

    def predict(self, wave: np.ndarray) -> dict:
        p = head_proba(self.head, self.embed(wave))[0]
        order = np.argsort(-p)
        return {"emotion": self.head["classes"][order[0]],
                "probabilities": {self.head["classes"][i]: round(float(p[i]), 4) for i in order}}


@lru_cache(maxsize=1)
def model() -> Model:
    return Model()


def create_app():
    from fastapi import FastAPI, File, HTTPException, UploadFile

    from vemo.features import load_trimmed

    app = FastAPI(title="Speech emotion (RAVDESS-trained, research use)")

    @app.get("/health")
    def health():
        ready = (MODEL_DIR / ONNX_FILE).exists() and (MODEL_DIR / HEAD_FILE).exists()
        return {"ready": ready, "model_dir": str(MODEL_DIR)}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        data = await file.read()
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(413 if data else 400, "Send one audio clip under 10 MB.")
        suffix = Path(file.filename or "clip.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
        try:
            wave = load_trimmed(f.name)
        except Exception:
            raise HTTPException(400, "Could not read that audio. Send WAV, FLAC, OGG or MP3.")
        finally:
            os.unlink(f.name)
        t = time.perf_counter()
        out = model().predict(wave)
        out["inference_ms"] = round((time.perf_counter() - t) * 1000, 1)
        out["note"] = "Trained on acted English speech (RAVDESS). Not for decisions about people."
        return out

    return app


app = create_app() if os.getenv("VEMO_NO_APP") != "1" else None
