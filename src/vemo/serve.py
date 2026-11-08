"""Serve the WavLM + logistic-regression model over HTTP, on CPU, without PyTorch.

    pip install -e ".[serve]"
    python scripts/export_onnx.py --ravdess <RAVDESS dir> --out models   # once; writes models/
    uvicorn vemo.serve:app --port 8000
    curl -F "file=@clip.wav" localhost:8000/predict

The embedding model is WavLM-base-plus exported to ONNX (``scripts/export_onnx.py``), stored
with fp16 weights that ``vemo.weights`` upcasts to fp32 at load, so compute is fp32.
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

from vemo.head import head_proba  # noqa: F401  (re-exported; the API imports vemo.head)

MODEL_DIR = Path(os.getenv("VEMO_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
ONNX_FILE = "wavlm_pooled_fp16w.onnx"
#: The fp32 export, still accepted so a local models/ from before fp16 storage works.
ONNX_FP32_FILE = "wavlm_pooled_fp32.onnx"
HEAD_FILE = "head.json"
MAX_BYTES = 10 * 1024 * 1024


def head_from_pipeline(pipeline) -> dict:
    """The numbers needed to reproduce a StandardScaler + LogisticRegression pipeline."""
    scaler, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
    return {"classes": [str(c) for c in lr.classes_], "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(), "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}


def onnx_path(model_dir: Path = MODEL_DIR) -> Path:
    """The fp16-stored model if present, else the fp32 export."""
    preferred = model_dir / ONNX_FILE
    return preferred if preferred.exists() else model_dir / ONNX_FP32_FILE


class Model:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort

        self.head = json.loads((model_dir / HEAD_FILE).read_text(encoding="utf-8"))
        from vemo.weights import load_model_bytes

        self.session = ort.InferenceSession(
            load_model_bytes(onnx_path(model_dir)), providers=["CPUExecutionProvider"]
        )
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

    app = FastAPI(title="Speech emotion (RAVDESS-trained, research use)")

    @app.get("/health")
    def health():
        ready = onnx_path().exists() and (MODEL_DIR / HEAD_FILE).exists()
        return {"ready": ready, "model_dir": str(MODEL_DIR)}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        data = await file.read()
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(413 if data else 400, "Send one audio clip under 10 MB.")
        from vemo.features import load_trimmed  # librosa, imported only when a clip arrives

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
