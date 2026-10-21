import io
import os

import numpy as np
import pytest
import soundfile as sf
from sklearn.datasets import make_classification

os.environ["VEMO_NO_APP"] = "1"
from vemo import serve  # noqa: E402
from vemo.models import make_linear  # noqa: E402


def test_exported_head_matches_sklearn():
    X, y = make_classification(n_samples=300, n_features=20, n_informative=8, n_classes=4, random_state=0)
    labels = np.array(["angry", "calm", "happy", "sad"])[y]
    pipe = make_linear(c=0.5).fit(X, labels)
    head = serve.head_from_pipeline(pipe)
    assert head["classes"] == ["angry", "calm", "happy", "sad"]
    assert np.allclose(serve.head_proba(head, X), pipe.predict_proba(X), atol=1e-8)


class FakeModel:
    def predict(self, wave):
        assert wave.shape == (48_000,)
        return {"emotion": "happy", "probabilities": {"happy": 0.7, "sad": 0.3}}


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(serve, "model", lambda: FakeModel())
    return TestClient(serve.create_app())


def wav_bytes(seconds=1.5, sr=16_000):
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_predict_returns_the_model_output(client):
    r = client.post("/predict", files={"file": ("clip.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["emotion"] == "happy" and "inference_ms" in body and "Not for decisions" in body["note"]


def test_bad_audio_is_a_400(client):
    r = client.post("/predict", files={"file": ("clip.wav", b"not audio", "audio/wav")})
    assert r.status_code == 400


def test_empty_upload_is_rejected(client):
    assert client.post("/predict", files={"file": ("clip.wav", b"", "audio/wav")}).status_code == 400
