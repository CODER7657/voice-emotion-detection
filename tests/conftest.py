import numpy as np
import pandas as pd
import pytest

from vemo.ravdess import EMOTIONS


@pytest.fixture
def manifest() -> pd.DataFrame:
    """A RAVDESS-shaped manifest with no audio behind it: 24 actors x 8 emotions x 2 clips."""
    rows = []
    for actor in range(1, 25):
        for emotion in EMOTIONS.values():
            for rep in (1, 2):
                rows.append(
                    {
                        "path": f"Actor_{actor:02d}/{emotion}-{rep}.wav",
                        "emotion": emotion,
                        "actor": actor,
                        "gender": "male" if actor % 2 else "female",
                        "intensity": "normal",
                        "statement": 1,
                        "repetition": rep,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --- Live API (api/index.py) ---------------------------------------------------
#
# A tiny stand-in for WavLM keeps the API tests fast and lets CI run them without
# the 190 MB release asset: one MatMul from the 48,000-sample waveform to a
# 16-dim "embedding", stored with fp16 weights so the real loader path (upcast in
# vemo.weights) is exercised, plus an 8-class head over those 16 features.

API_CLASSES = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
API_DIM = 16


def _load_api_module():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "api" / "index.py"
    spec = importlib.util.spec_from_file_location("vemo_api_index", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def api_module():
    pytest.importorskip("fastapi")
    pytest.importorskip("onnxruntime")
    return _load_api_module()


def write_standin_model(directory, seed: int = 0):
    """Write wavlm_pooled_fp16w.onnx + head.json for the API into ``directory``."""
    import json

    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from vemo.weights import store_fp16

    g = np.random.default_rng(seed)
    w = g.normal(0, 0.05, (48_000, API_DIM)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input_values", "w"], ["pooled"])],
        "standin",
        [helper.make_tensor_value_info("input_values", TensorProto.FLOAT, ["batch", 48_000])],
        [helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["batch", API_DIM])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    fp32 = directory / "standin_fp32.onnx"
    onnx.save(model, str(fp32))
    store_fp16(fp32, directory / "wavlm_pooled_fp16w.onnx")
    fp32.unlink()
    head = {
        "classes": API_CLASSES,
        "mean": g.normal(0, 0.1, API_DIM).tolist(),
        "scale": (1 + g.random(API_DIM)).tolist(),
        "coef": g.normal(0, 1, (len(API_CLASSES), API_DIM)).tolist(),
        "intercept": g.normal(0, 0.1, len(API_CLASSES)).tolist(),
    }
    (directory / "head.json").write_text(json.dumps(head), encoding="utf-8")
    return directory


@pytest.fixture(scope="session")
def standin_model_dir(tmp_path_factory):
    return write_standin_model(tmp_path_factory.mktemp("api_model"))


@pytest.fixture
def api_env(monkeypatch, api_module, standin_model_dir):
    """Clean environment pointing the API at the stand-in model."""
    for var in ("MODEL_SHA256", "VERCEL", "ALLOWED_ORIGINS", "RATE_LIMIT_PER_MIN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VEMO_MODEL_DIR", str(standin_model_dir))
    api_module.reset_model_cache()
    yield api_module
    api_module.reset_model_cache()
