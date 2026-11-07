"""fp16-stored weights: half the bytes, fp32 compute, nothing else changed.

A small two-layer graph stands in for WavLM so the test runs anywhere in a
second; the real model's cosine and macro-F1 are measured by
``scripts/export_onnx.py`` on Colab and recorded in the export report.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from vemo import weights  # noqa: E402

RNG = np.random.default_rng(0)
W1 = RNG.normal(0, 0.2, (256, 128)).astype(np.float32)
B1 = RNG.normal(0, 0.1, 128).astype(np.float32)
W2 = RNG.normal(0, 0.2, (128, 32)).astype(np.float32)
SHAPE = np.array([1, 32], dtype=np.int64)


def _write_model(path, w1=W1):
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["h"]),
            helper.make_node("Add", ["h", "b1"], ["hb"]),
            helper.make_node("Relu", ["hb"], ["r"]),
            helper.make_node("MatMul", ["r", "w2"], ["o"]),
            helper.make_node("Reshape", ["o", "shape"], ["y"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 256])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(B1, "b1"),
            numpy_helper.from_array(W2, "w2"),
            numpy_helper.from_array(SHAPE, "shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def _run(model_bytes, x):
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": x})[0]


@pytest.fixture
def models(tmp_path):
    fp32 = _write_model(tmp_path / "fp32.onnx")
    fp16 = tmp_path / "fp16w.onnx"
    report = weights.store_fp16(fp32, fp16)
    return fp32, fp16, report


def _cosine(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_outputs_match_fp32_to_cosine_0_999(models):
    fp32, fp16, _ = models
    for seed in range(20):
        x = np.random.default_rng(seed).normal(size=(1, 256)).astype(np.float32)
        a = _run(weights.load_model_bytes(fp32), x)
        b = _run(weights.load_model_bytes(fp16), x)
        assert _cosine(a, b) >= 0.999


def test_compute_is_fp32_on_rounded_weights(models):
    """Exactly what the decision says: the only change is weight rounding."""
    _, fp16, _ = models
    x = np.random.default_rng(1).normal(size=(1, 256)).astype(np.float32)
    w1, b1, w2 = (w.astype(np.float16).astype(np.float32) for w in (W1, B1, W2))
    expected = np.maximum(x @ w1 + b1, 0) @ w2
    np.testing.assert_allclose(_run(weights.load_model_bytes(fp16), x), expected, rtol=1e-5, atol=1e-5)


def test_stored_file_is_about_half_the_size(models):
    fp32, fp16, report = models
    assert report["converted"] == 3
    assert fp16.stat().st_size < 0.6 * fp32.stat().st_size


def test_only_float32_initializers_are_converted(models):
    _, fp16, _ = models
    stored = {i.name: i.data_type for i in onnx.load(str(fp16)).graph.initializer}
    assert stored["shape"] == TensorProto.INT64
    assert {stored[n] for n in ("w1", "b1", "w2")} == {TensorProto.FLOAT16}


def test_loaded_model_is_clean_fp32(models):
    _, fp16, _ = models
    loaded = onnx.load_from_string(weights.load_model_bytes(fp16))
    assert all(i.data_type in (TensorProto.FLOAT, TensorProto.INT64) for i in loaded.graph.initializer)
    assert not any(p.key == weights.META_KEY for p in loaded.metadata_props)
    onnx.checker.check_model(loaded)


def test_plain_fp32_model_loads_unchanged(models):
    fp32, _, _ = models
    assert weights.load_model_bytes(fp32) == onnx.load(str(fp32)).SerializeToString()


def test_converting_twice_is_refused(models, tmp_path):
    _, fp16, _ = models
    with pytest.raises(ValueError, match="already"):
        weights.store_fp16(fp16, tmp_path / "again.onnx")


def test_weights_that_overflow_fp16_are_refused(tmp_path):
    big = W1.copy()
    big[0, 0] = 1e6
    src = _write_model(tmp_path / "big.onnx", w1=big)
    with pytest.raises(ValueError, match="overflows"):
        weights.store_fp16(src, tmp_path / "out.onnx")


def test_tampered_metadata_is_caught(models, tmp_path):
    _, fp16, _ = models
    model = onnx.load(str(fp16))
    for prop in model.metadata_props:
        if prop.key == weights.META_KEY:
            prop.value = '["w1", "not_in_graph"]'
    bad = tmp_path / "bad.onnx"
    onnx.save(model, str(bad))
    with pytest.raises(ValueError, match="missing"):
        weights.load_model_bytes(bad)


def test_sha256_pin(models):
    fp32, _, _ = models
    digest = weights.sha256_file(fp32)
    weights.verify_sha256(fp32, digest.upper())  # case-insensitive, whitespace tolerant
    with pytest.raises(ValueError, match="does not match"):
        weights.verify_sha256(fp32, "0" * 64)


def test_the_server_loads_the_fp16_file_and_prefers_it(models, tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("VEMO_NO_APP", "1")
    from vemo import serve

    fp32, fp16, _ = models
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    head = {"classes": [f"c{i}" for i in range(4)], "mean": [0.0] * 32, "scale": [1.0] * 32,
            "coef": RNG.normal(size=(4, 32)).tolist(), "intercept": [0.0] * 4}
    (model_dir / serve.HEAD_FILE).write_text(json.dumps(head), encoding="utf-8")

    (model_dir / serve.ONNX_FP32_FILE).write_bytes(fp32.read_bytes())
    assert serve.onnx_path(model_dir).name == serve.ONNX_FP32_FILE
    (model_dir / serve.ONNX_FILE).write_bytes(fp16.read_bytes())
    assert serve.onnx_path(model_dir).name == serve.ONNX_FILE

    wave = np.random.default_rng(3).normal(size=256).astype(np.float32)
    served = serve.Model(model_dir)
    out = served.predict(wave)
    assert set(out["probabilities"]) == set(head["classes"])
    reference = _run(weights.load_model_bytes(fp32), wave[None])
    assert _cosine(served.embed(wave), reference) >= 0.999
