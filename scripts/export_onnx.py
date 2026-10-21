"""Export the served model and measure what quantisation costs. Run on Colab:

    python scripts/export_onnx.py --ravdess /content/ravdess --out models

1. WavLM-base-plus, with the same pooling as ``vemo.ssl.embed`` (mean over time,
   then over layers), exported to ONNX for a fixed 3 s input.
2. A dynamic int8 copy, measured for comparison (on 2 CPU threads it was slower and less
   accurate, so the fp32 model is the one served).
3. The logistic-regression head, fitted on all 1,440 RAVDESS clips (8 emotions,
   ``load_trimmed``), saved to ``head.json``.
4. Checks: embedding agreement with PyTorch, speaker-independent macro-F1 using the
   ONNX embeddings against the PyTorch ones (same 6 folds), and CPU latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from vemo.experiment import FeatureSet, run_honest
from vemo.features import DURATION_S, SAMPLE_RATE, load_trimmed
from vemo.models import make_linear
from vemo.ravdess import build_manifest
from vemo.serve import HEAD_FILE, ONNX_FILE, head_from_pipeline, head_proba
from vemo.ssl import DEFAULT_MODEL

N = int(SAMPLE_RATE * DURATION_S)


def pooled_module(model_name: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    class Pooled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, wave):
            if extractor.do_normalize:  # what the feature extractor does before the model
                wave = (wave - wave.mean(-1, keepdim=True)) / torch.sqrt(wave.var(-1, keepdim=True) + 1e-7)
            hidden = self.model(wave, output_hidden_states=True).hidden_states
            return torch.stack(hidden).mean(dim=2).mean(dim=0)

    return Pooled().eval()


def latency(fn, x, runs=30) -> dict:
    for _ in range(3):
        fn(x)
    ms = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(x)
        ms.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(ms, 50)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1)}


def main() -> None:
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess", required=True)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    module = pooled_module(DEFAULT_MODEL)
    fp32 = out / "wavlm_pooled_fp32.onnx"
    int8 = out / "wavlm_pooled_int8.onnx"
    assert fp32.name == ONNX_FILE
    dummy = torch.zeros(1, N)
    kwargs = dict(input_names=["wave"], output_names=["embedding"], opset_version=17)
    try:
        torch.onnx.export(module, dummy, str(fp32), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, dummy, str(fp32), **kwargs)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    manifest = build_manifest(args.ravdess).reset_index(drop=True)
    waves = np.stack([load_trimmed(p) for p in manifest["path"]])
    y = manifest["emotion"].to_numpy()

    sessions = {k: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for k, p in
                (("onnx_fp32", fp32), ("onnx_int8", int8))}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module_dev = module.to(device)
    with torch.inference_mode():
        X_torch = np.concatenate([module_dev(torch.from_numpy(waves[i:i + 16]).to(device)).float().cpu().numpy()
                                  for i in range(0, len(waves), 16)])
    X = {"torch": X_torch}
    for k, s in sessions.items():
        t = time.time()
        X[k] = np.concatenate([s.run(None, {"wave": w[None]})[0] for w in waves])
        print(f"{k} embeddings: {time.time() - t:.0f}s", flush=True)

    def cosine(a, b):
        return float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))

    report = {"model": DEFAULT_MODEL, "clips": int(len(y)), "loader": "load_trimmed (top_db=30, 3 s)",
              "sizes_mb": {p.name: round(p.stat().st_size / 1e6, 1) for p in (fp32, int8)},
              "cosine_vs_torch": {k: round(cosine(X[k], X_torch), 5) for k in sessions},
              "honest_cv": {}, "cpu_latency_3s_clip": {}, "cpu_threads": os.cpu_count()}
    for k, Xk in X.items():
        r = run_honest(FeatureSet("wavlm", Xk, None, y), manifest, lambda: make_linear(c=0.5))
        report["honest_cv"][k] = {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"]}
        print(k, "speaker-independent macro-F1", round(r["macro_f1"]["mean"], 3), flush=True)

    # The served head: fitted on all clips, using the fp32 ONNX embeddings it will see in production.
    pipe = make_linear(c=0.5).fit(X["onnx_fp32"], y)
    head = head_from_pipeline(pipe)
    assert np.allclose(head_proba(head, X["onnx_fp32"][:50]), pipe.predict_proba(X["onnx_fp32"][:50]), atol=1e-6)
    (out / HEAD_FILE).write_text(json.dumps(head), encoding="utf-8")

    module_cpu = module.to("cpu")
    x = waves[:1]
    with torch.inference_mode():
        report["cpu_latency_3s_clip"]["torch_fp32"] = latency(lambda a: module_cpu(torch.from_numpy(a)), x)
    for k, s in sessions.items():
        report["cpu_latency_3s_clip"][k] = latency(lambda a: s.run(None, {"wave": a}), x)
    report["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fp32, out / HEAD_FILE)}
    (out / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("<<<EXPORT_REPORT>>>")
    print(json.dumps(report, separators=(",", ":")))
    print("<<<END_EXPORT_REPORT>>>")


if __name__ == "__main__":
    main()
