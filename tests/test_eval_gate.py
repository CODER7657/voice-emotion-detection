"""The evaluation gate: CI re-runs the speaker-independent evaluation on every change.

The committed embeddings (``models/embeddings_onnx_fp32.npz``, WavLM via the served ONNX
export, one row per RAVDESS clip) let CI repeat the 6-fold actor-grouped evaluation
without audio, a GPU or the 378 MB model. Two things must keep holding:

* **macro-F1 >= 0.72** over the same folds as the published 0.749. A change to the
  splits, the head's hyper-parameters or the scoring that costs more than that fails.
* **the shipped head is the head this code fits**: refitting ``make_linear(c=0.5)`` on
  all clips must agree with ``models/head.json`` on at least 99% of predictions, so the
  served classifier cannot drift away from what the README measures.

Derived from RAVDESS (CC BY-NC-SA 4.0); see NOTICE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vemo.experiment import FeatureSet, run_honest
from vemo.models import make_linear
from vemo.ravdess import parse_filename
from vemo.serve import head_proba

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "models" / "embeddings_onnx_fp32.npz"
HEAD = ROOT / "models" / "head.json"
MIN_MACRO_F1 = 0.72
MIN_HEAD_AGREEMENT = 0.99


@pytest.fixture(scope="module")
def data():
    # allow_pickle stays False (the default): every array is numeric or a fixed-width string.
    d = np.load(EMBEDDINGS)
    X = d["X"].astype(np.float32)
    rows = []
    for name, actor in zip(d["files"], d["actor"]):
        meta = parse_filename(str(name))
        assert meta is not None, name
        assert meta["actor"] == int(actor), name
        rows.append({"path": str(name), "emotion": meta["emotion"], "actor": meta["actor"],
                     "gender": meta["gender"]})
    manifest = pd.DataFrame(rows)
    assert (manifest["emotion"].to_numpy() == d["emotion"]).all()
    return X, manifest


def test_the_committed_set_is_the_full_corpus(data):
    X, manifest = data
    assert X.shape == (1440, 768)
    assert manifest["actor"].nunique() == 24
    assert manifest["emotion"].nunique() == 8
    assert manifest["path"].is_unique


def test_speaker_independent_macro_f1_stays_above_the_gate(data):
    X, manifest = data
    y = manifest["emotion"].to_numpy()
    result = run_honest(FeatureSet("wavlm", X, None, y), manifest, lambda: make_linear(c=0.5))
    macro_f1 = result["macro_f1"]["mean"]
    # Every fold holds 4 unseen actors (2 male, 2 female); none of them is in its training set.
    for fold in result["folds"]:
        assert len(fold["test_actors"]) == 4
    assert macro_f1 >= MIN_MACRO_F1, f"macro-F1 {macro_f1:.4f} fell below the {MIN_MACRO_F1} gate"


def test_the_shipped_head_is_the_head_this_code_fits(data):
    X, manifest = data
    y = manifest["emotion"].to_numpy()
    head = json.loads(HEAD.read_text(encoding="utf-8"))
    pipe = make_linear(c=0.5).fit(X, y)
    assert list(pipe.classes_) == head["classes"]
    shipped = np.asarray(head["classes"])[head_proba(head, X).argmax(axis=1)]
    agreement = float((pipe.predict(X) == shipped).mean())
    assert agreement >= MIN_HEAD_AGREEMENT, f"refit agrees with head.json on {agreement:.4f} of clips"
