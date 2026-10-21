"""How well does a RAVDESS model work on voices from another dataset?

Trains on one corpus and tests on the other, for the six emotions both share, and
compares that with speaker-independent cross-validation inside each corpus. Meant
for Colab (GPU), where WavLM embeddings take minutes instead of hours:

    python scripts/cross_dataset.py --ravdess /content/ravdess --cremad /content/cremad \
        --demographics /content/VideoDemographics.csv --out results/cross_dataset

Both corpora are loaded with ``load_trimmed`` (energy-based silence trimming, then
3 s), so neither gets a loader tuned to it. Writes ``cross_dataset.json`` and the
cached embeddings (``*.npz``, not committed).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vemo import cremad, ravdess
from vemo.experiment import FeatureSet, run_honest
from vemo.features import load_trimmed, mfcc_stats
from vemo.metrics import summarize
from vemo.models import make_linear, make_majority


def features(kind: str, paths: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)["X"]
    t = time.time()
    waves = [load_trimmed(p) for p in paths]
    if kind == "wavlm":
        from vemo.ssl import embed

        X = embed(waves)
    else:
        X = np.stack([mfcc_stats(w) for w in waves])
    np.savez_compressed(cache, X=X)
    print(f"{kind} features for {len(paths)} clips in {time.time() - t:.0f}s", flush=True)
    return X


def transfer(factory, X_train, y_train, X_test, y_test, labels) -> dict:
    model = factory().fit(X_train, y_train)
    return summarize(y_test, model.predict(X_test), labels)


def compact(score: dict) -> dict:
    return {k: score[k] for k in ("n", "accuracy", "macro_f1", "recall", "labels", "confusion")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ravdess", required=True)
    ap.add_argument("--cremad", required=True)
    ap.add_argument("--demographics", required=True)
    ap.add_argument("--out", default="results/cross_dataset")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shared = list(cremad.SHARED)
    rav = ravdess.build_manifest(args.ravdess, emotions=cremad.SHARED).reset_index(drop=True)
    cre = cremad.build_manifest(args.cremad, args.demographics).reset_index(drop=True)
    print(f"RAVDESS: {len(rav)} clips, {rav['actor'].nunique()} actors | "
          f"CREMA-D: {len(cre)} clips, {cre['actor'].nunique()} actors", flush=True)

    results = {"clips": {"ravdess": int(len(rav)), "cremad": int(len(cre))},
               "actors": {"ravdess": int(rav["actor"].nunique()), "cremad": int(cre["actor"].nunique())},
               "emotions": shared, "loader": "load_trimmed (top_db=30, 3 s)", "models": {}}
    y_rav, y_cre = rav["emotion"].to_numpy(), cre["emotion"].to_numpy()
    for kind, factory in (("wavlm", lambda: make_linear(c=0.5)), ("mfcc", make_linear)):
        X_rav = features(kind, rav["path"].tolist(), out / f"ravdess6_{kind}.npz")
        X_cre = features(kind, cre["path"].tolist(), out / f"cremad_{kind}.npz")
        res = {
            "ravdess_to_cremad": compact(transfer(factory, X_rav, y_rav, X_cre, y_cre, shared)),
            "cremad_to_ravdess": compact(transfer(factory, X_cre, y_cre, X_rav, y_rav, shared)),
        }
        for name, X, y, m, folds in (("ravdess_cv", X_rav, y_rav, rav, 6), ("cremad_cv", X_cre, y_cre, cre, 5)):
            r = run_honest(FeatureSet(kind, X, None, y), m, factory, n_folds=folds)
            res[name] = {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "pooled": compact(r["pooled"])}
        results["models"][f"{kind}_linear"] = res
        f1 = {k: v["macro_f1"]["mean"] if isinstance(v["macro_f1"], dict) else v["macro_f1"] for k, v in res.items()}
        print(kind, {k: round(v, 3) for k, v in f1.items()}, flush=True)
    chance = summarize(y_cre, make_majority().fit(X_rav, y_rav).predict(X_cre), shared)
    results["majority_ravdess_to_cremad"] = compact(chance)
    (out / "cross_dataset.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("<<<CROSS_RESULTS>>>")
    print(json.dumps(results, separators=(",", ":")))
    print("<<<END_CROSS_RESULTS>>>")


if __name__ == "__main__":
    main()
