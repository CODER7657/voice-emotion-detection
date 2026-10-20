"""Run models under the leaky and the speaker-independent protocol.

Both protocols see exactly the same features, so any gap between their scores is
the leak and nothing else.

leaky   Reproduces the original pipeline: every clip plus one augmented copy is
        pooled first, then split at random, so actors (and noisy copies of test
        clips) appear on both sides. A CNN also gets the test set as its
        validation data, as the original did.
honest  Actor-grouped folds. Augmented copies are made for training actors only,
        and a CNN's validation data comes from held-out training actors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vemo import features as F
from vemo.metrics import across_folds, summarize
from vemo.models import OriginalCNN, make_linear, make_majority
from vemo.splits import actor_folds


@dataclass
class FeatureSet:
    kind: str
    clean: np.ndarray
    augmented: np.ndarray | None
    y: np.ndarray


@dataclass(frozen=True)
class Spec:
    name: str
    feature: str  # "mfcc", "mel" or "wavlm"
    factory: Callable[[], object]
    description: str


def specs(epochs: int = 50, seed: int = 0) -> dict[str, Spec]:
    return {
        "majority": Spec("majority", "mfcc", make_majority, "always predicts the most frequent class"),
        "mfcc_linear": Spec("mfcc_linear", "mfcc", make_linear, "MFCC statistics + logistic regression"),
        "cnn": Spec(
            "cnn", "mel", lambda: OriginalCNN(epochs=epochs, seed=seed), "original CNN on log-mel spectrograms"
        ),
        "wavlm_linear": Spec(
            "wavlm_linear", "wavlm", lambda: make_linear(c=0.5), "frozen WavLM embeddings + logistic regression"
        ),
    }


def load_waveforms(manifest: pd.DataFrame) -> list[np.ndarray]:
    return [F.load_clip(p) for p in manifest["path"]]


def compute_features(
    kind: str,
    waves: list[np.ndarray],
    y: np.ndarray,
    seed: int = 0,
    cache: Path | None = None,
) -> FeatureSet:
    """Features for every clip, plus one augmented copy per clip for mfcc and mel.

    WavLM is used frozen and is expensive per clip, so it is not augmented.
    """
    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=False)
        aug = data["augmented"] if data["augmented"].size else None
        return FeatureSet(kind, data["clean"], aug, data["y"])

    if kind == "wavlm":
        from vemo.ssl import embed

        clean, augmented = embed(waves), None
    else:
        fn = {"mfcc": F.mfcc_stats, "mel": F.log_mel}[kind]
        clean = np.stack([fn(w) for w in waves])
        # One generator per clip, so a clip's augmented copy does not depend on order.
        augmented = np.stack([fn(F.augment(w, np.random.default_rng([seed, i]))) for i, w in enumerate(waves)])

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, clean=clean, augmented=augmented if augmented is not None else np.empty(0), y=y
        )
    return FeatureSet(kind, clean, augmented, y)


def _fit(model, X, y, X_val=None, y_val=None):
    if isinstance(model, OriginalCNN):
        return model.fit(X, y, X_val, y_val)
    return model.fit(X, y)


def run_leaky(fs: FeatureSet, factory: Callable[[], object], seed: int = 42) -> dict:
    X, y = fs.clean, fs.y
    if fs.augmented is not None:
        X, y = np.concatenate([fs.clean, fs.augmented]), np.concatenate([fs.y, fs.y])
    positions = np.arange(len(y))
    train, test = train_test_split(positions, test_size=0.2, random_state=seed, stratify=y)
    model = _fit(factory(), X[train], y[train], X[test], y[test])
    labels = sorted(set(fs.y))
    return {"protocol": "leaky", "pooled": summarize(y[test], model.predict(X[test]), labels)}


def _validation_actors(manifest: pd.DataFrame, train: np.ndarray, rng: np.random.Generator) -> set[int]:
    """One male and one female training actor, held out to drive the LR schedule."""
    actors = manifest.iloc[train].drop_duplicates("actor")
    return {int(rng.choice(g["actor"].to_numpy())) for _, g in actors.groupby("gender")}


def run_honest(
    fs: FeatureSet,
    manifest: pd.DataFrame,
    factory: Callable[[], object],
    n_folds: int = 6,
    seed: int = 0,
) -> dict:
    labels = sorted(set(fs.y))
    rng = np.random.default_rng(seed)
    fold_scores, all_true, all_pred = [], [], []

    for k, (train, test) in enumerate(actor_folds(manifest, n_folds=n_folds, seed=seed)):
        model = factory()
        X_val = y_val = None
        fit_idx = train
        if isinstance(model, OriginalCNN):
            val_actors = _validation_actors(manifest, train, rng)
            is_val = manifest.iloc[train]["actor"].isin(val_actors).to_numpy()
            fit_idx, val_idx = train[~is_val], train[is_val]
            X_val, y_val = fs.clean[val_idx], fs.y[val_idx]

        X_fit, y_fit = fs.clean[fit_idx], fs.y[fit_idx]
        if fs.augmented is not None:
            X_fit = np.concatenate([X_fit, fs.augmented[fit_idx]])
            y_fit = np.concatenate([y_fit, fs.y[fit_idx]])

        _fit(model, X_fit, y_fit, X_val, y_val)
        pred = model.predict(fs.clean[test])
        score = summarize(fs.y[test], pred, labels)
        score["fold"] = k
        score["test_actors"] = sorted(int(a) for a in manifest.iloc[test]["actor"].unique())
        fold_scores.append(score)
        all_true.append(fs.y[test])
        all_pred.append(pred)

    return {
        "protocol": "honest",
        "folds": fold_scores,
        "macro_f1": across_folds(fold_scores, "macro_f1"),
        "accuracy": across_folds(fold_scores, "accuracy"),
        "pooled": summarize(np.concatenate(all_true), np.concatenate(all_pred), labels),
    }


def to_markdown(rows: list[dict]) -> str:
    """One table row per (task, model, protocol)."""
    lines = [
        "| Task | Model | Protocol | Macro-F1 | Accuracy |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        res = r["result"]
        if res["protocol"] == "honest":
            f1 = f"{res['macro_f1']['mean']:.3f} ± {res['macro_f1']['std']:.3f}"
            acc = f"{res['accuracy']['mean']:.3f} ± {res['accuracy']['std']:.3f}"
        else:
            f1, acc = f"{res['pooled']['macro_f1']:.3f}", f"{res['pooled']['accuracy']:.3f}"
        lines.append(f"| {r['task']} | {r['model']} | {res['protocol']} | {f1} | {acc} |")
    return "\n".join(lines) + "\n"


def save(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "results.md").write_text(to_markdown(rows), encoding="utf-8")
