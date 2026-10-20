"""Run every requested model under every requested protocol and save the results.

Meant for Colab (GPU), not a laptop:

    python scripts/run_experiments.py --data /content/ravdess --out results

Tasks:  legacy3  happy / sad / angry, the original project's setup
        all8     all eight RAVDESS emotions
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from vemo.experiment import compute_features, load_waveforms, run_honest, run_leaky, save, specs
from vemo.ravdess import LEGACY_EMOTIONS, build_manifest, describe

TASKS = {"legacy3": LEGACY_EMOTIONS, "all8": None}


def plot_confusion(score: dict, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(score["confusion"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = score["labels"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(labels), 0.8 + 0.6 * len(labels)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder containing the RAVDESS Actor_XX folders")
    p.add_argument("--out", default="results")
    p.add_argument("--tasks", default="legacy3,all8")
    p.add_argument("--models", default="majority,mfcc_linear,cnn,wavlm_linear")
    p.add_argument("--protocols", default="leaky,honest")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    all_specs = specs(epochs=args.epochs, seed=args.seed)
    rows: list[dict] = []

    for task in args.tasks.split(","):
        manifest = build_manifest(args.data, emotions=TASKS[task])
        print(f"[{task}] {describe(manifest)}", flush=True)
        y = manifest["emotion"].to_numpy()
        waves = load_waveforms(manifest)

        feature_sets = {}
        for name in args.models.split(","):
            spec = all_specs[name]
            if spec.feature not in feature_sets:
                t = time.time()
                feature_sets[spec.feature] = compute_features(
                    spec.feature, waves, y, seed=args.seed, cache=out / "cache" / f"{task}_{spec.feature}.npz"
                )
                print(f"[{task}] {spec.feature} features in {time.time() - t:.0f}s", flush=True)
            fs = feature_sets[spec.feature]

            for protocol in args.protocols.split(","):
                t = time.time()
                if protocol == "leaky":
                    result = run_leaky(fs, spec.factory)
                else:
                    result = run_honest(fs, manifest, spec.factory, n_folds=args.folds, seed=args.seed)
                    plot_confusion(result["pooled"], f"{task} / {name} / speaker-independent",
                                   out / f"confusion_{task}_{name}.png")
                rows.append({"task": task, "model": name, "description": spec.description, "result": result})
                f1 = result["macro_f1"]["mean"] if protocol == "honest" else result["pooled"]["macro_f1"]
                print(f"[{task}] {name:13s} {protocol:6s} macro-F1 {f1:.3f}  ({time.time() - t:.0f}s)", flush=True)
                save(rows, out)

    print("\n" + (out / "results.md").read_text())


if __name__ == "__main__":
    main()
