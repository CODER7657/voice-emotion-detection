"""Rebuild the result table and charts from the compact JSON the Colab notebook prints.

    python scripts/plot_results.py results/compact.json --out results

Light enough for a laptop: it only reads numbers, it never touches audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAMES = {
    "majority": "Majority class",
    "mfcc_linear": "MFCC + logistic reg.",
    "cnn": "Original CNN",
    "wavlm_linear": "WavLM (frozen) + logistic reg.",
}


def f1_of(row: dict) -> tuple[float, float]:
    if row["protocol"] == "honest":
        return row["macro_f1"]["mean"], row["macro_f1"]["std"]
    return row["pooled"]["macro_f1"], 0.0


def leak_chart(rows: list[dict], task: str, path: Path) -> None:
    models = [m for m in NAMES if any(r["model"] == m and r["task"] == task for r in rows)]
    by = {(r["model"], r["protocol"]): f1_of(r) for r in rows if r["task"] == task}
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for offset, protocol, color, label in [
        (-0.2, "leaky", "#c9c9c9", "Original split (speakers leak)"),
        (0.2, "honest", "#1d3557", "Speaker-independent (6 folds)"),
    ]:
        vals = [by.get((m, protocol), (np.nan, 0)) for m in models]
        # Only the fold protocol has a spread; a single split gets no error bar.
        err = [v[1] for v in vals] if protocol == "honest" else None
        ax.bar(x + offset, [v[0] for v in vals], 0.38, yerr=err, color=color, label=label, capsize=3)
        for xi, (v, sd) in zip(x + offset, vals):
            if not np.isnan(v):
                ax.text(xi, v + sd + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [NAMES[m] for m in models], fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"{task}: the same models, scored two ways", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_chart(row: dict, path: Path) -> None:
    labels = row["pooled"]["labels"]
    cm = np.array(row["pooled"]["confusion"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
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
    ax.set_title(f"{row['task']} / {NAMES[row['model']]} / speaker-independent", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(rows: list[dict]) -> str:
    lines = ["| Task | Model | Original split | Speaker-independent |", "|---|---|---|---|"]
    for task in sorted({r["task"] for r in rows}, reverse=True):
        for m in NAMES:
            leaky = next((r for r in rows if r["task"] == task and r["model"] == m and r["protocol"] == "leaky"), None)
            honest = next((r for r in rows if r["task"] == task and r["model"] == m and r["protocol"] == "honest"), None)
            if not (leaky or honest):
                continue
            a = f"{f1_of(leaky)[0]:.3f}" if leaky else "n/a"
            b = "{:.3f} ± {:.3f}".format(*f1_of(honest)) if honest else "n/a"
            lines.append(f"| {task} | {NAMES[m]} | {a} | {b} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("compact")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    rows = json.loads(Path(args.compact).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in {r["task"] for r in rows}:
        leak_chart(rows, task, out / f"leak_{task}.png")
    for r in rows:
        if r["protocol"] == "honest" and r["model"] != "majority":
            confusion_chart(r, out / f"confusion_{r['task']}_{r['model']}.png")
    (out / "summary.md").write_text(table(rows), encoding="utf-8")
    print(table(rows))


if __name__ == "__main__":
    main()
