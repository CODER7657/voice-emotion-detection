import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from vemo.experiment import FeatureSet, run_honest, run_leaky, to_markdown
from vemo.models import make_linear


def noise_labels(manifest, rng) -> FeatureSet:
    """Random features and random labels: there is nothing real to learn.

    Each augmented copy is its clip plus a tiny perturbation, the way an
    augmented spectrogram stays close to its source.
    """
    clean = rng.normal(size=(len(manifest), 16))
    augmented = clean + rng.normal(scale=1e-3, size=clean.shape)
    y = rng.choice(sorted(manifest["emotion"].unique()), size=len(manifest))
    return FeatureSet("test", clean, augmented, y)


def test_the_leaky_protocol_rewards_memorising(manifest, rng):
    """A 1-nearest-neighbour model finds each test clip's augmented twin in training."""
    fs = noise_labels(manifest, rng)
    memoriser = lambda: KNeighborsClassifier(n_neighbors=1)  # noqa: E731

    leaky = run_leaky(fs, memoriser)["pooled"]["accuracy"]
    honest = run_honest(fs, manifest, memoriser)["accuracy"]["mean"]

    assert leaky > 0.4  # far above chance on labels that are pure noise
    assert honest < 0.25  # chance for 8 classes is 0.125


def test_honest_result_has_one_entry_per_fold(manifest, rng):
    fs = noise_labels(manifest, rng)
    result = run_honest(fs, manifest, make_linear, n_folds=6)
    assert len(result["folds"]) == 6
    assert result["pooled"]["n"] == len(manifest)
    tested = sorted(a for f in result["folds"] for a in f["test_actors"])
    assert tested == list(range(1, 25))


def test_markdown_table_has_a_row_per_result(manifest, rng):
    fs = noise_labels(manifest, rng)
    rows = [
        {"task": "t", "model": "m", "result": run_leaky(fs, make_linear)},
        {"task": "t", "model": "m", "result": run_honest(fs, manifest, make_linear)},
    ]
    table = to_markdown(rows)
    assert table.count("\n") == 4
    assert "± " in table
