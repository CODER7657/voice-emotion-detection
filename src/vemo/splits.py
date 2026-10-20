"""Train/test splits.

``actor_folds`` is the protocol every reported number uses: an actor's clips are
either all in training or all in test, so the model is always scored on voices
it has never heard.

``leaky_split`` exists only to reproduce the original evaluation, a random
stratified split that puts every actor on both sides, so the two can be
compared on the same data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Split = tuple[np.ndarray, np.ndarray]


def actor_folds(manifest: pd.DataFrame, n_folds: int = 6, seed: int = 0) -> list[Split]:
    """Group k-fold over actors, balancing gender within each test fold.

    Male and female actors are shuffled separately and dealt round-robin into
    folds, so with RAVDESS's 12 + 12 actors and 6 folds each test fold holds
    2 male and 2 female voices. Returns positional (train, test) index arrays.
    """
    actors = manifest.drop_duplicates("actor")[["actor", "gender"]]
    if actors["actor"].nunique() < n_folds:
        raise ValueError(f"need at least {n_folds} actors, got {actors['actor'].nunique()}")

    rng = np.random.default_rng(seed)
    fold_of: dict[int, int] = {}
    slot = 0
    for _, group in actors.groupby("gender", sort=True):
        # permutation shuffles a copy: with copy-on-write pandas, to_numpy() can return
        # a read-only view. Same order as shuffling in place with the same generator.
        for actor in rng.permutation(group["actor"].to_numpy()):
            fold_of[int(actor)] = slot % n_folds
            slot += 1

    folds = manifest["actor"].map(fold_of).to_numpy()
    positions = np.arange(len(manifest))
    return [(positions[folds != k], positions[folds == k]) for k in range(n_folds)]


def leaky_split(manifest: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Split:
    """The original protocol: a random split stratified by emotion, ignoring actors."""
    positions = np.arange(len(manifest))
    train, test = train_test_split(
        positions, test_size=test_size, random_state=seed, stratify=manifest["emotion"]
    )
    return np.sort(train), np.sort(test)


def shared_actors(manifest: pd.DataFrame, split: Split) -> set[int]:
    """Actors that appear on both sides of a split. Empty means speaker-independent."""
    train, test = split
    return set(manifest.iloc[train]["actor"]) & set(manifest.iloc[test]["actor"])
