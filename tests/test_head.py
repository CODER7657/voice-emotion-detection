"""vemo.head: the NumPy-only head and its validation."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vemo.head import head_proba, validate_head

GOOD = {
    "classes": ["a", "b", "c"],
    "mean": [0.0, 1.0],
    "scale": [1.0, 2.0],
    "coef": [[1.0, 0.0], [0.0, 1.0], [1, -1]],
    "intercept": [0.0, 0.1, -0.1],
}


def bad(**changes):
    head = copy.deepcopy(GOOD)
    head.update(changes)
    return head


def test_good_head_is_valid_and_gives_finite_probabilities():
    assert validate_head(GOOD) == 2
    p = head_proba(GOOD, np.array([[0.5, -3.0]]))
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1)


@pytest.mark.parametrize("head,match", [
    (bad(scale=[0.0, 2.0]), "> 0"),
    (bad(scale=[-1.0, 2.0]), "> 0"),
    (bad(mean=["x", 1.0]), "numbers"),
    (bad(mean=["1.0", 1.0]), "numbers"),
    (bad(mean=[True, 1.0]), "numbers"),
    (bad(mean=None), "numbers"),
    (bad(coef=[[float("inf"), 0.0], [0.0, 1.0], [1, -1]]), "NaN or infinity"),
    (bad(intercept=[float("nan"), 0.1, -0.1]), "NaN or infinity"),
    (bad(coef=[[1.0, 0.0], [0.0]]), "numbers|inconsistent"),
    (bad(coef=[[1.0, 0.0], [0.0, 1.0]]), "inconsistent"),
    (bad(scale=[1.0]), "inconsistent"),
    (bad(mean=[]), "inconsistent"),
    (bad(classes=["a", "a", "c"]), "distinct"),
    (bad(classes=["a"]), "distinct"),
    (bad(classes="abc"), "distinct"),
    ({k: v for k, v in GOOD.items() if k != "coef"}, "missing"),
    ([1, 2], "object"),
])
def test_bad_heads_are_refused(head, match):
    with pytest.raises(ValueError, match=match):
        validate_head(head)
