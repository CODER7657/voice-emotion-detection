import numpy as np
import pytest

from vemo.metrics import across_folds, summarize


def test_macro_f1_punishes_ignoring_the_rare_class():
    y_true = np.array(["a"] * 9 + ["b"])
    y_pred = np.array(["a"] * 10)
    s = summarize(y_true, y_pred, ["a", "b"])
    assert s["accuracy"] == pytest.approx(0.9)
    assert s["macro_f1"] < 0.5
    assert s["recall"] == {"a": 1.0, "b": 0.0}
    assert s["confusion"] == [[9, 0], [1, 0]]


def test_across_folds_mean_and_std():
    assert across_folds([{"macro_f1": 0.4}, {"macro_f1": 0.6}]) == pytest.approx({"mean": 0.5, "std": 0.1414}, abs=1e-4)
