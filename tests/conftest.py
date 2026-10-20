import numpy as np
import pandas as pd
import pytest

from vemo.ravdess import EMOTIONS


@pytest.fixture
def manifest() -> pd.DataFrame:
    """A RAVDESS-shaped manifest with no audio behind it: 24 actors x 8 emotions x 2 clips."""
    rows = []
    for actor in range(1, 25):
        for emotion in EMOTIONS.values():
            for rep in (1, 2):
                rows.append(
                    {
                        "path": f"Actor_{actor:02d}/{emotion}-{rep}.wav",
                        "emotion": emotion,
                        "actor": actor,
                        "gender": "male" if actor % 2 else "female",
                        "intensity": "normal",
                        "statement": 1,
                        "repetition": rep,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
