"""CREMA-D, used only as an outside test set.

CREMA-D (Cao et al., 2014) has 7,442 clips from 91 actors of varied age and
ethnicity, each saying one of 12 sentences in one of six emotions. Its filenames
look like ``1001_DFA_ANG_XX.wav``: actor, sentence, emotion, intensity. Labels are
the emotion the actor was asked to portray, read from the filename, as for RAVDESS.

The six CREMA-D emotions all exist in RAVDESS, so a model trained on one can be
scored on the other. Calm and surprised (RAVDESS only) are left out of that test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EMOTIONS = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
SHARED = tuple(sorted(EMOTIONS.values()))

DEMOGRAPHICS_URL = (
    "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoDemographics.csv"
)


def parse_name(name: str) -> dict:
    stem = Path(name).stem
    actor, sentence, emotion, level = stem.split("_")
    if emotion not in EMOTIONS:
        raise ValueError(f"unknown emotion code in {name!r}")
    return {"actor": int(actor), "sentence": sentence, "emotion": EMOTIONS[emotion], "level": level}


def build_manifest(root: str | Path, demographics: str | Path | None = None) -> pd.DataFrame:
    """One row per clip with path, emotion, actor and (if demographics given) gender."""
    rows = []
    for p in sorted(Path(root).glob("*.wav")):
        try:
            rows.append({"path": str(p), **parse_name(p.name)})
        except ValueError:
            continue
    m = pd.DataFrame(rows)
    if demographics is not None and len(m):
        demo = pd.read_csv(demographics)
        sex = dict(zip(demo["ActorID"].astype(int), demo["Sex"].str.lower()))
        m["gender"] = m["actor"].map(sex).fillna("unknown")
    return m
