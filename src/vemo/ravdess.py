"""Parse RAVDESS filenames into a manifest.

Every label used anywhere in this project comes from the filename, never from the
folder a file was sorted into. The format is documented by the dataset authors:

    MM-VV-EE-II-SS-RR-AA.wav
    modality - vocal channel - emotion - intensity - statement - repetition - actor

Odd actor numbers are male, even are female.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

EMOTIONS: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# The subset the original version of this project was trained on.
LEGACY_EMOTIONS = ("happy", "sad", "angry")

AUDIO_ONLY = "03"
SPEECH = "01"


@dataclass(frozen=True)
class Clip:
    path: str
    emotion: str
    actor: int
    gender: str
    intensity: str
    statement: int
    repetition: int


def parse_filename(name: str) -> dict[str, str | int] | None:
    """Return the fields encoded in a RAVDESS filename, or None if it isn't one."""
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 7 or not all(p.isdigit() and len(p) == 2 for p in parts):
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if emotion not in EMOTIONS:
        return None
    actor_id = int(actor)
    return {
        "modality": modality,
        "channel": channel,
        "emotion": EMOTIONS[emotion],
        "intensity": "strong" if intensity == "02" else "normal",
        "statement": int(statement),
        "repetition": int(repetition),
        "actor": actor_id,
        "gender": "male" if actor_id % 2 else "female",
    }


def build_manifest(root: str | Path, emotions: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Walk ``root`` for audio-only speech clips and return one row per clip.

    Files are de-duplicated by filename, so a dataset that was copied into emotion
    folders and also kept in actor folders is counted once.
    """
    seen: dict[str, Clip] = {}
    for path in sorted(Path(root).rglob("*.wav")):
        fields = parse_filename(path.name)
        if fields is None or fields["modality"] != AUDIO_ONLY or fields["channel"] != SPEECH:
            continue
        if emotions is not None and fields["emotion"] not in emotions:
            continue
        seen.setdefault(
            path.name,
            Clip(
                path=str(path),
                emotion=str(fields["emotion"]),
                actor=int(fields["actor"]),
                gender=str(fields["gender"]),
                intensity=str(fields["intensity"]),
                statement=int(fields["statement"]),
                repetition=int(fields["repetition"]),
            ),
        )
    columns = list(Clip.__dataclass_fields__)
    return pd.DataFrame([asdict(c) for c in seen.values()], columns=columns)


def describe(manifest: pd.DataFrame) -> str:
    """A short summary to check a download before spending GPU time on it."""
    if manifest.empty:
        return "0 clips"
    per_emotion = manifest["emotion"].value_counts().sort_index()
    return (
        f"{len(manifest)} clips, {manifest['actor'].nunique()} actors "
        f"({(manifest.drop_duplicates('actor')['gender'] == 'female').sum()} female), "
        + ", ".join(f"{e}={n}" for e, n in per_emotion.items())
    )
