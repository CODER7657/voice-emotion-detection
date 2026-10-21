import pandas as pd
import pytest

from vemo import cremad


def test_parse_name():
    assert cremad.parse_name("1001_DFA_ANG_XX.wav") == {"actor": 1001, "sentence": "DFA", "emotion": "angry",
                                                        "level": "XX"}
    with pytest.raises(ValueError):
        cremad.parse_name("1001_DFA_XYZ_XX.wav")


def test_shared_emotions_exist_in_ravdess():
    from vemo.ravdess import EMOTIONS

    assert set(cremad.SHARED) <= set(EMOTIONS.values())
    assert len(cremad.SHARED) == 6


def test_manifest_with_gender(tmp_path):
    for name in ("1001_DFA_ANG_XX.wav", "1002_IEO_SAD_HI.wav", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    demo = tmp_path / "demo.csv"
    pd.DataFrame({"ActorID": [1001, 1002], "Sex": ["Male", "Female"]}).to_csv(demo, index=False)
    m = cremad.build_manifest(tmp_path, demo)
    assert list(m["emotion"]) == ["angry", "sad"]
    assert list(m["gender"]) == ["male", "female"]
