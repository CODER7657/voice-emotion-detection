from vemo.ravdess import build_manifest, describe, parse_filename


def test_parse_filename_reads_every_field():
    f = parse_filename("03-01-05-02-01-02-12.wav")
    assert f == {
        "modality": "03",
        "channel": "01",
        "emotion": "angry",
        "intensity": "strong",
        "statement": 1,
        "repetition": 2,
        "actor": 12,
        "gender": "female",
    }


def test_odd_actors_are_male():
    assert parse_filename("03-01-01-01-01-01-07.wav")["gender"] == "male"


def test_rejects_names_that_are_not_ravdess():
    for name in ["temp_recording.wav", "03-01-05-01-01-01.wav", "03-01-09-01-01-01-01.wav", "03-01-xx-01-01-01-01.wav"]:
        assert parse_filename(name) is None


def test_manifest_keeps_audio_only_speech_and_dedupes(tmp_path):
    actor = tmp_path / "Actor_01"
    sorted_copy = tmp_path / "angry"
    actor.mkdir()
    sorted_copy.mkdir()
    for name in [
        "03-01-05-01-01-01-01.wav",  # audio-only speech, angry
        "03-01-03-01-01-01-01.wav",  # audio-only speech, happy
        "01-01-05-01-01-01-01.wav",  # video: skipped
        "03-02-05-01-01-01-01.wav",  # song: skipped
    ]:
        (actor / name).touch()
    (sorted_copy / "03-01-05-01-01-01-01.wav").touch()  # same clip, sorted into a folder

    manifest = build_manifest(tmp_path)
    assert sorted(manifest["emotion"]) == ["angry", "happy"]
    assert describe(manifest).startswith("2 clips, 1 actors")


def test_manifest_emotion_filter(tmp_path):
    for name in ["03-01-05-01-01-01-01.wav", "03-01-01-01-01-01-01.wav"]:
        (tmp_path / name).touch()
    assert list(build_manifest(tmp_path, emotions=("angry",))["emotion"]) == ["angry"]
