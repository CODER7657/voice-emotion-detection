import numpy as np

from vemo.splits import actor_folds, leaky_split, shared_actors


def test_actor_folds_never_share_an_actor(manifest):
    for split in actor_folds(manifest, n_folds=6, seed=0):
        assert shared_actors(manifest, split) == set()


def test_every_actor_is_tested_exactly_once(manifest):
    tested = [a for _, test in actor_folds(manifest) for a in manifest.iloc[test]["actor"].unique()]
    assert sorted(tested) == list(range(1, 25))


def test_each_test_fold_has_two_male_and_two_female_voices(manifest):
    for _, test in actor_folds(manifest):
        voices = manifest.iloc[test].drop_duplicates("actor")["gender"].value_counts()
        assert voices.to_dict() == {"male": 2, "female": 2}


def test_folds_cover_every_clip_once(manifest):
    tests = np.concatenate([test for _, test in actor_folds(manifest)])
    assert sorted(tests) == list(range(len(manifest)))


def test_folds_are_reproducible(manifest):
    a = [t.tolist() for _, t in actor_folds(manifest, seed=3)]
    b = [t.tolist() for _, t in actor_folds(manifest, seed=3)]
    assert a == b


def test_the_leaky_split_puts_actors_on_both_sides(manifest):
    assert len(shared_actors(manifest, leaky_split(manifest))) == 24
