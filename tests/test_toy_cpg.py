import numpy as np

from benchmarks.compare.toy_cpg import (
    BOUNDS,
    INITIAL,
    PARAM_NAMES,
    ToyCPG,
    ToyConfig,
)


def test_toy_forward_is_deterministic_and_bounded():
    toy = ToyCPG(ToyConfig(seed=0, noise_std=0.0))
    first = toy.rollout(INITIAL)
    second = toy.rollout(INITIAL)
    assert np.array_equal(first["q_model"], second["q_model"])
    assert first["loss_vec"].shape == (2,)
    assert np.isfinite(first["loss_total"])
    assert np.isfinite(first["tail_loss"])
    for name in PARAM_NAMES:
        low, high = BOUNDS[name]
        assert low <= INITIAL[name] <= high
    assert len(set(PARAM_NAMES)) == 14


def test_toy_tail_and_held_out_are_deterministic():
    toy = ToyCPG(ToyConfig(seed=0))
    first = toy.rollout(INITIAL)["tail_loss"]
    second = toy.rollout(INITIAL)["tail_loss"]
    assert first == second
    assert toy.held_out_loss(INITIAL) == toy.held_out_loss(INITIAL)


def test_toy_step_matches_rollout_shape():
    toy = ToyCPG(ToyConfig(steps=10, seed=0))
    toy.reset()
    for _ in range(10):
        row = toy.step(INITIAL)
        assert np.isfinite(row["error2"])
        assert row["phase"] in ("E", "F")
