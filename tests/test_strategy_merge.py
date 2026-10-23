"""Per-replica strategy accumulation, weighted merge, and the batch comparison."""
import numpy as np
import pytest

from persona.parameter import PersonaParam
from persona.strategy import FisherMatrix

SITES = ("a", "b", "c")
NOISE = ("n1",)


def _strategy(**overrides) -> FisherMatrix:
    options = {
        "lr_mu": 0.1,
        "lr_sigma": 0.0,
        "warmup": 0,
        "beta": 0.9,
        "lambda_reg": 0.5,
        "refresh_every": 1,
    }
    options.update(overrides)
    return FisherMatrix(param_names=list(SITES), noise_names=list(NOISE),
                        **options)


def _sites(scale: float = 1.0, sigma: float = 0.5):
    return {
        name: (0.1 * scale * (i + 1), sigma, 0.01 * scale)
        for i, name in enumerate(SITES)
    }


def _noise(scale: float = 1.0):
    return {"n1": (0.02 * scale, 1.0)}


def _obs(scale: float, advantage):
    return {"sites": _sites(scale), "noise": _noise(scale),
            "advantage": np.asarray(advantage, dtype=np.float64)}


def test_accumulate_only_updates_state_without_policy_step():
    replica = _strategy()
    full = _strategy()
    advantage = np.array([0.5, -0.2])

    assert replica.step(_sites(), _noise(), advantage,
                        accumulate_only=True) == {}
    full.step(_sites(), _noise(), advantage)

    assert np.allclose(replica.F, full.F)
    assert np.allclose(replica.Xbar, full.Xbar)
    assert np.allclose(replica.G, full.G)
    assert replica.steps == full.steps == 1


def test_weighted_merge_matches_explicit_sums_and_refreshes_inverse():
    replicas = []
    for k in range(3):
        strategy = _strategy()
        strategy.step(_sites(k + 1), _noise(k + 1),
                      np.array([0.5, -0.2]), accumulate_only=True)
        replicas.append(strategy)

    weights = np.array([0.2, 0.3, 0.5])
    merged = _strategy()
    merged.merge(replicas, weights)

    expected_F = sum(w * s.F for w, s in zip(weights, replicas))
    expected_X = sum(w * s.Xbar for w, s in zip(weights, replicas))
    expected_G = sum(w * s.G for w, s in zip(weights, replicas))

    assert np.allclose(merged.F, expected_F)
    assert np.allclose(merged.Xbar, expected_X)
    assert np.allclose(merged.G, expected_G)
    assert merged.steps == max(s.steps for s in replicas)

    ridge = merged.F + merged.lambda_reg * np.eye(merged.n_total)
    assert np.allclose(merged.P @ ridge, np.eye(merged.n_total), atol=1e-10)


def test_merge_respects_a_dominant_weight():
    replicas = []
    for k in range(3):
        strategy = _strategy()
        strategy.step(_sites(k + 1), _noise(k + 1),
                      np.array([0.1 * k, 0.0]), accumulate_only=True)
        replicas.append(strategy)

    merged = _strategy()
    merged.merge(replicas, [1.0, 0.0, 0.0])
    assert np.allclose(merged.F, replicas[0].F)
    assert np.allclose(merged.Xbar, replicas[0].Xbar)
    assert np.allclose(merged.G, replicas[0].G)


def test_step_batch_with_one_observation_equals_step():
    observation = _obs(1.0, [0.5, -0.2])
    single = _strategy()
    batch = _strategy()

    delta_single = single.step(observation["sites"], observation["noise"],
                               observation["advantage"])
    delta_batch = batch.step_batch([observation])

    assert np.allclose(single.F, batch.F)
    assert np.allclose(single.Xbar, batch.Xbar)
    assert np.allclose(single.G, batch.G)
    assert set(delta_single) == set(delta_batch)
    for name in delta_single:
        assert np.isclose(delta_single[name], delta_batch[name])


def test_step_batch_averages_the_sufficient_statistics():
    observations = [_obs(1.0, [0.5, -0.2]), _obs(2.0, [-0.1, 0.4])]
    strategy = _strategy()
    scores = [strategy._score_vectors(o["sites"], o["noise"])
              for o in observations]
    b = strategy.beta
    expected_F = (1.0 - b) * sum(
        np.outer(s, s) for s, _ in scores) / len(scores)
    expected_X = (1.0 - b) * sum(a for _, a in scores) / len(scores)

    strategy.step_batch(observations)

    assert np.allclose(strategy.F, expected_F)
    assert np.allclose(strategy.Xbar, expected_X)


def test_merge_and_combined_update_both_run(capsys):
    """The merge vs combined-update question is exercised on one stream."""
    replicates = 3
    observations = [_obs(1.0 + 0.5 * i, [0.2 * i, -0.1 * i])
                    for i in range(replicates)]
    per_replica = []
    for observation in observations:
        strategy = _strategy()
        strategy.step(observation["sites"], observation["noise"],
                      observation["advantage"], accumulate_only=True)
        per_replica.append(strategy)

    weights = np.array([0.2, 0.3, 0.5])
    merged = _strategy()
    merged.merge(per_replica, weights)
    weighted_A = sum(w * o["advantage"] for w, o in zip(weights, observations))
    delta_merge = merged.deltas(observations[0]["sites"], weighted_A)

    combined = _strategy()
    delta_batch = combined.step_batch(observations)

    norm_merge = float(np.linalg.norm(list(delta_merge.values())))
    norm_batch = float(np.linalg.norm(list(delta_batch.values())))
    with capsys.disabled():
        print(f"\n[strategy merge] weighted merge |dmu|={norm_merge:.6g} "
              f"combined |dmu|={norm_batch:.6g} "
              f"relative={abs(norm_merge - norm_batch) / max(norm_batch, 1e-12):.3f}")

    assert np.isfinite(norm_merge) and norm_merge > 0.0
    assert np.isfinite(norm_batch) and norm_batch > 0.0
    assert set(delta_merge) == set(delta_batch)


def test_sigma_batch_update_is_the_weighted_mean_step():
    param = PersonaParam(mean=0.0, sigma=1.0)
    samples = [0.4, 0.5, 0.6]
    advantages = [0.2, 0.2, 0.2]
    weights = [0.2, 0.3, 0.5]

    steps = [
        PersonaParam(mean=0.0, sigma=1.0).sigma_step(
            s, 0.0, 1.0, 0.2, 0.01)
        for s in samples
    ]
    expected = np.average(steps, weights=weights)
    param.update_sigma_batch(samples, 0.0, 1.0, advantages,
                             lr_sigma=0.01, weights=weights)

    assert np.isclose(param.sigma, np.clip(np.exp(expected), 1e-4, 2.0))


def test_merge_rejects_different_site_sets():
    left = _strategy()
    right = FisherMatrix(param_names=["a"], noise_names=list(NOISE))
    with pytest.raises(ValueError):
        left.merge([right], [1.0])
