"""Per-replica sampling: different samples, reproducible streams, low overhead."""
import time

from persona.parameter import PersonaParam, reset_rng_stream
from persona.sampling import ReplicaSampler


def _params(seed: int, n: int = 64):
    reset_rng_stream(seed)
    return {
        f"site_{i:03d}": PersonaParam(
            mean=0.25 * i,
            sigma=0.1 + 0.01 * i,
            min=-100.0,
            max=100.0,
        )
        for i in range(n)
    }


def test_replicas_draw_different_samples():
    params = _params(seed=0, n=32)
    result = ReplicaSampler(replicas=3).sample(params)
    actions = [actions for actions, _ in result]

    for site in params:
        values = {actions[replica][site] for replica in range(3)}
        assert len(values) > 1, f"{site} was identical in every replica"


def test_replica_samples_are_reproducible_from_the_base_seed():
    first = ReplicaSampler(replicas=4).sample(_params(seed=7))
    second = ReplicaSampler(replicas=4).sample(_params(seed=7))
    assert first == second


def test_single_replica_matches_direct_sampling():
    params = _params(seed=3)
    same_stream = _params(seed=3)
    result = ReplicaSampler(replicas=1).sample(params)
    assert len(result) == 1
    actions, snapshot = result[0]
    for name, param in same_stream.items():
        expected = param.sample_exploration()
        assert actions[name] == expected
        assert snapshot[name] == (param.mean, param.sigma, expected)


def test_samples_respect_site_bounds():
    params = _params(seed=1, n=64)
    for actions, snapshot in ReplicaSampler(replicas=4).sample(params):
        for site, value in actions.items():
            param = params[site]
            assert param.min <= value <= param.max
            assert snapshot[site][2] == value


def _timed_vectorised(replicas: int, n: int, steps: int) -> float:
    params = _params(seed=0, n=n)
    sampler = ReplicaSampler(replicas=replicas)
    start = time.perf_counter()
    for _ in range(steps):
        sampler.sample(params)
    return time.perf_counter() - start


def _timed_scalar(replicas: int, n: int, steps: int) -> float:
    params = _params(seed=0, n=n)
    start = time.perf_counter()
    for _ in range(steps):
        for _ in range(replicas):
            {name: param.sample_exploration()
             for name, param in params.items()}
    return time.perf_counter() - start


def test_vectorised_replica_sampling_speed_is_bounded(capsys):
    replicas, n, steps = 4, 400, 50
    scalar = min(_timed_scalar(replicas, n, steps) for _ in range(3))
    vectorised = min(_timed_vectorised(replicas, n, steps) for _ in range(3))

    with capsys.disabled():
        print(f"\n[replica sampling] {replicas} replicas x {n} sites x {steps} steps "
              f"scalar={scalar * 1e3:.1f} ms vectorised={vectorised * 1e3:.1f} ms "
              f"ratio={vectorised / scalar:.2f}x "
              f"({scalar / vectorised:.2f}x speedup)")

    assert vectorised <= scalar * 2.0
    assert vectorised / steps < 0.05


def test_thread_count_larger_than_replica_count_is_fine():
    params = _params(seed=2)
    assert len(ReplicaSampler(replicas=64).sample(params)) == 64
