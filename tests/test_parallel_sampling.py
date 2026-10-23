"""Accuracy and speed of the threaded PERSONA exploration sampler.

The contract: the number of sampling threads is an implementation detail.
Every site owns its own RNG stream and is drawn exactly once per step, so a
threaded run must reproduce a single-threaded run bit for bit.
"""
import time

import pytest

from persona.parameter import PersonaParam, reset_rng_stream
from persona.sampling import ExplorationSampler


def _make_params(seed: int = 0, n: int = 64) -> dict[str, PersonaParam]:
    reset_rng_stream(seed)
    return {
        f"site_{i:03d}": PersonaParam(
            mean=0.25 * i,
            sigma=0.1 + 0.01 * i,
            min=-5.0,
            max=5.0,
        )
        for i in range(n)
    }


def _sample_steps(params, threads: int, steps: int):
    sampler = ExplorationSampler(threads)
    try:
        return [sampler.sample(params) for _ in range(steps)]
    finally:
        sampler.close()


def test_parallel_sampling_matches_sequential_exactly():
    sequential = _sample_steps(_make_params(), threads=1, steps=25)
    parallel = _sample_steps(_make_params(), threads=4, steps=25)

    assert len(sequential) == len(parallel)
    for step, (seq, par) in enumerate(zip(sequential, parallel)):
        assert seq.keys() == par.keys(), f"key set changed at step {step}"
        for key in seq:
            assert seq[key] == par[key], f"{key} differs at step {step}"


@pytest.mark.parametrize("threads", [2, 3, 8, 64, 1000])
def test_every_thread_count_matches_single_thread(threads):
    sequential = _sample_steps(_make_params(), threads=1, steps=10)
    parallel = _sample_steps(_make_params(), threads=threads, steps=10)
    assert sequential == parallel


def test_parallel_sampling_respects_site_bounds():
    params = _make_params(n=64)
    draws = _sample_steps(params, threads=4, steps=50)
    for step in draws:
        for key, value in step.items():
            assert params[key].min <= value <= params[key].max


def test_sites_keep_independent_streams_under_threads():
    reset_rng_stream(0)
    params = {
        "a": PersonaParam(mean=1.0, sigma=1.0),
        "b": PersonaParam(mean=1.0, sigma=1.0),
    }
    sampler = ExplorationSampler(threads=4)
    try:
        draws = [sampler.sample(params) for _ in range(10)]
    finally:
        sampler.close()

    # A shared stream would draw identical values for identical sites.
    assert any(step["a"] != step["b"] for step in draws)


def _time_sampling(threads: int, steps: int, n: int) -> float:
    params = _make_params(n=n)
    sampler = ExplorationSampler(threads)
    try:
        sampler.sample(params)  # warm up the pool / import path
        start = time.perf_counter()
        for _ in range(steps):
            sampler.sample(params)
        return time.perf_counter() - start
    finally:
        sampler.close()


def test_parallel_sampling_speed_is_bounded(capsys):
    """Threads must not turn a cheap loop into a per-step bottleneck.

    The draws are tiny, GIL-bound Python calls, so threading is not expected
    to beat a single thread here. This test bounds the overhead; the printed
    ratio is the measurement to watch if the sampler changes.
    """
    n, steps = 400, 50
    sequential = min(_time_sampling(1, steps, n) for _ in range(3))
    parallel = min(_time_sampling(4, steps, n) for _ in range(3))

    with capsys.disabled():
        print(f"\n[sampling speed] n={n} steps={steps} "
              f"1-thread={sequential * 1e3:.1f} ms "
              f"4-thread={parallel * 1e3:.1f} ms "
              f"ratio={parallel / sequential:.2f}x")

    assert parallel <= sequential * 5.0
    assert parallel / steps < 0.05
