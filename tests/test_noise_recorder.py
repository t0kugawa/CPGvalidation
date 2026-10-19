import numpy as np

from persona.noise_recorder import NoiseRecorder


def test_take_delta_is_fresh():
    nr = NoiseRecorder("producer:stdp:x")
    assert nr.take_delta() is None
    nr.record_delta(0.0, 0.5)
    assert nr.take_delta() == 0.5
    assert nr.take_delta() is None                    # cleared


def test_latest_sample_persists():
    nr = NoiseRecorder("site:pop_p_a")
    assert nr.latest_sample() is None
    nr.record_sample(3, 1.5, 0.2)
    assert nr.latest_sample() == (1.5, 0.2)
    assert nr.latest_sample() == (1.5, 0.2)           # never cleared


def test_sample_recordings_are_not_deltas():
    nr = NoiseRecorder("site:pop_p_a")
    nr.record_sample(1, 2.0, 0.5)
    assert nr.take_delta() is None


def test_stats_converge():
    nr = NoiseRecorder("producer:poisson:cut_L", beta=0.98)
    rng = np.random.default_rng(0)
    vals = rng.normal(2.0, 0.5, 4000)
    for i, v in enumerate(vals):
        nr.record_delta(float(i), float(v))
    s = nr.stats()
    assert abs(s["mean"] - 2.0) < 0.1 and 0.1 < s["std"] < 1.0


def test_stats_are_constant_cost():
    nr = NoiseRecorder("producer:x")
    for i in range(10_000):
        nr.record_delta(float(i), 1.0)
    s = nr.stats()
    assert s["n"] == 10_000 and np.isclose(s["std"], 0.0) and np.isclose(s["mean"], 1.0)
