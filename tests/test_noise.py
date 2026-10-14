import numpy as np

from persona.noise_recorder import NoiseRecorder


def test_weight_delta_tagged():
    nr = NoiseRecorder("n")
    nr.observe_weight("cut_rge", 0.0, 10.0)
    nr.observe_weight("cut_rge", 1.0, 12.5)
    d = nr.get_latest_step_deltas()
    assert np.isclose(d["noise_cut_rge_weight"], 2.5)


def test_rate_and_weight_coexist():
    nr = NoiseRecorder("n")
    nr.observe("cut", 0.0, "rate", 5.0)
    nr.observe("cut", 1.0, "rate", 7.0)
    nr.observe_weight("cut_rge", 0.0, 1.0)
    nr.observe_weight("cut_rge", 1.0, 3.0)
    d = nr.get_latest_step_deltas()
    assert np.isclose(d["noise_cut_rate"], 2.0)
    assert np.isclose(d["noise_cut_rge_weight"], 2.0)


def test_first_weight_observation_is_zero_delta():
    nr = NoiseRecorder("n")
    s = nr.observe_weight("ia_rg", 0.0, 4.2)
    assert np.isclose(s.value, 0.0)