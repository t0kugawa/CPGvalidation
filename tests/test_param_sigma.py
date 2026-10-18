import numpy as np

from persona.parameter import PersonaParam, reset_rng_stream


def test_sigma_drives_sample_and_realization():
    reset_rng_stream(0)
    p = PersonaParam(mean=0.0, sigma=0.5)
    x = p.sample_exploration()
    pop = p.sample_population(4000, center=x)
    assert abs(pop.mean() - x) < 0.05
    assert 0.4 < pop.std() < 0.6


def test_update_sigma_uses_sample_time_snapshot():
    p = PersonaParam(mean=9.9, sigma=1.0)          # current mean is irrelevant
    assert np.isclose(p.sigma_step(sample=3.0, mu_at_sample=0.0,
                                   sigma_at_sample=1.0, advantage=1.0,
                                   lr_sigma=0.01), 0.08)   # 0.01 * (9 - 1)
    new = p.update_sigma(sample=3.0, mu_at_sample=0.0, sigma_at_sample=1.0,
                         advantage=1.0, lr_sigma=0.01)
    assert np.isclose(new, np.exp(0.08)) and np.isclose(p.sigma, new)


def test_sigma_step_caps_and_bounds():
    p = PersonaParam(mean=0.0, sigma=1.0, sigma_max=2.0)
    assert np.isclose(p.sigma_step(sample=100.0, mu_at_sample=0.0,
                                   sigma_at_sample=1.0, advantage=1.0,
                                   lr_sigma=1.0), 0.25)
    assert np.isclose(p.sigma_step(sample=0.0, mu_at_sample=0.0,
                                   sigma_at_sample=1.0, advantage=1.0,
                                   lr_sigma=1.0), -0.25)
    p.sigma = 1.9
    assert p.update_sigma(sample=1e6, mu_at_sample=0.0, sigma_at_sample=1.9,
                          advantage=1.0, lr_sigma=1.0) == 2.0


def test_sigma_max_default_keeps_large_initial_sigma():
    p = PersonaParam(mean=-65.0, sigma=5.0)
    assert p.sigma == 5.0 and p.sigma_max == 5.0
    q = PersonaParam(mean=1.0, sigma=0.005)
    assert q.sigma_max == 2.0                                    # default floor


def test_bounds_still_clip():
    p = PersonaParam(mean=0.0, sigma=1.0, min=0.0, max=2.0)
    p.update_mean(+10.0)
    assert p.mean == 2.0
