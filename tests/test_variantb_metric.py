import numpy as np

from persona.strategy import FisherMatrix


def _fm(n=5, window=600, lam=1e-9):
    names = [f"p{i}" for i in range(n)]
    return FisherMatrix(param_names=names, window_size=window, lambda_reg=lam)


def test_influence_ranks_high_impact_first():
    fm = _fm()
    rng = np.random.default_rng(0)
    for _ in range(500):
        dth = rng.standard_normal(5) * 0.1
        A = np.array([dth[0] * 10.0, dth[0] * 5.0])  # only p0 covaries with A
        fm.record_response(dth, A)
    infl = fm.influence()
    assert infl.shape == (5,)
    assert infl[0] > 0.9 and np.all(infl[1:] < 0.1)


def test_short_window_is_finite():
    fm = _fm(n=40, window=3)
    rng = np.random.default_rng(1)
    for _ in range(3):
        fm.record_response(rng.standard_normal(40), rng.standard_normal(2))
    F, s = fm.compute_metric()
    assert np.isfinite(F).all() and np.isfinite(s).all()
    d = fm.compute_natural_gradient_step(
        {f"p{i}": 0.1 for i in range(40)}, {f"p{i}": 0.0 for i in range(40)},
        {f"p{i}": 1.0 for i in range(40)}, np.array([1.0, 1.0]))
    assert all(np.isfinite(v) for v in d.values())


def test_step_is_unit_invariant():
    rng = np.random.default_rng(2)
    base = [rng.standard_normal(5) * 0.1 for _ in range(100)]

    def build(c):
        fm = _fm()
        for dth in base:
            A = np.array([dth[0] * 10.0, dth[1]])   # reward from the physical dth
            rec = dth.copy(); rec[0] *= c            # same data, p0 units rescaled
            fm.record_response(rec, A)
        return fm

    def step(fm):
        return fm.compute_natural_gradient_step(
            {f"p{i}": 1.0 for i in range(5)}, {f"p{i}": 0.0 for i in range(5)},
            {f"p{i}": 1.0 for i in range(5)}, np.array([1.0, 1.0]))

    d1 = step(build(1.0))["p0"]
    d2 = step(build(10.0))["p0"]
    # The physical step (Δμ·scale) is independent of the arbitrary unit scale.
    assert np.isclose(d2 * 10.0, d1, rtol=1e-6)