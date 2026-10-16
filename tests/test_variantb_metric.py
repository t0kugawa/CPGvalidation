import numpy as np

from persona.strategy import FisherMatrix


def _fm(n=5, window=60):
    names = [f"p{i}" for i in range(n)]
    return FisherMatrix(param_names=names, window_size=window, lambda_reg=1e-3)


def test_influence_ranks_high_impact_first():
    fm = _fm()
    rng = np.random.default_rng(0)
    for _ in range(50):
        dth = rng.standard_normal(5) * 0.1
        dy = np.array([dth[0] * 10.0, dth[0] * 5.0])  # only p0 drives y
        fm.record_response(dth, dy)
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
    # Physical plant: y0 = 10 * q, where q = p0. p0 is recorded in units that
    # make its perturbation numbers scale0 times larger; dy stays physical.
    rng = np.random.default_rng(2)
    q = rng.standard_normal(40) * 0.1
    other = rng.standard_normal((40, 4)) * 0.1

    def build(scale0):
        fm = _fm()
        for i in range(40):
            d = np.array([scale0 * q[i], other[i, 0], other[i, 1], other[i, 2], other[i, 3]])
            fm.record_response(d, np.array([10.0 * q[i], other[i, 0]]))
        return fm

    def step(fm):
        return fm.compute_natural_gradient_step(
            {f"p{i}": 1.0 for i in range(5)}, {f"p{i}": 0.0 for i in range(5)},
            {f"p{i}": 1.0 for i in range(5)}, np.array([1.0, 1.0]))["p0"]

    # Same physical step; a 10x-smaller unit gives 10x the step number.
    assert np.isclose(step(build(10.0)), 10.0 * step(build(1.0)), rtol=1e-2)