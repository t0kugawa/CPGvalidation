import numpy as np

from persona.strategy import FisherMatrix


def make_metric(n_feat=4, noise_names=None, window=50, seed=0, n_samples=120):
    names = [f"p{i}" for i in range(n_feat)]
    fm = FisherMatrix(param_names=names, noise_names=list(noise_names or []),
                      window_size=window)
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        dth = rng.standard_normal(n_feat)
        A = np.array([dth[0], dth[1] if n_feat > 1 else 0.0])  # p0, p1 drive A
        fm.record_response(dth, A)
    return fm


def test_metric_shape_symmetric_pd():
    fm = make_metric(n_feat=6)
    F, scales = fm.compute_metric()
    assert F.shape == (fm.n_total, fm.n_total)
    assert np.allclose(F, F.T, atol=1e-9)
    assert np.linalg.eigvalsh(F).min() > 0  # ridge => PD even when rank-deficient
    assert scales.shape == (fm.n_total,)
    assert np.all(np.isfinite(scales)) and np.all(scales > 0)


def test_influence_ranks_impact_on_error():
    fm = make_metric(n_feat=4)  # p0, p1 covary with A; p2, p3 do not
    infl = fm.influence()
    assert infl.shape == (fm.n_total,)
    assert infl[0] > infl[2] and infl[1] > infl[3]
    assert not np.allclose(infl, 1.0)


def test_noise_features_enter_metric_not_score():
    fm = make_metric(n_feat=3, noise_names=["noise_x"])
    deltas = fm.compute_natural_gradient_step(
        {"p0": 1.0, "p1": 1.0, "p2": 1.0},
        {"p0": 0.0, "p1": 0.0, "p2": 0.0},
        {"p0": 1.0, "p1": 1.0, "p2": 1.0},
        np.array([1.0]),
    )
    assert set(deltas) == {"p0", "p1", "p2"}  # never a noise key


def test_singular_window_no_crash():
    fm = FisherMatrix(param_names=[f"p{i}" for i in range(40)], window_size=5)
    rng = np.random.default_rng(3)
    for _ in range(3):
        fm.record_response(rng.standard_normal(40), rng.standard_normal(2))
    d = fm.compute_natural_gradient_step(
        {f"p{i}": 0.1 for i in range(40)},
        {f"p{i}": 0.0 for i in range(40)},
        {f"p{i}": 1.0 for i in range(40)},
        np.array([1.0, 1.0]),
    )
    assert set(d) == {f"p{i}" for i in range(40)}
    assert all(np.isfinite(v) for v in d.values())