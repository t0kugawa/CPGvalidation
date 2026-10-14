import numpy as np

from persona.strategy import FisherMatrix


def make_hist_fisher(n_feat=4, noise_names=None, scales=None, window=50,
                     lambda_reg=1e-3, seed=0, n_samples=20):
    names = [f"p{i}" for i in range(n_feat)]
    fm = FisherMatrix(param_names=names, noise_names=list(noise_names or []),
                      window_size=window, lambda_reg=lambda_reg)
    rng = np.random.default_rng(seed)
    if scales is None:
        scales = np.ones(n_feat, dtype=np.float64)
    for _ in range(n_samples):
        base = rng.standard_normal(n_feat)
        pert = {names[i]: float(base[i] * scales[i]) for i in range(n_feat)}
        fm.record_step(pert, np.array([0.0]), ("stance",))
    return fm


def test_correlation_unit_diagonal_and_symmetric():
    fm = make_hist_fisher(n_feat=6, scales=[1e-3, 1.0, 1e3, 1e-1, 50.0, 0.2])
    F = fm.compute_correlation_matrix()
    assert F.shape == (fm.n_total, fm.n_total)
    assert np.allclose(np.diag(F), 1.0, atol=1e-6)
    assert np.allclose(F, F.T, atol=1e-9)
    assert np.linalg.eigvalsh(F).min() > 0  # ridge => PD even when rank-deficient


def test_mixed_units_unified():
    F1 = make_hist_fisher(n_feat=4, scales=[1e-3, 1.0, 1e3, 1e-1]).compute_correlation_matrix()
    F2 = make_hist_fisher(n_feat=4, scales=[1.0, 1.0, 1.0, 1.0]).compute_correlation_matrix()
    assert np.allclose(np.sort(F1.ravel()), np.sort(F2.ravel()), atol=1e-8)


def test_feature_scales_present():
    fm = make_hist_fisher(n_feat=3, scales=[2.0, 1.0, 0.5])
    fm.compute_correlation_matrix()
    s = fm.feature_scales()
    assert s.shape == (fm.n_total,)
    assert np.all(np.isfinite(s)) and np.all(s > 0)


def test_noise_features_enter_fisher_not_score():
    fm = make_hist_fisher(n_feat=3, noise_names=["noise_x"])
    deltas = fm.compute_natural_gradient_step(
        {"p0": 1.0, "p1": 1.0, "p2": 1.0},
        {"p0": 0.0, "p1": 0.0, "p2": 0.0},
        {"p0": 1.0, "p1": 1.0, "p2": 1.0},
        np.array([1.0]),
    )
    assert set(deltas) == {"p0", "p1", "p2"}  # never a noise key


def test_singular_window_no_crash():
    fm = make_hist_fisher(n_feat=40, window=5)
    d = fm.compute_natural_gradient_step(
        {f"p{i}": 0.1 for i in range(40)},
        {f"p{i}": 0.0 for i in range(40)},
        {f"p{i}": 1.0 for i in range(40)},
        np.array([1.0]),
    )
    assert set(d) == {f"p{i}" for i in range(40)}
    assert all(np.isfinite(v) for v in d.values())