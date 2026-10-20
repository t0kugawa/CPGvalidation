import numpy as np
import pytest

from persona.strategy import FisherMatrix


def _ref_run(steps, names, noise_names, beta, warmup, lam, lr_mu):
    n = 2 * len(names) + len(noise_names)
    F = np.zeros((n, n)); Xb = np.zeros(n); G = np.zeros((n, 2))
    for (dth, sig, dlog, nd, A) in steps:
        nz = len(names)
        s = np.zeros(n); a = np.zeros(n)
        for i, (k, (d, sg, dl)) in enumerate(zip(names, zip(dth, sig, dlog))):
            s[i] = d / sg ** 2
            s[nz + i] = (d / sg) ** 2 - 1
            a[i] = d / sg ** 2
            a[nz + i] = dl
        for j, k in enumerate(noise_names):
            d, t = nd[j]
            s[2 * nz + j] = d / t ** 2
        F = beta * F + (1 - beta) * np.outer(s, s)
        Xb = beta * Xb + (1 - beta) * a
        G = beta * G + (1 - beta) * np.outer(Xb, A)
    return F, Xb, G


def test_recursions_match_reference():
    rng = np.random.default_rng(0)
    names = ["p0", "p1"]; noise = ["n0"]
    fm = FisherMatrix(param_names=names, noise_names=noise, beta=0.9,
                      warmup=0, lr_mu=0.1)
    site_list, noise_list, adv_list = [], [], []
    for t in range(50):
        d = rng.normal(0, 1, 2) * 0.1; sg = rng.uniform(0.1, 1.0, 2)
        dl = rng.normal(0, 0.01, 2); nd = (rng.normal(0, 0.1), 0.5)
        A = rng.normal(0, 1, 2)
        fm.step({k: (float(d[i]), float(sg[i]), float(dl[i])) for i, k in enumerate(names)},
                {noise[0]: nd}, A)
        site_list.append((d.tolist(), sg.tolist(), dl.tolist()))
        noise_list.append([nd]); adv_list.append(A.tolist())
    F_ref, X_ref, G_ref = _ref_run(
        list(zip([s[0] for s in site_list], [s[1] for s in site_list],
                 [s[2] for s in site_list], noise_list, adv_list)),
        names, noise, 0.9, 0, 1e-3, 0.1)
    assert np.allclose(fm.F, F_ref, atol=1e-10)
    assert np.allclose(fm.Xbar, X_ref, atol=1e-10)
    assert np.allclose(fm.G, G_ref, atol=1e-10)


def test_delayed_credit_ranks_coupled_site():
    """A_t depends on the sample k steps back; the trace's memory must credit `coupled`."""
    rng = np.random.default_rng(1)
    names = ["coupled", "inert"]
    fm = FisherMatrix(param_names=names, beta=0.95, warmup=0, lr_mu=0.1)
    k = 4
    hist = []
    for t in range(3000):
        d = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3)])
        sg = np.array([0.3, 0.3])
        a_t = np.array([1.0, 1.0]) * (hist[t - k][0] if t >= k else 0.0)
        fm.step({"coupled": (d[0], sg[0], 0.0), "inert": (d[1], sg[1], 0.0)},
                {}, a_t)
        hist.append(d)
    deltas = fm.step({"coupled": (0.0, 0.3, 0.0), "inert": (0.0, 0.3, 0.0)}, {},
                     np.array([1.0, 1.0]))
    assert deltas["coupled"] > 0.0
    assert abs(deltas["inert"]) < 0.1 * deltas["coupled"]


def test_step_is_unit_covariant():
    """Scaling a site's units by c (θ→cθ, σ→cσ) must scale its step as c.

    With raw Δθ trace rows each site's gain acquires a spurious σ² factor
    instead, so this pins the score-unit layout (spec §4, V22)."""
    def run(sa):
        rng = np.random.default_rng(2)
        fm = FisherMatrix(param_names=["a", "b"], beta=0.9, warmup=0,
                          lambda_reg=1e-6)
        for _ in range(400):
            d = np.array([rng.normal(0, sa), rng.normal(0, 0.01)])
            A = np.array([d[0] / sa, 0.0])   # same physical plant in both unit systems
            fm.step({"a": (d[0], sa, 0.0), "b": (d[1], 0.01, 0.0)}, {}, A)
        return fm.step({"a": (0.0, sa, 0.0), "b": (0.0, 0.01, 0.0)}, {},
                       np.array([1.0, 0.0]))

    d1 = run(0.01)
    d2 = run(0.001)
    assert abs(d1["a"]) > 1e-6
    assert np.isclose(d2["a"], 0.1 * d1["a"], rtol=1e-4)


def test_warmup_and_noise_not_in_trace():
    fm = FisherMatrix(param_names=["p"], noise_names=["n"], warmup=3, beta=0.9)
    for t in range(3):
        out = fm.step({"p": (0.1, 0.1, 0.01)}, {"n": (0.5, 0.5)},
                      np.array([1.0, 1.0]))
        assert out["p"] == 0.0
    # layout is [μ_p, σ_p, noise_n]: the noise row is never traced
    assert fm.Xbar[2] == 0.0
    assert fm.Xbar[1] != 0.0
    out = fm.step({"p": (0.1, 0.1, 0.01)}, {"n": (0.5, 0.5)},
                  np.array([1.0, 1.0]))
    assert abs(out["p"]) > 0.0
    assert set(out) == {"p"}                      # noise never gets a step


def test_noise_score_uses_producer_current_value_from_the_first_sample():
    beta = 0.5
    fm = FisherMatrix(param_names=[], noise_names=["n"], beta=beta)
    fm.step({}, {"n": (0.5, 2.0)}, np.array([1.0, 1.0]))
    # s = δ/R²; no estimation, no "n > 2" gate: the first sample already counts
    assert np.isclose(fm.F[0, 0], (1.0 - beta) * (0.5 / 2.0 ** 2) ** 2)
    assert fm.noise_scales == {"n": 2.0}

    fm0 = FisherMatrix(param_names=[], noise_names=["n"], beta=beta)
    fm0.step({}, {"n": (0.5, 0.0)}, np.array([1.0, 1.0]))
    assert fm0.F[0, 0] == 0.0                      # no current value = no influence
    assert fm0.noise_scales == {}


def test_few_steps_singular_no_crash():
    fm = FisherMatrix(param_names=[f"p{i}" for i in range(40)], warmup=0, lr_mu=0.1)
    rng = np.random.default_rng(3)
    for _ in range(3):
        fm.step({f"p{i}": (float(rng.standard_normal()), 1.0, 0.0)
                 for i in range(40)}, {}, np.array([1.0, 1.0]))
    d = fm.step({f"p{i}": (float(rng.standard_normal()), 1.0, 0.0)
                 for i in range(40)}, {}, np.array([1.0, 1.0]))
    assert set(d) == {f"p{i}" for i in range(40)}
    assert all(np.isfinite(v) for v in d.values())
    assert np.all(np.isfinite(fm.feature_scales())) and np.all(fm.feature_scales() > 0)


def test_influence_ranks_high_impact_first():
    fm = FisherMatrix(param_names=["coupled", "inert"], beta=0.95, warmup=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        d = rng.standard_normal(2) * 0.3
        fm.step({"coupled": (d[0], 0.3, 0.0), "inert": (d[1], 0.3, 0.0)}, {},
                np.array([d[0] / 0.3, d[0] / 0.3]))   # A tracks `coupled` only
    infl = fm.influence()
    assert infl.shape == (2 * fm.n_sites,)
    assert max(infl[0], infl[1]) > 5 * max(infl[2], infl[3])
    assert np.all(np.isfinite(infl)) and np.all(infl >= 0.0)


def test_all_names_is_blocked_and_matches_row_count():
    fm = FisherMatrix(param_names=["s1", "s2"], noise_names=["n1"], warmup=0)
    assert fm.all_names == ["s1 μ", "s2 μ", "s1 σ", "s2 σ", "noise n1"]
    assert len(fm.all_names) == fm.n_total == 2 * 2 + 1
    fm.ensure_names(["s3"], ["n2"])
    assert len(fm.all_names) == fm.n_total
    assert fm.all_names[:3] == ["s1 μ", "s2 μ", "s3 μ"]
    assert fm.all_names[3:6] == ["s1 σ", "s2 σ", "s3 σ"]
    assert fm.all_names[6:] == ["noise n1", "noise n2"]


def test_inert_low_sigma_parameter_does_not_drift():
    # RF1: an inert low-σ site's moves do not covary with A, so its (ensemble)
    # step must stay ~0 — the metric must not amplify its noise into a bound.
    # One run cannot show this: G is a forgetting statistic (memory 1/(1−β)),
    # so the test averages over independent runs.
    sig = 0.005
    drive_steps, inert_steps = [], []
    for seed in range(120):
        fm = FisherMatrix(param_names=["drive", "inert"], beta=0.99, warmup=0,
                          lambda_reg=1e-3)
        rng = np.random.default_rng(seed)
        for _ in range(300):
            d = rng.standard_normal(2) * sig
            fm.step({"drive": (d[0], sig, 0.0), "inert": (d[1], sig, 0.0)}, {},
                    np.array([d[0] / sig * 0.1, 0.0]))   # drive only
        out = fm.step({"drive": (0.0, sig, 0.0), "inert": (0.0, sig, 0.0)}, {},
                      np.array([1.0, 1.0]))
        drive_steps.append(out["drive"])
        inert_steps.append(out["inert"])
    m_drive = float(np.mean(drive_steps))
    m_inert = float(np.mean(inert_steps))
    assert m_drive > 0.0                            # the drive site's signal
    assert abs(m_inert) < 0.25 * abs(m_drive)       # inert: no amplified drift


def test_credit_gain_is_ewma_weighted():
    """
    E[G] = (1−β)·E[trace·A] for a same-step coupling (spec §4): the trace hands
    the lagged sample the EWMA weight (1−β)βᵏ, so the credit (and the effective
    step) carries that gain. Pinned here so the estimator's scale is explicit
    rather than silently tuned against the old windowed Cov(Δθ, A).
    """
    sig, k, beta = 0.5, 2.0, 0.9
    expected = (1.0 - beta) * k                 # (1−β)·E[(d/σ²)·(k·d)], E[d²]=σ²
    vals = []
    for seed in range(160):
        fm = FisherMatrix(param_names=["p"], beta=beta, warmup=3, lr_mu=0.0)
        rng = np.random.default_rng(seed)
        for _ in range(400):
            d = float(rng.normal(0, sig))
            fm.step({"p": (d, sig, 0.0)}, {}, np.array([k * d, k * d]))
        vals.append(float(np.mean(fm.G[0, :])))
    got = float(np.mean(vals))
    assert abs(got - expected) < 0.15           # full-scale E[trace·A] would be k = 2.0


def test_ensure_names_pads_and_preserves_order():
    fm = FisherMatrix(param_names=["a"], noise_names=["n"], beta=0.9)
    rng = np.random.default_rng(7)
    for _ in range(20):
        fm.step({"a": (float(rng.normal()), 0.5, 0.0)},
                {"n": (float(rng.normal()), 0.5)}, np.array([1.0, 1.0]))
    F0, X0, G0 = fm.F.copy(), fm.Xbar.copy(), fm.G.copy()
    fm.ensure_names(["a", "b"], ["n"])
    assert fm.param_names == ["a", "b"] and fm.F.shape == (5, 5)
    # old μ_a/σ_a/noise blocks are preserved at their new indices (μ_a=0, σ_a=2, noise=4)
    assert fm.F[0, 0] == F0[0, 0] and fm.F[2, 2] == F0[1, 1]
    assert fm.F[0, 2] == F0[0, 1] and fm.F[0, 4] == F0[0, 2] and fm.F[2, 4] == F0[1, 2]
    assert fm.F[1, :].sum() == 0.0 and fm.F[:, 1].sum() == 0.0
    assert np.array_equal(fm.Xbar[[0, 2, 4]], X0)
    assert fm.Xbar[1] == 0.0
    assert np.array_equal(fm.G[[0, 2, 4]], G0)


def _direct(fm, ridge):
    return np.linalg.inv(fm.F + ridge * np.eye(fm.n_total))


def test_recursive_inverse_tracks_the_decaying_ridge():
    beta, lam = 0.9, 0.5
    fm = FisherMatrix(param_names=["p"], beta=beta, lambda_reg=lam,
                      warmup=0, refresh_every=10**9)
    fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta), atol=1e-10)
    fm.step({"p": (-0.1, 0.4, 0.0)}, {}, np.array([0.0, 1.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta ** 2), atol=1e-10)


def test_kmax_refresh_restores_the_constant_ridge():
    beta, lam = 0.9, 0.5
    fm = FisherMatrix(param_names=["p"], beta=beta, lambda_reg=lam,
                      warmup=0, refresh_every=2)
    fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta), atol=1e-10)
    fm.step({"p": (-0.1, 0.4, 0.0)}, {}, np.array([0.0, 1.0]))
    assert np.allclose(fm.P, _direct(fm, lam), atol=1e-10)


def test_default_kmax_tracks_beta():
    assert FisherMatrix(param_names=["p"], beta=0.995).refresh_every == 10
    assert FisherMatrix(param_names=["p"], beta=0.99).refresh_every == 5
    assert FisherMatrix(param_names=["p"], beta=0.9).refresh_every == 1
    assert FisherMatrix(param_names=["p"], beta=0.0, refresh_every=5).refresh_every == 1
    assert FisherMatrix(param_names=["p"], beta=1.0).refresh_every >= 10**6


def test_invalid_beta_and_ridge_are_rejected():
    for beta in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            FisherMatrix(param_names=["p"], beta=beta)
    for lam in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            FisherMatrix(param_names=["p"], lambda_reg=lam)


def test_k1_step_uses_direct_solve_without_building_p():
    beta, lam = 0.9, 0.5
    fm = FisherMatrix(param_names=["p"], beta=beta, lambda_reg=lam,
                      warmup=0, refresh_every=1, lr_mu=0.1)
    calls = []
    orig = fm._refresh_inverse

    def spy():
        calls.append(1)
        return orig()

    fm._refresh_inverse = spy
    out = fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    assert calls == []
    g = fm.G.sum(axis=1)
    g[2 * fm.n_sites:] = 0.0
    expected = np.linalg.solve(fm.F + lam * np.eye(fm.n_total), g)
    assert np.isclose(out["p"], 0.1 * expected[0])
    P = fm.P
    assert calls == [1]
    assert np.allclose(P, np.linalg.inv(fm.F + lam * np.eye(fm.n_total)),
                       atol=1e-10)


def test_beta_zero_forces_constant_ridge_and_stays_finite():
    fm = FisherMatrix(param_names=["p"], beta=0.0, lambda_reg=0.5,
                      warmup=0, refresh_every=5, lr_mu=0.1)
    assert fm.refresh_every == 1
    out = fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    g = fm.G.sum(axis=1)
    g[2 * fm.n_sites:] = 0.0
    expected = np.linalg.solve(fm.F + 0.5 * np.eye(fm.n_total), g)
    assert np.isclose(out["p"], 0.1 * expected[0])
    assert np.all(np.isfinite(fm.P))


def test_beta_one_keeps_the_state_finite():
    fm = FisherMatrix(param_names=["p"], beta=1.0, lambda_reg=0.5,
                      warmup=0)
    out = fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    assert out["p"] == 0.0
    assert np.array_equal(fm.F, np.zeros_like(fm.F))
    assert np.allclose(fm.P, 2.0 * np.eye(fm.n_total))


def test_nonfinite_inverse_is_repaired():
    fm = FisherMatrix(param_names=["p"], beta=0.9, lambda_reg=0.5,
                      warmup=0, refresh_every=10**9)
    fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    fm.P = np.full_like(fm.P, np.nan)
    out = fm.step({"p": (0.1, 0.4, 0.0)}, {}, np.array([0.0, 1.0]))
    assert np.all(np.isfinite(fm.P))
    assert np.isfinite(out["p"])


def test_ensure_names_rebuilds_the_inverse():
    fm = FisherMatrix(param_names=["a"], beta=0.9, lambda_reg=0.5,
                      warmup=0, refresh_every=10**9)
    for _ in range(3):
        fm.step({"a": (0.1, 0.5, 0.0)}, {}, np.array([1.0, 0.0]))
    assert fm._steps_since_refresh == 3
    fm.ensure_names(["a", "b"], [])
    assert fm._steps_since_refresh == 0
    assert fm.P.shape == (4, 4)
    assert np.allclose(fm.P @ (fm.F + fm.lambda_reg * np.eye(4)), np.eye(4),
                       atol=1e-10)


def test_recursive_step_matches_the_direct_solve():
    rng = np.random.default_rng(11)
    beta, lam, kmax = 0.95, 1e-3, 4
    fm = FisherMatrix(param_names=["p"], noise_names=["n"], beta=beta,
                      lambda_reg=lam, warmup=0, refresh_every=kmax, lr_mu=0.1)
    for _ in range(40):
        d = rng.normal(0, 0.2); sig = rng.uniform(0.2, 0.8)
        nd = (rng.normal(0, 0.1), 1.0)
        out = fm.step({"p": (d, sig, 0.0)}, {"n": nd},
                      np.array([1.0, -1.0]))
        g = fm.G.sum(axis=1); g[2 * fm.n_sites:] = 0.0
        k = fm._steps_since_refresh
        ridge = lam if k == 0 else lam * beta ** k
        expected = np.linalg.solve(fm.F + ridge * np.eye(fm.n_total), g)
        assert np.allclose(fm.P @ g, expected, rtol=1e-8, atol=1e-10)
        assert np.isclose(out["p"], fm.lr_mu * expected[0],
                          rtol=1e-8, atol=1e-10)


def test_inverse_stays_symmetric_and_positive_definite():
    rng = np.random.default_rng(12)
    fm = FisherMatrix(param_names=[f"p{i}" for i in range(10)],
                      beta=0.99, warmup=0, refresh_every=5)
    for _ in range(200):
        sites = {f"p{i}": (float(rng.normal(0, 0.2)), 0.5, 0.0)
                 for i in range(10)}
        fm.step(sites, {}, rng.normal(0, 1, 2))
        assert 0 <= fm._steps_since_refresh < 5
        assert np.all(np.isfinite(fm.P))
        assert np.allclose(fm.P, fm.P.T, atol=1e-10)
        assert np.linalg.eigvalsh(fm.P).min() > 0.0


def test_phase_shift_refreshes_before_kmax():
    beta, lam = 0.9, 0.5
    fm = FisherMatrix(param_names=["p"], beta=beta, lambda_reg=lam,
                      warmup=0, refresh_every=100)
    fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    fm.step({"p": (-0.1, 0.4, 0.0)}, {}, np.array([0.0, 1.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta ** 2), atol=1e-10)
    fm.step({"p": (0.05, 0.6, 0.0)}, {}, np.array([1.0, 0.0]),
            phase_shift=True)
    assert np.allclose(fm.P, _direct(fm, lam), atol=1e-10)
