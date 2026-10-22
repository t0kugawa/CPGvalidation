import numpy as np

from persona.homeostasis import (
    HomeostaticConfig,
    HomeostaticRegulator,
    PhysicalMap,
    PlayerSpec,
)
from persona.strategy import FisherMatrix


def _map():
    return PhysicalMap.from_specs(
        [PlayerSpec("p0p", "population", ("p0",), size=2, ptype="exc"),
         PlayerSpec("p1p", "population", ("p1",), size=2, ptype="inh")],
        [("p0p", "p1p")])


def _sequence(n=8, seed=13):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sites = {
            "p0": (float(rng.normal(0.0, 0.2)), 0.1, 0.0),
            "p1": (float(rng.normal(0.0, 0.2)), 0.2, 0.01),
        }
        noise = {"n0": (float(rng.normal(0.0, 0.1)), 0.5)}
        advantage = rng.normal(0.0, 1.0, 2)
        out.append((sites, noise, advantage))
    return out


def _run(fm, sequence):
    results = []
    for sites, noise, advantage in sequence:
        results.append(fm.step(sites, noise, advantage))
    return results


def test_fisher_default_has_no_regulator_and_matches_old_math():
    fm = FisherMatrix(param_names=["p0", "p1"], noise_names=["n0"],
                      beta=0.9, warmup=0, lr_mu=0.1)
    assert fm.homeostasis is None
    assert fm.dt_s == 0.01
    seq = _sequence()
    got = _run(fm, seq)
    for out in got:
        assert set(out) == {"p0", "p1"}
        assert np.all(np.isfinite(list(out.values())))


def test_lambda_zero_regulator_reproduces_old_deltas_exactly():
    pm = _map()
    cfg = HomeostaticConfig(enabled=True, h_burnin_steps=0, lambda_H=0.0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.9,
                               site_names=["p0", "p1"])
    base = FisherMatrix(param_names=["p0", "p1"], noise_names=["n0"],
                        beta=0.9, warmup=0, lr_mu=0.1)
    h = FisherMatrix(param_names=["p0", "p1"], noise_names=["n0"],
                     beta=0.9, warmup=0, lr_mu=0.1, homeostasis=reg)
    seq = _sequence()
    base_out = _run(base, seq)
    h_out = _run(h, seq)
    for a, b in zip(base_out, h_out):
        assert set(a) == set(b)
        for key in a:
            assert np.array_equal(a[key], b[key])
    assert np.array_equal(base.F, h.F)
    assert np.array_equal(base.Xbar, h.Xbar)
    assert np.array_equal(base.G, h.G)


def test_h_burnin_returns_old_deltas_then_activates_without_mutating_G():
    pm = _map()
    cfg = HomeostaticConfig(enabled=True, h_burnin_steps=2)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.9,
                               site_names=["p0", "p1"])
    base = FisherMatrix(param_names=["p0", "p1"], noise_names=["n0"],
                        beta=0.9, warmup=0, lr_mu=0.1)
    h = FisherMatrix(param_names=["p0", "p1"], noise_names=["n0"],
                     beta=0.9, warmup=0, lr_mu=0.1, homeostasis=reg)
    seq = _sequence(5)
    base_out = _run(base, seq)
    h_out = []
    reg_stages = []
    for sites, noise, advantage in seq:
        h_out.append(h.step(sites, noise, advantage))
        reg_stages.append(reg.last["stage"])
    assert np.array_equal(base.F, h.F)
    assert np.array_equal(base.Xbar, h.Xbar)
    assert np.array_equal(base.G, h.G)
    for t in range(2):
        assert np.array_equal(base_out[t]["p0"], h_out[t]["p0"])
        assert reg_stages[t] == "burnin"
    assert reg.stage == "active"
    assert reg.last["stage"] == "active"
    assert reg.last["u_H"].shape == (2,)
    assert np.all(np.isfinite(list(h_out[-1].values())))


def test_strategy_ensure_names_maps_new_site_without_crashing():
    pm = _map()
    cfg = HomeostaticConfig(enabled=True, h_burnin_steps=0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.9,
                               site_names=["p0", "p1"])
    fm = FisherMatrix(param_names=["p0", "p1"], beta=0.9, warmup=0,
                      lr_mu=0.1, homeostasis=reg)
    fm.step({"p0": (0.1, 0.1, 0.0), "p1": (0.2, 0.2, 0.0)}, {},
            np.array([1.0, 0.0]))
    out = fm.step({"p0": (0.1, 0.1, 0.0), "p1": (0.2, 0.2, 0.0),
                   "late": (0.3, 0.1, 0.0)}, {}, np.array([1.0, 0.0]))
    assert "late" in out and np.isfinite(out["late"])
    assert "late" in reg.site_names
    assert "late" in reg.g_ref
    assert "late" in pm.player_of
    assert pm.specs[pm.player_of["late"]].kind == "unknown"


def test_all_modes_run_through_the_strategy_and_log_diagnostics():
    keys = {
        "stage", "mode", "T", "V_obs", "V_A0", "V_F", "V_F0",
        "S_c", "S_c_raw", "S_c_prev", "c", "z", "mu", "mu_prev", "v",
        "ell", "a", "pi", "e", "e_bar", "projected_pi", "u_H",
        "lambda_H", "tau_H", "correction_norm", "gate",
        "skipped_endpoints", "player_names", "step", "burnin_step",
    }
    seq = _sequence(4)
    for mode in ("advantage", "budget", "fisher", "band"):
        pm = _map()
        cfg = HomeostaticConfig(enabled=True, h_burnin_steps=0,
                                global_variance_mode=mode)
        reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.9,
                                   site_names=["p0", "p1"])
        fm = FisherMatrix(param_names=["p0", "p1"], noise_names=["n0"],
                          beta=0.9, warmup=0, lr_mu=0.1, homeostasis=reg)
        outs = _run(fm, seq)
        assert all(np.all(np.isfinite(list(o.values()))) for o in outs)
        assert keys <= set(reg.last)
        assert reg.last["mode"] == mode
        assert np.isfinite(reg.last["T"])
        if mode == "band":
            assert 0.5 <= reg.last["T"] <= 2.0
