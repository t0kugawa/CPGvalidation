from types import SimpleNamespace

import numpy as np

from persona.connection import Connection
from persona.homeostasis import (
    GlobalVarianceRegulator,
    HomeostaticConfig,
    HomeostaticRegulator,
    PhysicalMap,
    PlayerSpec,
    gaussian_log_score,
)
from persona.parameter import PersonaParam
from persona.population import Population


def _param(mean=1.0, sigma=0.1, lo=-10.0, hi=10.0):
    return PersonaParam(mean=mean, sigma=sigma, min=lo, max=hi)


def _fake_registry():
    pop_a = Population("izhikevich", 5, {"a": _param(), "b": _param()}, "A")
    pop_in = Population("izhikevich", 3, {"a": _param(), "b": _param()}, "in1")
    conns = {
        "A_in1": Connection(
            "A", "in1", parameters={"weight": _param(2.0, lo=0.0, hi=10.0)},
            key="A_in1"),
        "A_in1_inh": Connection(
            "A", "in1", parameters={"weight": _param(-2.0, lo=-10.0, hi=0.0)},
            key="A_in1_inh"),
        "in1_A": Connection(
            "in1", "A", parameters={"weight": _param(3.0, lo=0.0, hi=10.0)},
            key="in1_A"),
        "OUT_A": Connection(
            "OUT", "A", parameters={"weight": _param(1.5, lo=0.0, hi=10.0)},
            key="OUT_A"),
        "OUT_OTHER": Connection(
            "OUT", "OTHER", parameters={"weight": _param(1.5, lo=0.0, hi=10.0)},
            key="OUT_OTHER"),
    }
    counts = {"A_in1": 7, "A_in1_inh": 0, "in1_A": 4,
              "OUT_A": 2, "OUT_OTHER": 6}
    reg = SimpleNamespace(
        population_cfgs={"A": pop_a, "in1": pop_in},
        connection_cfgs=conns,
        population_nodes={"A": object(), "in1": object(),
                          "OUT": object(), "OTHER": object()},
        _syn_counts=dict(counts),
        synapse_count=lambda key: counts.get(key, 0),
    )
    return reg


def _by_name(pm):
    return {spec.name: spec for spec in pm.specs}


def test_physical_map_covers_every_site_once_and_keeps_sizes_vs_counts():
    pm = PhysicalMap.from_registry(_fake_registry())
    specs = _by_name(pm)
    assert set(pm.player_of) == {
        "pop_A_a", "pop_A_b", "pop_in1_a", "pop_in1_b",
        "conn_A_in1_weight", "conn_A_in1_inh_weight",
        "conn_in1_A_weight", "conn_OUT_A_weight",
        "conn_OUT_OTHER_weight",
    }
    assert len({pm.player_of[k] for k in pm.player_of}) == len(pm.specs)
    assert len(pm.player_of) == 9

    a = specs["pop_A"]
    assert a.kind == "population" and a.ptype == "exc"
    assert a.size == 5 and a.m_count == 2 and a.param_names == ("pop_A_a", "pop_A_b")
    inh = specs["pop_in1"]
    assert inh.kind == "population" and inh.ptype == "inh"
    assert inh.size == 3 and inh.m_count == 2

    # Registry synapse counts win when present, including a real zero.
    assert specs["con_A_in1"].kind == "synapse"
    assert specs["con_A_in1"].ptype == "static_exc"
    assert specs["con_A_in1"].size == 7 and specs["con_A_in1"].m_count == 1
    assert specs["con_A_in1_inh"].ptype == "static_inh"
    assert specs["con_A_in1_inh"].size == 0


def test_physical_map_incidence_antagonism_reciprocity_and_skipped_endpoints():
    pm = PhysicalMap.from_registry(_fake_registry())
    idx = pm.player_of
    names = {spec.name: i for i, spec in enumerate(pm.specs)}

    def edge(a, b):
        return bool(pm.adjacency[names[a], names[b]])

    # Incidence; OUT is a boundary endpoint and is skipped, not treated as a player.
    assert edge("pop_A", "con_A_in1")
    assert edge("con_A_in1", "pop_in1")
    assert edge("pop_in1", "con_in1_A")
    assert edge("con_in1_A", "pop_A")
    assert edge("con_OUT_A", "pop_A")
    assert "OUT" in pm.skipped_endpoints
    assert "OTHER" in pm.skipped_endpoints

    # Antagonism between the two opposite-sign A -> in1 projections and between
    # their source populations (self pairs are ignored).
    assert edge("con_A_in1", "con_A_in1_inh")
    # Reciprocity between each A -> in1 synapse and the in1 -> A synapse.
    assert edge("con_A_in1", "con_in1_A")
    assert edge("con_A_in1_inh", "con_in1_A")

    # The projection whose both endpoints are boundaries is a singleton.
    singleton = names["con_OUT_OTHER"]
    assert pm.adjacency[singleton].sum() == 0.0
    assert pm.laplacian[singleton].sum() == 0.0

    # The physical map has no duplicate/weighted edge proposals.
    assert np.all((pm.adjacency == 0.0) | (pm.adjacency == 1.0))


def test_physical_map_simple_graph_ignores_duplicate_specs_edges():
    specs = [
        PlayerSpec("a", "population", ("a_x",), size=2, ptype="exc"),
        PlayerSpec("b", "population", ("b_x",), size=2, ptype="inh"),
    ]
    pm = PhysicalMap.from_specs(specs, [("a", "b"), ("b", "a"), (0, 1), ("a", "b")])
    assert pm.adjacency[0, 1] == 1.0
    assert pm.adjacency[1, 0] == 1.0
    assert pm.adjacency.sum() == 2.0


def test_physical_map_h_is_symmetric_psd_zero_sum_and_fixed():
    pm = PhysicalMap.from_registry(_fake_registry())
    H = pm.H
    assert H.shape == (len(pm.specs), len(pm.specs))
    assert np.allclose(H, H.T, atol=1e-15)
    assert np.allclose(H @ np.ones(len(pm.specs)), 0.0, atol=1e-12)
    assert np.allclose(np.ones(len(pm.specs)) @ H, 0.0, atol=1e-12)
    assert np.linalg.eigvalsh(H).min() > -1e-12
    assert pm.H is H

    singleton = pm.specs.index(next(s for s in pm.specs if s.name == "con_OUT_OTHER"))
    # Isolated Laplacian row is zero; the eps_H regularizer is the only coupling.
    assert np.allclose(pm.laplacian[singleton], 0.0)
    assert np.any(np.abs(H[singleton]) > 0.0)


def test_physical_map_unknown_names_become_singletons_without_losing_old_map():
    pm = PhysicalMap.from_specs(
        [PlayerSpec("a", "population", ("a_x",), size=2),
         PlayerSpec("b", "population", ("b_x",), size=2)],
        [("a", "b")])
    old = pm.player_of["a_x"]
    old_specs = list(pm.specs)
    pm.ensure_names(["a_x", "b_x", "late_site"])
    assert pm.player_of["a_x"] == old
    assert pm.specs[:len(old_specs)] == old_specs
    late_idx = pm.player_of["late_site"]
    assert pm.specs[late_idx].kind == "unknown"
    assert pm.specs[late_idx].param_names == ("late_site",)
    assert pm.H.shape == (3, 3)
    assert np.allclose(pm.H @ np.ones(3), 0.0)

    # A later rebuild is lazy: touching H after ensure_names returns the new shape.
    assert pm._H is not None


def test_physical_map_unknown_site_after_registry_map_is_singleton():
    pm = PhysicalMap.from_registry(_fake_registry())
    n0 = len(pm.specs)
    pm.ensure_names(["late_site"])
    assert pm.player_of["late_site"] == n0
    assert pm.specs[-1].name == "unknown:late_site"
    assert pm.specs[-1].kind == "unknown"
    assert pm.H.shape == (n0 + 1, n0 + 1)


# --- score and share math ----------------------------------------------------


def _two_player_map():
    specs = [
        PlayerSpec("p0", "population", ("s0",), size=1, ptype="exc"),
        PlayerSpec("p1", "population", ("s1",), size=1, ptype="inh"),
    ]
    return PhysicalMap.from_specs(specs, [("p0", "p1")])


def _config(**overrides):
    values = dict(enabled=True, h_burnin_steps=3,
                  tau_m_s=2.0, tau_S_s=2.0, tau_A_s=2.0, tau_H_s=5.0)
    values.update(overrides)
    return HomeostaticConfig(**values)


def test_config_defaults_and_validation():
    cfg = HomeostaticConfig(enabled=True)
    assert cfg.lambda_H == 1.0
    assert cfg.tau_H_s == 5.0
    assert cfg.tau_m_s == cfg.tau_S_s == cfg.tau_A_s == 2.0
    assert cfg.tau_V_s == 5.0
    assert cfg.h_burnin_steps == 100
    assert cfg.global_variance_mode == "advantage"
    assert cfg.lift_mode == "player"
    assert cfg.h_delay_steps == 0
    assert cfg.T_min == 0.1 and cfg.T_max == 10.0
    assert cfg.band_lo == 0.5 and cfg.band_hi == 2.0
    assert cfg.v_min == 1e-8 and cfg.v_max == 1e6
    assert cfg.S_min == 1e-12 and cfg.g_min == 1e-12
    assert cfg.eps_H == 1e-3 and cfg.eps_credit == 1e-9
    assert HomeostaticConfig().enabled is False

    for bad in (
        dict(global_variance_mode="nope"),
        dict(lift_mode="nope"), dict(h_delay_steps=-1),
        dict(tau_H_s=0.0), dict(tau_m_s=-1.0), dict(h_burnin_steps=-1),
        dict(lambda_H=-0.1), dict(T_min=0.0), dict(T_max=0.05),
        dict(v_min=0.0), dict(v_max=0.0), dict(S_min=0.0),
    ):
        with np.testing.assert_raises(ValueError):
            HomeostaticConfig(**bad)


def test_current_best_preset_is_experimental_centered_delay():
    cfg = HomeostaticConfig.current_best()
    assert cfg.enabled is True
    assert cfg.global_variance_mode == "fisher"
    assert cfg.lift_mode == "site_centered"
    assert cfg.h_delay_steps == 500
    assert cfg.lambda_H == 1.0


def test_site_centered_lift_is_zero_sum_within_each_player():
    pm = PhysicalMap.from_specs(
        [PlayerSpec("A", "population", ("a0", "a1"), size=2, ptype="exc"),
         PlayerSpec("B", "population", ("b0", "b1"), size=2, ptype="inh")],
        [("A", "B")])
    cfg = _config(h_burnin_steps=0, lift_mode="site_centered")
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["a0", "a1", "b0", "b1"])
    g0 = np.array([1.0, 2.0, 3.0, 4.0])
    reg.adjust_credit(g0, np.full(4, 0.1), 0.0, np.eye(4), 1)
    g1 = np.array([1.2, 2.4, 3.0, 4.2])
    out = reg.adjust_credit(g1, np.full(4, 0.1), 1.0, np.eye(4), 2)
    gh = out - g1
    assert np.isclose(float(gh.sum()), 0.0, atol=1e-12)
    assert np.isclose(float(gh[:2].sum()), 0.0, atol=1e-12)
    assert np.isclose(float(gh[2:].sum()), 0.0, atol=1e-12)
    assert np.any(np.abs(gh) > 0.0)


def test_h_delay_suppresses_correction_until_the_absolute_step():
    pm = PhysicalMap.from_specs(
        [PlayerSpec("A", "population", ("a0", "a1"), size=2, ptype="exc"),
         PlayerSpec("B", "population", ("b0", "b1"), size=2, ptype="inh")],
        [("A", "B")])
    cfg = _config(h_burnin_steps=0, lift_mode="site_centered",
                  h_delay_steps=3)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["a0", "a1", "b0", "b1"])
    g0 = np.array([1.0, 2.0, 3.0, 4.0])
    out0 = reg.adjust_credit(g0, np.full(4, 0.1), 0.0, np.eye(4), 1)
    assert np.array_equal(out0, g0)

    g1 = np.array([1.2, 2.4, 3.0, 4.2])
    out1 = reg.adjust_credit(g1, np.full(4, 0.1), 1.0, np.eye(4), 2)
    assert np.array_equal(out1, g1)
    assert reg.last["h_delayed"] is True
    assert np.all(reg.last["u_H"] == 0.0)

    out2 = reg.adjust_credit(g1, np.full(4, 0.1), 1.0, np.eye(4), 3)
    assert reg.last["h_delayed"] is False
    assert np.any(np.abs(reg.last["u_H"]) > 0.0)
    assert not np.array_equal(out2, g1)


def test_burnin_estimates_ref_and_initial_state_without_correction():
    pm = _two_player_map()
    cfg = _config(h_burnin_steps=3)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1"])
    g_hist = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    a_hist = np.array([0.5, 1.0, 1.5])
    F = 2.0 * np.eye(2)
    for t, (g, a) in enumerate(zip(g_hist, a_hist), start=1):
        out = reg.adjust_credit(g, np.array([0.1, 0.2]), float(a), F, t)
        assert np.array_equal(out, g)
        assert np.all(reg.last["u_H"] == 0.0)
        assert reg.last["stage"] == "burnin"

    gref = np.maximum(np.sqrt(np.mean(g_hist ** 2, axis=0)), cfg.g_min)
    assert np.allclose([reg.g_ref["s0"], reg.g_ref["s1"]], gref)
    c_hist = g_hist / gref
    S_c0 = max(float(np.mean(np.std(c_hist, axis=1))), cfg.S_min)
    assert np.isclose(reg.S_c, S_c0)
    z0 = (c_hist - c_hist.mean(axis=1, keepdims=True)) / S_c0
    assert np.allclose(reg.mu, z0.mean(axis=0))

    V_A0 = float(np.var(a_hist, ddof=1))
    V_F = float(np.trace(F) / 2.0)
    denom = max(1.0 - 0.5 ** 3, 0.1)
    V_F0 = V_F / denom
    assert np.isclose(reg.global_variance.V_A0, V_A0)
    assert np.isclose(reg.global_variance.V_obs, V_A0)
    assert np.isclose(reg.global_variance.mean_A, float(np.mean(a_hist)))
    assert np.isclose(reg.global_variance.V_F0, V_F0)
    assert np.isclose(reg.global_variance.V_F_ewma, V_F0)
    assert reg.stage == "active"


def test_active_correction_lifts_to_sites_and_is_player_zero_sum():
    pm = _two_player_map()
    cfg = _config(h_burnin_steps=3)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1"])
    g_hist = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    for t, g in enumerate(g_hist, start=1):
        reg.adjust_credit(g, np.array([0.1, 0.2]), 1.0, 2.0 * np.eye(2), t)
    gref = np.array([reg.g_ref["s0"], reg.g_ref["s1"]])
    g = np.array([4.0, 6.0])
    out = reg.adjust_credit(g, np.array([0.1, 0.2]), 1.0,
                            2.0 * np.eye(2), 4)
    u = np.asarray(reg.last["u_H"], dtype=float)
    assert np.isclose(float(u.sum()), 0.0, atol=1e-12)
    assert np.allclose(out, g + u * gref)
    assert np.isclose(float(reg.last["c"].sum() + u.sum()),
                      float(reg.last["c"].sum()))
    assert reg.last["stage"] == "active"
    assert np.all(np.isfinite(out))


def test_h_zero_burnin_initializes_on_first_step_then_corrects():
    pm = _two_player_map()
    cfg = _config(h_burnin_steps=0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1"])
    g0 = np.array([1.0, -2.0])
    out0 = reg.adjust_credit(g0, np.array([0.1, 0.2]), 0.0,
                             np.zeros((2, 2)), 1)
    assert np.array_equal(out0, g0)
    assert np.all(reg.last["u_H"] == 0.0)
    assert reg.last["stage"] == "active"
    assert reg.g_ref["s0"] == 1.0 and reg.g_ref["s1"] == 2.0
    assert reg.global_variance.V_A0 == 1.0
    assert reg.global_variance.V_F0 == 1.0
    assert np.allclose(reg.mu, [1.0, -1.0])

    out1 = reg.adjust_credit(np.array([1.1, -1.9]),
                             np.array([0.1, 0.2]), 1.0,
                             np.zeros((2, 2)), 2)
    assert reg.last["stage"] == "active"
    assert np.any(np.abs(reg.last["u_H"]) > 0.0)
    assert np.all(np.isfinite(out1))


def test_credit_standardization_prevents_tight_site_dominance():
    pm = _two_player_map()
    cfg = _config(h_burnin_steps=0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1"])
    reg.adjust_credit(np.array([1.0, 100.0]), np.array([0.1, 0.1]), 0.0,
                      np.zeros((2, 2)), 1)
    reg.adjust_credit(np.array([2.0, 100.0]), np.array([0.1, 0.1]), 1.0,
                      np.zeros((2, 2)), 2)
    # c_i is each site's credit in its own burn-in unit, not the raw sites.
    assert np.allclose(reg.last["c"], [2.0, 1.0])


def test_scaling_and_common_offset_leave_standardized_credit_unchanged():
    def run(offset, scale):
        pm = _two_player_map()
        cfg = _config(h_burnin_steps=0)
        reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                                   site_names=["s0", "s1"])
        sig = np.array([0.1, 0.2])
        reg.step_players(np.array([1.0, -2.0]) * scale + offset, sig,
                         1.0, np.eye(2), 1)
        reg.step_players(np.array([2.0, -1.0]) * scale + offset, sig,
                         1.0, np.eye(2), 2)
        return reg.last["z"].copy(), reg.last["S_c_prev"], reg.S_c

    z0, s0, sc0 = run(0.0, 1.0)
    z1, s1, sc1 = run(5.0, 1.0)
    z2, s2, sc2 = run(0.0, 10.0)
    assert np.allclose(z0, z1, atol=1e-12)
    assert np.allclose(z0, z2, atol=1e-12)
    assert np.isclose(s1, s0)
    assert np.isclose(s2, 10.0 * s0)

    # S_c uses the previous step for z and updates only afterwards.
    alpha_S = 1.0 - np.exp(-0.01 / 2.0)
    c = np.array([2.0, -1.0])
    expected_S = (1.0 - alpha_S) * s0 + alpha_S * max(np.std(c), 1e-12)
    assert np.isclose(sc0, expected_S)


def test_sigma_ratio_variance_is_dimensionless_and_scale_invariant():
    def run(scale):
        pm = _two_player_map()
        cfg = _config(h_burnin_steps=0)
        reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                                   site_names=["s0", "s1"])
        sig = np.array([0.1, 0.2]) * scale
        reg.adjust_credit(np.array([1.0, -2.0]), sig, 0.0,
                          np.eye(2), 1)
        reg.adjust_credit(np.array([2.0, -1.0]), sig, 1.0,
                          np.eye(2), 2)
        return reg.last["v"].copy()

    v1 = run(1.0)
    v10 = run(10.0)
    assert np.allclose(v1, v10, atol=1e-12)
    assert np.all(v1 >= 1e-8) and np.all(v1 <= 1e6)


def test_forecast_mean_is_causal_and_updates_after_scoring():
    pm = _two_player_map()
    cfg = _config(h_burnin_steps=0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1"])
    reg.step_players(np.array([1.0, -2.0]), np.array([0.1, 0.2]),
                     1.0, np.eye(2), 1)
    mu_prev = reg.mu.copy()
    z_prev_scale = reg.S_c
    reg.step_players(np.array([2.0, -1.0]), np.array([0.1, 0.2]),
                     1.0, np.eye(2), 2)
    assert np.allclose(reg.last["mu_prev"], mu_prev)
    z = reg.last["z"]
    v = reg.last["v"]
    T = reg.last["T"]
    expected_ell = -0.5 * (np.log(2.0 * np.pi * v)
                           + (z - mu_prev) ** 2 / v)
    assert np.allclose(reg.last["ell"], expected_ell)
    expected_z = (reg.last["c"] - reg.last["c"].mean()) / z_prev_scale
    assert np.allclose(z, expected_z)

    alpha_m = 1.0 - np.exp(-0.01 / 2.0)
    expected_mu = (1.0 - alpha_m) * mu_prev + alpha_m * z
    assert np.allclose(reg.mu, expected_mu)
    assert np.isfinite(T)


def test_softmax_shares_error_and_T_clipping():
    cfg = _config(h_burnin_steps=0)
    gvr = GlobalVarianceRegulator(cfg, dt_s=0.01)
    gvr.initialize(V_A0=1e-12, V_F0=1e-12, mean_A0=0.0)
    gvr.V_A0 = 1.0
    gvr.V_F0 = 1.0
    gvr.update(A=1e12, V_F=1e12)
    assert cfg.T_min <= gvr.T <= cfg.T_max
    ell = np.array([100.0, 0.0]); z = np.array([0.0, 0.0])
    a = np.exp(ell / gvr.T - np.max(ell / gvr.T))
    a /= a.sum()
    pi = np.ones(2) / 2.0
    e = a - pi
    assert np.isclose(a.sum(), 1.0) and np.isclose(pi.sum(), 1.0)
    assert np.isclose(e.sum(), 0.0)


def test_gaussian_log_score_proper_and_overdispersion_penalized():
    z = np.array([1.0, -1.0])
    mu = np.zeros(2)
    res2 = (z - mu) ** 2
    best = gaussian_log_score(z, mu, res2)
    assert np.all(gaussian_log_score(z, mu, 4.0 * res2) < best)
    assert np.all(gaussian_log_score(z, mu, 0.25 * res2) < best)
    # At a fixed mean the optimum is the realized squared error.
    grid = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    scores = np.array([gaussian_log_score(z, mu, np.full(2, v)) for v in grid])
    assert np.allclose(grid[np.argmax(scores[:, 0])], 1.0)


def test_all_global_variance_modes_are_finite_and_band_stays_in_band():
    for mode in ("advantage", "budget", "fisher", "band"):
        cfg = _config(h_burnin_steps=0, global_variance_mode=mode)
        gvr = GlobalVarianceRegulator(cfg, dt_s=0.01)
        gvr.initialize(V_A0=1.0, V_F0=2.0, mean_A0=0.5)
        T = gvr.update(A=2.0, V_F=3.0)
        assert np.isfinite(T)
        if mode == "budget":
            assert T == 1.0
        if mode == "band":
            assert cfg.band_lo <= T <= cfg.band_hi
            gvr.R_ewma = 100.0
            assert cfg.band_lo <= gvr.update(A=2.0, V_F=3.0) <= cfg.band_hi


def test_global_variance_ewma_recursions_match_the_design():
    cfg = _config(global_variance_mode="advantage")
    gvr = GlobalVarianceRegulator(cfg, dt_s=0.01)
    gvr.initialize(V_A0=1.0, V_F0=2.0, mean_A0=0.0)
    beta_A = np.exp(-0.01 / 2.0)
    T = gvr.update(A=3.0, V_F=4.0)
    delta = 3.0
    mean_A = (1.0 - beta_A) * delta
    var_A = beta_A * (1.0 + (1.0 - beta_A) * delta ** 2)
    V_F = beta_A * 2.0 + (1.0 - beta_A) * 4.0
    assert np.isclose(gvr.mean_A, mean_A)
    assert np.isclose(gvr.V_obs, max(var_A, 0.0))
    assert np.isclose(gvr.V_F_ewma, V_F)
    assert np.isclose(T, np.clip(gvr.V_obs / 1.0, cfg.T_min, cfg.T_max))

    cfg_f = _config(global_variance_mode="fisher")
    gf = GlobalVarianceRegulator(cfg_f, dt_s=0.01)
    gf.initialize(V_A0=1.0, V_F0=2.0, mean_A0=0.0)
    Tf = gf.update(A=3.0, V_F=4.0)
    assert np.isclose(Tf, np.clip(gf.V_F_ewma / 2.0,
                                  cfg_f.T_min, cfg_f.T_max))


def test_null_game_gate_returns_zero_correction():
    pm = _two_player_map()
    cfg = _config(h_burnin_steps=0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1"])
    reg.step_players(np.array([1.0, 1.0]), np.array([0.1, 0.2]),
                     1.0, np.eye(2), 1)
    u = reg.step_players(np.array([1.0, 1.0]), np.array([0.1, 0.2]),
                         1.0, np.eye(2), 2)
    assert reg.last["gate"] is True
    assert np.all(u == 0.0)


# --- H diffusion -------------------------------------------------------------


def test_h_positive_sign_moves_projected_share_toward_target():
    pm = _two_player_map()
    e = np.array([0.3, -0.3])
    pi = np.array([0.4, 0.6])
    a = np.array([0.7, 0.3])
    u = 0.1 * 1.0 * (pm.H @ e)
    logits = np.log(pi) + u
    projected = np.exp(logits - np.max(logits))
    projected /= projected.sum()
    err_before = np.abs(a - pi).sum()
    err_after = np.abs(a - projected).sum()
    assert err_after < err_before
    opposite = np.exp(np.log(pi) - u - np.max(np.log(pi) - u))
    opposite /= opposite.sum()
    assert np.abs(a - opposite).sum() > err_before


def test_h_lift_is_adjoint_of_player_credit_aggregation():
    pm = PhysicalMap.from_specs(
        [PlayerSpec("A", "population", ("s0", "s1"), size=2, ptype="exc"),
         PlayerSpec("B", "population", ("s2",), size=2, ptype="inh")],
        [("A", "B")])
    cfg = _config(h_burnin_steps=0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1", "s2"])
    g0 = np.array([1.0, 2.0, 3.0])
    g1 = np.array([1.2, 2.4, 3.0])
    reg.adjust_credit(g0, np.array([0.1, 0.2, 0.3]), 0.0,
                      np.eye(3), 1)
    out = reg.adjust_credit(g1, np.array([0.1, 0.2, 0.3]), 1.0,
                            np.eye(3), 2)
    u = reg.last["u_H"]
    g_h = reg.last["g_H"]
    assert np.allclose(out - g1, g_h)
    lifted = pm.player_credit(g_h, reg.site_names, reg.g_ref)
    assert np.allclose(lifted, u)
    assert np.isclose(float(u.sum()), 0.0, atol=1e-12)
    assert np.allclose(reg.last["c"] + u,
                       pm.player_credit(g1, reg.site_names, reg.g_ref)
                       + u)


def test_lambda_zero_short_circuits_to_exact_old_credit():
    pm = _two_player_map()
    cfg = _config(h_burnin_steps=0, lambda_H=0.0)
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                               site_names=["s0", "s1"])
    reg.adjust_credit(np.array([1.0, -2.0]), np.array([0.1, 0.2]),
                      0.0, np.eye(2), 1)
    g = np.array([1.2, -1.8])
    out = reg.adjust_credit(g, np.array([0.1, 0.2]), 1.0,
                            np.eye(2), 2)
    assert np.array_equal(out, g)
    assert np.all(reg.last["u_H"] == 0.0)


def test_tau_H_controls_the_error_ewma_rate():
    def run(tau):
        pm = _two_player_map()
        cfg = _config(h_burnin_steps=0, tau_H_s=tau)
        reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.5,
                                   site_names=["s0", "s1"])
        reg.step_players(np.array([1.0, -2.0]), np.array([0.1, 0.2]),
                         1.0, np.eye(2), 1)
        reg.step_players(np.array([2.0, -1.0]), np.array([0.1, 0.2]),
                         1.0, np.eye(2), 2)
        return reg.last["e"].copy(), reg.last["e_bar"].copy()

    e_fast, eb_fast = run(1.0)
    e_slow, eb_slow = run(5.0)
    alpha_fast = 1.0 - np.exp(-0.01 / 1.0)
    alpha_slow = 1.0 - np.exp(-0.01 / 5.0)
    assert np.allclose(eb_fast, alpha_fast * e_fast)
    assert np.allclose(eb_slow, alpha_slow * e_slow)
    assert np.linalg.norm(eb_slow) < np.linalg.norm(eb_fast)


def test_disconnected_players_are_coupled_by_the_eps_regularizer():
    pm = PhysicalMap.from_specs(
        [PlayerSpec("a", "population", ("a_x",), size=1),
         PlayerSpec("b", "population", ("b_x",), size=1)])
    assert pm.adjacency.sum() == 0.0
    H = pm.H
    assert H[0, 1] < 0.0 and H[1, 0] < 0.0
    assert np.isclose(H[0, 1], -pm.eps_H / 2.0)
    assert np.allclose(H @ np.ones(2), 0.0)

