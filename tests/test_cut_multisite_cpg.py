"""Option B: one Poisson relay and one STDP projection per plantar site.

These tests exercise the CPG side only (no MuJoCo). Option B is opt-in so the
default single-aggregate CUT path keeps its existing PERSONA dimension.
"""
import numpy as np

from model.adapter import CPGInput
from persona.registry import Registry
from t2xcpg.bs import BaseDrive, BrainStem
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg

N_SITES = 8


def _leg(n_aff: int = 16, **kwargs) -> tuple[Registry, Leg]:
    reg = Registry()
    leg = Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=n_aff, **kwargs)
    return reg, leg


def test_multisite_builds_site_relays_and_stdp_keys(nest_kernel):
    reg, leg = _leg(cut_multisite=True)

    assert len(leg.cut_site_pg) == N_SITES
    assert len(leg.cut_site_in) == N_SITES
    for k in range(N_SITES):
        assert f"cut_pg_L_s{k}" in reg.population_nodes
        assert f"cut_in_L_s{k}" in reg.population_nodes
        assert f"cut_rge_L_s{k}" in reg.connection_cfgs
        assert f"cut_in_L_s{k}_in_e_L" in reg.connection_cfgs

    # The single aggregate projections are replaced, not duplicated.
    assert "cut_rge_L" not in reg.connection_cfgs
    assert "cut_in_L_in_e_L" not in reg.connection_cfgs
    # The aggregate Poisson relay survives only as the cut_L noise tap.
    assert len(reg.get("cut_pg_L")) == 1
    assert len(reg.get("cut_in_L")) == 1


def test_multisite_site_rates_are_zeroed_when_not_active(nest_kernel):
    reg, leg = _leg(cut_multisite=True)
    rates = tuple(float(10 + 5 * k) for k in range(N_SITES))
    active = (True, False, True, False, True, False, True, False)

    leg.update_sensory_feedback(
        cut_hz=42.0, ia_e_hz=0.0, ia_f_hz=0.0, ib_e_hz=0.0, ib_f_hz=0.0,
        cut_sites=rates, cut_active=active)

    for k, pg_key in enumerate(leg.cut_site_pg):
        expected = rates[k] if active[k] else 0.0
        got = np.asarray(reg.get(pg_key).get("rate"), dtype=float)
        assert np.allclose(got, expected)

    # The aggregate relay still receives the back-compatible scalar for the
    # cut_L producer recorder, even though it drives no synapses in Option B.
    assert np.isclose(reg.get("cut_pg_L").get("rate"), 42.0)


def test_multisite_empty_per_site_input_starts_silent(nest_kernel):
    reg, leg = _leg(cut_multisite=True)
    # Personalizer.run() starts from CPGInput() before the first MuJoCo step.
    leg.update_sensory_feedback(
        cut_hz=0.0, ia_e_hz=0.0, ia_f_hz=0.0, ib_e_hz=0.0, ib_f_hz=0.0,
        cut_sites=(), cut_active=())
    for pg_key in leg.cut_site_pg:
        assert np.allclose(reg.get(pg_key).get("rate"), 0.0)


def test_multisite_update_requires_one_gate_per_site(nest_kernel):
    _, leg = _leg(cut_multisite=True)
    try:
        leg.update_sensory_feedback(
            cut_hz=1.0, ia_e_hz=0.0, ia_f_hz=0.0, ib_e_hz=0.0, ib_f_hz=0.0,
            cut_sites=(1.0,), cut_active=(True,))
    except ValueError as exc:
        assert "per-site" in str(exc)
    else:
        raise AssertionError("mismatched per-site CUT inputs must be rejected")


def test_multisite_cut_parameter_dimension(nest_kernel):
    reg, leg = _leg(cut_multisite=True)
    params = reg.get_all_optimizable_params()

    stdp = [k for k in params if k.startswith("conn_cut_rge_L_s")]
    static = [k for k in params
              if k.startswith("conn_cut_in_L_s") and k.endswith("_weight")]
    assert len(stdp) == N_SITES * 6          # six STDP hyper-parameters/site
    assert len(static) == N_SITES            # one static CUT->InE weight/site
    assert "conn_cut_rge_L_tau_plus" not in params

    prods = set(reg.producer_recorders())
    for k in range(N_SITES):
        assert f"producer:stdp:cut_rge_L_s{k}" in prods
    assert "producer:poisson:cut_L" in prods


def test_default_leg_keeps_single_aggregate_cut(nest_kernel):
    reg, leg = _leg(cut_multisite=False)
    assert leg.cut_site_pg == []
    assert "cut_rge_L" in reg.connection_cfgs
    assert "cut_in_L_in_e_L" in reg.connection_cfgs
    assert len(reg.get("cut_pg_L")) == 16


def test_l2cpg_routes_per_site_rates(nest_kernel):
    reg = Registry()
    bs = BrainStem(reg, n_bs=5, bs_e_hz=2.0, bs_f_hz=2.0)
    base = BaseDrive(reg, n_bs=5, base_hz=2.0)
    legs = {
        "L": Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=16,
                 cut_multisite=True),
        "R": Leg("R", reg, n_rg=6, n_mn=6, n_int=5, n_aff=16,
                 cut_multisite=True),
    }
    cpg = L2CPG(reg, bs, legs, dt_ms=1.0, base=base)

    rates_l = tuple(float(k) for k in range(N_SITES))
    rates_r = tuple(float(100 + k) for k in range(N_SITES))
    active = (True,) * N_SITES
    cpg.update(CPGInput(
        cut_L=20.0, cut_R=30.0,
        cut_sites_L=rates_l, cut_sites_R=rates_r,
        cut_active_L=active, cut_active_R=active))

    assert np.allclose(reg.get("cut_pg_L_s3").get("rate"), 3.0)
    assert np.allclose(reg.get("cut_pg_R_s7").get("rate"), 107.0)
    assert np.isclose(reg.get("cut_pg_L").get("rate"), 20.0)
    assert np.isclose(reg.get("cut_pg_R").get("rate"), 30.0)
