import numpy as np

from persona.registry import Registry
from t2xcpg.bs import BaseDrive, BrainStem
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg


def _leg():
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    return reg, leg


def test_halfcentre_is_via_in(nest_kernel):
    reg, leg = _leg()
    assert "rg_e_L_in_e_L" in reg.connection_cfgs
    assert "in_e_L_rg_f_L" in reg.connection_cfgs
    assert "rg_e_L_rg_f_L" not in reg.connection_cfgs      # no direct half-centre
    assert "in_e_L_m_e_L" not in reg.connection_cfgs       # InE does not drive MN


def test_rg_to_mn_direct(nest_kernel):
    reg, leg = _leg()
    assert "rg_e_L_m_e_L" in reg.connection_cfgs


def test_reference_values(nest_kernel):
    reg, leg = _leg()
    w = lambda k: reg.connection_cfgs[k].parameters["weight"].mean
    assert np.isclose(w("rg_e_L_in_e_L"), 12.0)
    assert np.isclose(w("in_e_L_rg_f_L"), -8.0)
    assert np.isclose(w("rg_f_L_in_f_L"), 18.0)
    assert np.isclose(w("in_f_L_rg_e_L"), -48.0)


def test_interneurons_are_fast_spiking(nest_kernel):
    reg, leg = _leg()
    for name in ("in_e_L", "in_f_L"):
        params = reg.population_cfgs[name].params
        assert params["a"].mean == 0.1
        assert params["d"].mean == 2.0
    # the change must not leak into the rhythm generators
    assert reg.population_cfgs["rg_e_L"].params["a"].mean == 0.02
    assert reg.population_cfgs["rg_f_L"].params["c"].mean == -55.0


def test_weak_ef_pathway_density(nest_kernel):
    reg, leg = _leg()
    assert reg.connection_cfgs["in_e_L_rg_f_L"].p == 0.15
    assert reg.connection_cfgs["in_f_L_rg_e_L"].p == 0.30


def test_renshaw_gain_is_attenuated(nest_kernel):
    reg, leg = _leg()
    w = lambda k: reg.connection_cfgs[k].parameters["weight"].mean
    assert np.isclose(w("m_e_L_r_e_L"), 3.0)
    assert np.isclose(w("r_e_L_m_e_L"), -4.0)
    assert np.isclose(w("r_e_L_ia_e_L"), -2.4)
    assert np.isclose(w("m_f_L_r_f_L"), 3.0)
    assert np.isclose(w("r_f_L_m_f_L"), -4.0)
    assert np.isclose(w("r_f_L_ia_f_L"), -2.4)


def test_bs_drive_seed_weight(nest_kernel):
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=8, bs_e_hz=60.0, bs_f_hz=60.0)
    base = BaseDrive(registry=reg, n_bs=8, base_hz=0.0)
    legs = {s: Leg(s, reg, n_rg=8, n_mn=8, n_int=6, n_aff=6) for s in "LR"}
    cpg = L2CPG(registry=reg, bs=bs, legs=legs, dt_ms=10.0, base=base)

    bs_conns = {k: c for k, c in reg.connection_cfgs.items()
                if c.source.startswith("BS_in_")}
    assert bs_conns
    for key, cfg in bs_conns.items():
        expected = 6.0 if key.endswith("_L") else 4.0
        assert np.isclose(cfg.parameters["weight"].mean, expected), key