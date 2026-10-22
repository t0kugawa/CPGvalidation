"""Structural CPG parameters are frozen, not deleted.

PERSONA's homeostatic update drove the Renshaw recurrent inhibition and the M-F
Izhikevich parameters away from reference until the flexor pool went silent.
These parameters stay in the network, but their applied value is clamped; the
optimizer therefore no longer maximises them.
"""
import pytest

from persona.registry import Registry
from t2xcpg.leg import Leg, M_F_IZH_REF

RENSHAW = {
    ("m_e", "r_e"): 3.0,
    ("r_e", "m_e"): -4.0,
    ("r_e", "ia_e"): -2.4,
    ("m_f", "r_f"): 3.0,
    ("r_f", "m_f"): -4.0,
    ("r_f", "ia_f"): -2.4,
}


def _leg():
    return Leg("L", Registry(), n_rg=8, n_mn=8, n_int=6, n_aff=6)


def test_renshaw_weights_are_frozen(nest_kernel):
    leg = _leg()
    for (src, tgt), value in RENSHAW.items():
        p = leg.registry.connection_cfgs[f"{src}_L_{tgt}_L"].parameters["weight"]
        assert p.min == pytest.approx(value)
        assert p.max == pytest.approx(value)
        assert p.sample_exploration() == pytest.approx(value)
        p.update_mean(100.0)
        assert p.mean == pytest.approx(value)


def test_mf_izh_is_frozen_at_reference(nest_kernel):
    leg = _leg()
    for name, value in M_F_IZH_REF.items():
        p = leg.registry.population_cfgs["m_f_L"].params[name]
        assert p.min == pytest.approx(value)
        assert p.max == pytest.approx(value)
        assert p.sample_exploration() == pytest.approx(value)
        p.update_mean(100.0)
        assert p.mean == pytest.approx(value)


def test_freezes_can_be_disabled_for_ab(nest_kernel, monkeypatch):
    monkeypatch.setenv("CPG_RENSHAW_CLAMP", "0")
    monkeypatch.setenv("CPG_MF_IZH_FREEZE", "0")
    leg = _leg()
    p = leg.registry.connection_cfgs["r_f_L_m_f_L"].parameters["weight"]
    assert p.min < p.max
    p = leg.registry.population_cfgs["m_f_L"].params["d"]
    assert p.min < p.max
