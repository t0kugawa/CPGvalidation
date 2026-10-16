import numpy as np

from persona.registry import Registry
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