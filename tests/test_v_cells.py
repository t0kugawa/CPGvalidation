"""Commissural V-cell relay, ported from the NEURON model.

`memcpg/leg.py` (ipsilateral) and `memcpg/utils_cpg.py::add_external_connections`
(commissural) replace the direct RG<->RG commissural inhibition that
`cpg_2legs_fast` abstracts as "V0v/V0d/V2a/In1 classes":

    RG_F -> V2a -> V0v -> (contralateral) In1 -| RG_F
    RG_F -> V0d -| (contralateral) RG_F
    RG_F -> V3F -> (contralateral) RG_F

Nothing else changes: the ipsilateral half-centre, motor output, Ia, Renshaw,
Ib and CUT stay as they are.
"""
import numpy as np

from persona.registry import Registry
from t2xcpg.bs import BaseDrive, BrainStem
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg

V_CELLS = ("v2a", "v0v", "v0d", "v3f", "in1")


def _net():
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=8, bs_e_hz=60.0, bs_f_hz=60.0)
    base = BaseDrive(registry=reg, n_bs=8, base_hz=0.0)
    legs = {s: Leg(s, reg, n_rg=8, n_mn=8, n_int=6, n_aff=6) for s in "LR"}
    L2CPG(registry=reg, bs=bs, legs=legs, dt_ms=10.0, base=base)
    return reg


def test_v_cell_populations(nest_kernel):
    reg = _net()
    for leg in "LR":
        for name in V_CELLS:
            assert f"{name}_{leg}" in reg.population_cfgs


def test_v_cell_ipsilateral_weights(nest_kernel):
    reg = _net()
    w = lambda k: reg.connection_cfgs[k].parameters["weight"].mean
    for leg in "LR":
        assert np.isclose(w(f"rg_f_{leg}_v2a_{leg}"), 0.75)
        assert np.isclose(w(f"rg_f_{leg}_v0d_{leg}"), 0.75)
        assert np.isclose(w(f"rg_f_{leg}_v3f_{leg}"), 1.5)
        assert np.isclose(w(f"v2a_{leg}_v0v_{leg}"), 1.2)
        assert np.isclose(w(f"in1_{leg}_rg_f_{leg}"), -0.5)


def test_v_cell_commissural_weights(nest_kernel):
    reg = _net()
    w = lambda k: reg.connection_cfgs[k].parameters["weight"].mean
    assert np.isclose(w("v0v_L_in1_R"), 1.3)
    assert np.isclose(w("v0v_R_in1_L"), 1.3)
    assert np.isclose(w("v0d_L_rg_f_R"), -1.3)
    assert np.isclose(w("v0d_R_rg_f_L"), -1.3)
    # NEURON WALK sets v3f_cross_weight=0.0 and RUN 0.5; the scheme draws the
    # projection, so the enabled (RUN) value is the port's seed.
    assert np.isclose(w("v3f_L_rg_f_R"), 0.5)
    assert np.isclose(w("v3f_R_rg_f_L"), 0.5)


def test_no_direct_commissural_rg(nest_kernel):
    reg = _net()
    for key in ("rg_f_L_rg_f_R", "rg_f_R_rg_f_L",
                "rg_e_L_rg_e_R", "rg_e_R_rg_e_L"):
        assert key not in reg.connection_cfgs


def test_v_cell_neuron_types(nest_kernel):
    reg = _net()
    for leg in "LR":
        for name in ("v2a", "v0v", "v3f"):
            assert reg.population_cfgs[f"{name}_{leg}"].params["a"].mean == 0.02
            assert reg.population_cfgs[f"{name}_{leg}"].params["d"].mean == 8.0
        for name in ("v0d", "in1"):
            assert reg.population_cfgs[f"{name}_{leg}"].params["a"].mean == 0.1
            assert reg.population_cfgs[f"{name}_{leg}"].params["d"].mean == 2.0
