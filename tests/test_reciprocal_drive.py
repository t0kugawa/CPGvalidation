"""Fast test for the phase-gated reciprocal E/F drive."""
from __future__ import annotations

from model.adapter import CPGOutput
from model.leg import Leg, reciprocal_drives


def test_reciprocal_drives_never_co_contract():
    assert reciprocal_drives(1.0, 1.0, "stance") == (1.0, 0.0)
    assert reciprocal_drives(1.0, 1.0, "swing") == (0.0, 1.0)
    assert reciprocal_drives(0.5, 0.5, "stance") == (0.5, 0.0)
    assert reciprocal_drives(0.5, 0.5, "swing") == (0.0, 0.5)


def test_contract_muscles_uses_the_contact_phase():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    leg.phase_clock["L"].stance = True
    leg.phase_clock["R"].stance = False
    leg.contract_muscles(CPGOutput(mus_E_L=1.0, mus_F_L=1.0,
                                   mus_E_R=1.0, mus_F_R=1.0))
    assert leg.data.ctrl[leg.act_sol_m["L"]] == 1.0
    assert leg.data.ctrl[leg.act_ta_m["L"]] == 0.0
    assert leg.data.ctrl[leg.act_sol_m["R"]] == 0.0
    assert leg.data.ctrl[leg.act_ta_m["R"]] == 1.0
