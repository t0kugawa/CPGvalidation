"""Fast test for passive ankle tone.

The controlled ankle must not park on the lower stop when the muscles are
silent. A small ligament-like spring/damper should pull it away from the stop.
"""
from __future__ import annotations

import os

import mujoco

from model.adapter import CPGOutput
from model.leg import Leg

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")


def test_ankle_joints_have_passive_tone():
    import mujoco
    from model.leg import Leg

    leg = Leg(mujoco_cfg=XML, dt_ms=10.0)
    for side in "LR":
        jid = leg.j_ankle_m[side]
        assert leg.model.jnt_stiffness[jid] > 0.0, side


def test_ankle_spring_lifts_from_the_lower_stop():
    leg = Leg(mujoco_cfg=XML, dt_ms=10.0)
    leg.reset()  # default stance pose and seated treadmill
    for side in "LR":
        adr = leg.model.jnt_qposadr[leg.j_ankle_m[side]]
        leg.data.qpos[adr] = 0.19
    mujoco.mj_forward(leg.model, leg.data)

    out = CPGOutput()  # zero muscle drive
    for _ in range(50):
        leg.step(out, target_ankle_L=1.0, target_ankle_R=1.0)

    q = {
        s: float(leg.data.qpos[leg.model.jnt_qposadr[leg.j_ankle_m[s]]])
        for s in "LR"
    }
    assert q["L"] > 0.22, q
    assert q["R"] > 0.22, q
