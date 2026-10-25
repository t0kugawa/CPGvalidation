"""Fast foot-orientation test for the ankle stops.

The foot's local -z is the sole/down direction. At the upper ankle stop it must
not point upward. This is the cheap, deterministic detector for the
"ankle stuck upside down" bug; it does not need NEST, PERSONA or a render.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from model.model_cfg import ANKLE_RANGE_RAD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")


def _sole_world_z(q: float, side: str) -> float:
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    for s, qq in (("L", q), ("R", q)):
        for joint, val in ((f"ankle_m_{s}", qq), (f"hip_m_{s}", 0.0),
                           (f"knee_m_{s}", 0.0)):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
            d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot_m_{side}")
    rot = d.xmat[bid].reshape(3, 3)
    return float(-rot[2, 2])


def test_ankle_stops_never_put_the_sole_up():
    lo, hi = ANKLE_RANGE_RAD
    for side in "LR":
        z_lo = _sole_world_z(lo, side)
        z_hi = _sole_world_z(hi, side)
        # Tolerate a nearly vertical sole, not an inverted one.
        assert z_lo <= 0.05, (side, "lower", lo, z_lo)
        assert z_hi <= 0.05, (side, "upper", hi, z_hi)


def test_ankle_limits_match_the_xml():
    m = mujoco.MjModel.from_xml_path(XML)
    lo, hi = ANKLE_RANGE_RAD
    for side in "LR":
        jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"ankle_m_{side}")
        assert np.isclose(m.jnt_range[jid, 0], lo)
        assert np.isclose(m.jnt_range[jid, 1], hi)
