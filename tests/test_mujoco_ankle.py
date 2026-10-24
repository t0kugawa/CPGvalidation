"""MuJoCo ankle geometry regression tests.

The controlled ankle is muscle-driven. If the SOL moment arm flips sign in the
physiological range, both muscles pull the ankle the same way and no command
can hold the target. These tests use the model directly, with contacts disabled,
so they isolate the tendon geometry from the belt and the CPG.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

from model.model_cfg import ANKLE_RANGE_RAD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")


def _model():
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return m, mujoco.MjData(m)


def _id(m, obj, name):
    i = mujoco.mj_name2id(m, obj, name)
    assert i >= 0, name
    return i


def _set_pose(m, d, q):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = 0.0
    for side in "LR":
        jid = _id(m, mujoco.mjtObj.mjOBJ_JOINT, f"ankle_m_{side}")
        d.qpos[m.jnt_qposadr[jid]] = q
        for name in (f"hip_m_ctrl_{side}", f"knee_m_ctrl_{side}",
                     f"toe_m_ctrl_{side}", f"sol_m_{side}", f"ta_m_{side}"):
            d.ctrl[_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = 0.0
    mujoco.mj_forward(m, d)


def _length(m, d, name):
    return float(d.ten_length[_id(m, mujoco.mjtObj.mjOBJ_TENDON, name)])


def _moment_arm(m, d, tendon, q, eps=1e-4):
    _set_pose(m, d, q + eps)
    hi = _length(m, d, tendon)
    _set_pose(m, d, q - eps)
    lo = _length(m, d, tendon)
    return (hi - lo) / (2 * eps)


def test_ankle_limits_are_the_physical_range():
    m, _ = _model()
    lo, hi = ANKLE_RANGE_RAD
    for side in "LR":
        for name in (f"ankle_m_{side}", f"ankle_ref_{side}"):
            jid = _id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert np.isclose(m.jnt_range[jid, 0], lo), name
            assert np.isclose(m.jnt_range[jid, 1], hi), name
        aid = _id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"ankle_ref_ctrl_{side}")
        assert np.isclose(m.actuator_ctrlrange[aid, 0], lo)
        assert np.isclose(m.actuator_ctrlrange[aid, 1], hi)


@pytest.mark.parametrize("q", [0.3, 1.0, 1.4, 1.66, 1.9])
def test_sol_stays_plantarflexor(q):
    """SOL must shorten as the ankle plantarflexes (q increases)."""
    m, d = _model()
    for side in "LR":
        arm = _moment_arm(m, d, f"sol_tendon_m_{side}", q)
        assert arm < 0.0, (side, q, arm)


@pytest.mark.parametrize("q", [0.3, 1.0, 1.4, 1.66, 1.9])
def test_ta_stays_dorsiflexor(q):
    """TA must shorten as the ankle dorsiflexes (q decreases), so dL/dq > 0."""
    m, d = _model()
    for side in "LR":
        arm = _moment_arm(m, d, f"ta_tendon_m_{side}", q)
        assert arm > 0.0, (side, q, arm)


def test_full_sol_lifts_the_ankle_from_the_target_pose():
    m, d = _model()
    q0 = 1.66
    _set_pose(m, d, q0)
    for side in "LR":
        d.ctrl[_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"sol_m_{side}")] = 1.0
    for _ in range(50):  # 100 ms at the model timestep
        mujoco.mj_step(m, d)
    q = float(d.qpos[m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, "ankle_m_L")]])
    assert q > q0 + 0.01, q
