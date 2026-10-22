"""The ankle must not hyper-plantarflex past a physiological stop.

With a low muscle drive the controlled ankle collapsed onto the old wide range
([-1.0, 2.6] rad) and the soft limit let qpos reach +2.97 rad, so the paw
visually trailed behind the knee. The guard narrows both ankles to the trial's
mocap envelope plus a small margin and stiffens the joint limit.
"""
import os

import mujoco
import numpy as np
import pytest

from loader.mocap import MocapLoader
from model.adapter import CPGOutput
from model.leg import Leg
from model.model_cfg import ANKLE_RANGE_MARGIN_RAD, TREADMILL_PENETRATION_M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")
REF = os.path.join(HERE, "data/zenodo/DatasetMotionCaputure/RatA",
                   "RatA_x1p50_Lfast/RatA_x1p50_Lfast_T1.csv")
DT_MS = 10.0


def test_ankle_joint_limits_are_stiff():
    m = mujoco.MjModel.from_xml_path(XML)
    for side in "LR":
        for name in (f"ankle_m_{side}", f"ankle_ref_{side}"):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, name
            assert np.isclose(m.jnt_solref[jid, 0], 0.004), name
            assert m.jnt_solimp[jid, 0] >= 0.95, name
            assert m.jnt_solimp[jid, 1] >= 0.99, name


def test_set_ankle_range_applies_to_all_four_ankles():
    leg = Leg(mujoco_cfg=XML, dt_ms=DT_MS)
    leg.set_ankle_range(-0.3, 1.9)
    for side in "LR":
        for jid in (leg.j_ankle_m[side], leg.j_ankle_ref[side]):
            lo, hi = leg.model.jnt_range[jid]
            assert np.isclose(lo, -0.3)
            assert np.isclose(hi, 1.9)
        lo, hi = leg.model.actuator_ctrlrange[leg.act_ankle_ref[side]]
        assert np.isclose(lo, -0.3)
        assert np.isclose(hi, 1.9)


@pytest.fixture(scope="module")
def trial():
    return MocapLoader(REF, dt_ms=DT_MS).resample()


def _build_guarded_leg(trial):
    leg = Leg(mujoco_cfg=XML, dt_ms=DT_MS)
    for side in "LR":
        thigh = float(np.mean(trial[f"seg_thigh_{side}"])) / 1000.0
        shank = float(np.mean(trial[f"seg_shank_{side}"])) / 1000.0
        foot = float(np.mean(trial[f"seg_foot_{side}"])) / 1000.0
        leg.set_limb_lengths("m", side, thigh, shank, foot)
        leg.set_limb_lengths("ref", side, thigh, shank, foot)
    lo = min(float(np.min(trial[f"ankle_{s}"][:200])) for s in "LR") - ANKLE_RANGE_MARGIN_RAD
    hi = max(float(np.max(trial[f"ankle_{s}"][:200])) for s in "LR") + ANKLE_RANGE_MARGIN_RAD
    leg.set_ankle_range(lo, hi)
    hip_x = {s: (trial[f"hip_x_{s}"] - np.mean(trial[f"hip_x_{s}"])) / 1000.0
             for s in "LR"}
    hip_z = {s: trial[f"hip_z_{s}"] / 1000.0 for s in "LR"}
    belt = min(float(np.min(trial["mh_z_L"])), float(np.min(trial["mh_z_R"]))) / 1000.0
    leg.set_treadmill_height(belt + TREADMILL_PENETRATION_M)
    leg.reset(pose={s: {"hip": float(trial[f"hip_{s}"][0]),
                        "knee": float(trial[f"knee_{s}"][0]),
                        "ankle": float(trial[f"ankle_{s}"][0])} for s in "LR"},
              hip_pos={s: (float(hip_x[s][0]), float(hip_z[s][0])) for s in "LR"})
    return leg, lo, hi, hip_x, hip_z


def test_controlled_ankle_stays_inside_the_guard_under_contact(trial):
    leg, lo, hi, hip_x, hip_z = _build_guarded_leg(trial)
    out = CPGOutput()
    max_m = -np.inf
    max_ref = -np.inf
    for i in range(min(200, len(trial["hip_L"]))):
        ci = leg.step(
            out,
            target_ankle_L=float(trial["ankle_L"][i]),
            target_ankle_R=float(trial["ankle_R"][i]),
            target_hip_L=float(trial["hip_L"][i]),
            target_knee_L=float(trial["knee_L"][i]),
            target_hip_R=float(trial["hip_R"][i]),
            target_knee_R=float(trial["knee_R"][i]),
            target_hipx_L=float(hip_x["L"][i]), target_hipz_L=float(hip_z["L"][i]),
            target_hipx_R=float(hip_x["R"][i]), target_hipz_R=float(hip_z["R"][i]),
        )
        max_m = max(max_m, ci.ankle_angle_m_L, ci.ankle_angle_m_R)
        max_ref = max(max_ref, ci.ankle_angle_ref_L, ci.ankle_angle_ref_R)
    # 0.10 rad of soft-limit overshoot is still a physiological stop, not a flip.
    assert max_m <= hi + 0.10, f"controlled ankle {max_m:+.3f} > guard {hi:+.3f}"
    assert max_ref <= hi + 0.10, f"reference ankle {max_ref:+.3f} > guard {hi:+.3f}"
    assert max_m >= lo - 0.10 and max_ref >= lo - 0.10
