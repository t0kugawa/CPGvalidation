"""The target limb must obey the treadmill as a physical mirror.

The target limb used to be a non-colliding kinematic mirror: over RatA T1 its
toe went up to 26 mm below the belt (72-77% of frames) while the controlled
foot sat on the belt, and the geometric phase_ref reported ~80% stance. It now
carries the same foot/toe collision pair as the controlled limb and its phase
comes from the actual contacts.
"""
import os

import mujoco
import numpy as np
import pytest

from loader.mocap import MocapLoader
from model.adapter import CPGOutput
from model.leg import Leg
from model.model_cfg import TREADMILL_PENETRATION_M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")
REF = os.path.join(HERE, "data/zenodo/DatasetMotionCaputure/RatA",
                   "RatA_x1p50_Lfast/RatA_x1p50_Lfast_T1.csv")
DT_MS = 10.0


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _aid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def test_target_foot_geoms_collide_with_the_treadmill():
    m = mujoco.MjModel.from_xml_path(XML)
    for side in "LR":
        for name in (f"foot_ref_geom_{side}", f"toe_ref_geom_{side}"):
            gid = _gid(m, name)
            assert gid >= 0, name
            assert m.geom_contype[gid] == 2, name
            assert m.geom_conaffinity[gid] == 1, name


def test_target_toe_servo_matches_the_controlled_one():
    m = mujoco.MjModel.from_xml_path(XML)
    for side in "LR":
        ref = _aid(m, f"toe_ref_ctrl_{side}")
        controlled = _aid(m, f"toe_m_ctrl_{side}")
        assert ref >= 0 and controlled >= 0
        assert np.isclose(m.actuator_gainprm[ref, 0], m.actuator_gainprm[controlled, 0])
        assert np.isclose(m.actuator_gainprm[ref, 0], 2.0)


def test_phase_ref_uses_actual_contact_not_height_epsilon():
    """1.5 mm above the belt is inside the old 3 mm epsilon but is not contact."""
    leg = Leg(mujoco_cfg=XML, dt_ms=DT_MS)
    leg.model.opt.gravity[:] = 0.0
    pose = {s: {"hip": -0.2, "knee": 0.6, "ankle": 0.3} for s in "LR"}
    out = CPGOutput()
    targets = {
        "target_ankle_L": 0.3, "target_ankle_R": 0.3,
        "target_hip_L": -0.2, "target_knee_L": 0.6,
        "target_hip_R": -0.2, "target_knee_R": 0.6,
    }

    leg.reset(pose=pose)
    low = leg._foot_lowest_z(leg.ref_foot_geoms["L"])

    leg.set_treadmill_height(low - 0.0015)
    leg.reset(pose=pose)
    ci = leg.step(out, **targets)
    assert ci.phase_ref_L == "swing"
    assert ci.contact_ref_L == 0.0

    leg.set_treadmill_height(low + 0.0015)
    leg.reset(pose=pose)
    ci = leg.step(out, **targets)
    assert ci.phase_ref_L == "stance"
    assert ci.contact_ref_L == 1.0


@pytest.fixture(scope="module")
def trial():
    loader = MocapLoader(REF, dt_ms=DT_MS)
    return loader.resample()


def test_target_foot_stays_on_the_belt_over_the_trial(trial):
    """The target toe must not plunge 20-26 mm below the belt any more."""
    rig = Leg(mujoco_cfg=XML, dt_ms=DT_MS)
    for side in "LR":
        thigh = float(np.mean(trial[f"seg_thigh_{side}"])) / 1000.0
        shank = float(np.mean(trial[f"seg_shank_{side}"])) / 1000.0
        foot = float(np.mean(trial[f"seg_foot_{side}"])) / 1000.0
        rig.set_limb_lengths("m", side, thigh, shank, foot)
        rig.set_limb_lengths("ref", side, thigh, shank, foot)
    hip_x = {s: (trial[f"hip_x_{s}"] - np.mean(trial[f"hip_x_{s}"])) / 1000.0
             for s in "LR"}
    hip_z = {s: trial[f"hip_z_{s}"] / 1000.0 for s in "LR"}
    belt = min(float(np.min(trial["mh_z_L"])), float(np.min(trial["mh_z_R"]))) / 1000.0
    rig.set_treadmill_height(belt + TREADMILL_PENETRATION_M)
    rig.reset(pose={s: {"hip": float(trial[f"hip_{s}"][0]),
                        "knee": float(trial[f"knee_{s}"][0]),
                        "ankle": float(trial[f"ankle_{s}"][0])} for s in "LR"},
              hip_pos={s: (float(hip_x[s][0]), float(hip_z[s][0])) for s in "LR"})

    out = CPGOutput()
    lows, contacts = [], []
    n = min(200, len(trial["hip_L"]))
    for i in range(n):
        ci = rig.step(
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
        lows.append(min(rig._foot_lowest_z(rig.ref_foot_geoms[s]) for s in "LR") - rig.belt_z)
        contacts.append(max(ci.contact_ref_L, ci.contact_ref_R))
        for side in "LR":
            phase = getattr(ci, f"phase_ref_{side}")
            contact = getattr(ci, f"contact_ref_{side}")
            assert phase == ("stance" if contact else "swing")

    assert min(lows) > -0.010, f"target toe still {1000*min(lows):.2f} mm below the belt"
    assert np.mean(contacts) > 0.5, "target foot never contacts the treadmill"
