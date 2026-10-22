import numpy as np
import mujoco

XML = "dual_rat_hindlimb.xml"


def _load():
    m = mujoco.MjModel.from_xml_path(XML)
    return m, mujoco.MjData(m)


def _jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def _aid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)


def _sid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)


def test_four_limb_joints_and_actuators():
    m, _ = _load()
    for side in "LR":
        for j in ["hip_m", "knee_m", "ankle_m", "hip_ref", "knee_ref", "ankle_ref"]:
            assert _jid(m, f"{j}_{side}") >= 0, f"{j}_{side}"
        for a in ["sol_m", "ta_m"]:
            assert _aid(m, f"{a}_{side}") >= 0, f"{a}_{side}"


def test_lr_symmetry():
    m, d = _load()
    for j in ["hip_m", "knee_m", "ankle_m"]:
        loL, hiL = m.jnt_range[_jid(m, f"{j}_L")]
        loR, hiR = m.jnt_range[_jid(m, f"{j}_R")]
        assert np.allclose([loL, hiL], [loR, hiR])


def test_foot_contact_geoms_and_bodies():
    m, _ = _load()
    for side in "LR":
        # foot/toe geoms collide with the ground (contype=2)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot_m_geom_{side}")
        assert gid >= 0 and m.geom_contype[gid] == 2
        # CUT is computed from contacts on the foot/toe bodies
        for b in (f"foot_m_{side}", f"toe_m_{side}"):
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0


def test_treadmill_levels_on_the_controlled_foot():
    """The controlled foot is the one whose CUT feeds the CPG, so it must seat
    on the belt. The target limb is now a physical mirror and collides too."""
    from model.leg import Leg

    leg = Leg(mujoco_cfg=XML, dt_ms=10.0)
    for side in "LR":
        gid = mujoco.mj_name2id(leg.model, mujoco.mjtObj.mjOBJ_GEOM,
                                f"foot_ref_geom_{side}")
        assert leg.model.geom_contype[gid] == 2
        assert leg.model.geom_conaffinity[gid] == 1
    lo = min(leg._foot_lowest_z(leg.model_foot_geoms[s]) for s in "LR")
    assert lo <= leg.belt_z, "controlled foot must reach the belt"


def test_non_ankle_joint_servos_present():
    m, _ = _load()
    for side in "LR":
        for a in ["hip_m_ctrl", "knee_m_ctrl", "toe_m_ctrl",
                  "hip_ref_ctrl", "knee_ref_ctrl", "ankle_ref_ctrl", "toe_ref_ctrl"]:
            assert _aid(m, f"{a}_{side}") >= 0, f"{a}_{side}"