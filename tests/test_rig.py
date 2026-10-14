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


def test_touch_sensors_both_sides():
    m, _ = _load()
    for side in "LR":
        assert _sid(m, f"cut_heel_0_{side}") >= 0


def test_non_ankle_joint_servos_present():
    m, _ = _load()
    for side in "LR":
        for a in ["hip_m_ctrl", "knee_m_ctrl", "toe_m_ctrl",
                  "hip_ref_ctrl", "knee_ref_ctrl", "ankle_ref_ctrl", "toe_ref_ctrl"]:
            assert _aid(m, f"{a}_{side}") >= 0, f"{a}_{side}"