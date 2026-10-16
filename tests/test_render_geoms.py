import mujoco


def _tendon_width(model, name):
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    return float(model.tendon_width[tid])


def test_muscle_tendon_width_thin():
    m = mujoco.MjModel.from_xml_path("dual_rat_hindlimb.xml")
    for name in ("sol_tendon_m_L", "ta_tendon_m_L", "sol_tendon_m_R", "ta_tendon_m_R"):
        assert _tendon_width(m, name) <= 0.0012, f"{name} too thick"


def test_muscle_sites_offset_off_axis():
    m = mujoco.MjModel.from_xml_path("dual_rat_hindlimb.xml")
    # Origin/insert sites must sit outside the bone capsule (radius ~0.006) so
    # the tendon is visible when it shortens.
    for name in ("sol_origin_m_L", "ta_origin_m_L", "sol_insert_m_L", "ta_insert_m_L",
                 "sol_origin_m_R", "ta_origin_m_R", "sol_insert_m_R", "ta_insert_m_R"):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        assert abs(float(m.site_pos[sid, 1])) >= 0.0065, f"{name} not offset off-axis"