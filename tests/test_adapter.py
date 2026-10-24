from model.adapter import CPGOutput
from model.leg import Leg


def test_both_legs_sensory_populated():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    out = CPGOutput(mus_E_L=0.5, mus_F_L=0.1, mus_E_R=0.1, mus_F_R=0.5)
    ci = leg.step(out, target_ankle_L=0.3, target_ankle_R=-0.1)
    for f in ["cut_L", "cut_R", "ia_rate_E_L", "ia_rate_E_R",
              "ib_force_E_L", "ib_force_E_R"]:
        assert getattr(ci, f) is not None


def test_non_ankle_joints_track_target():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    leg.set_treadmill_height(-1.0)  # keep the target foot out of ground contact
    out = CPGOutput()
    for _ in range(50):
        ci = leg.step(out, target_ankle_L=0.2, target_ankle_R=0.2)
    assert abs(ci.ankle_angle_ref_L - 0.2) < 0.05


def test_sides_driven_independently():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    leg.set_treadmill_height(-1.0)  # keep the target foot out of ground contact
    out = CPGOutput()
    for _ in range(60):
        ci = leg.step(out, target_ankle_L=0.4, target_ankle_R=0.3)
    assert abs(ci.ankle_angle_ref_L - 0.4) < 0.05
    assert abs(ci.ankle_angle_ref_R - 0.3) < 0.05
    assert abs(ci.ankle_angle_ref_L - ci.ankle_angle_ref_R) > 0.05