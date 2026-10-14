import numpy as np
import pytest

from persona.registry import Registry
from persona.agent import Agent
from persona.personalization import Personalizer, RunLog, MocapTargetDriver
from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.leg import Leg as Rig


def build_small_personalizer(nest_kernel, max_steps=40, loss_stop=0.05,
                             eps_pose=10.0, pose_hold_steps=3):
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=8, bs_e_hz=30.0, bs_f_hz=30.0)
    leg_l = CPG_Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    leg_r = CPG_Leg("R", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": leg_l, "R": leg_r}, dt_ms=10.0)
    rig = Rig(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    agent = Agent(registry=reg, window_size=30)
    n = 400
    t = np.arange(n) * 10.0
    driver = MocapTargetDriver(
        ankle_L=0.2 + 0.05 * np.sin(t / 100.0),
        ankle_R=0.2 - 0.05 * np.sin(t / 100.0),
        dt_ms=10.0,
    )
    return Personalizer(registry=reg, cpg=cpg, rig=rig, agent=agent,
                        target_driver=driver, dt_ms=10.0,
                        eps_pose=eps_pose, pose_hold_steps=pose_hold_steps,
                        loss_stop=loss_stop, loss_window=5, max_steps=max_steps)


def test_stage1_pose_then_stage2_track(nest_kernel):
    log = build_small_personalizer(nest_kernel, max_steps=40).run()
    assert log.stage[0] == "pose"
    assert "track" in log.stage
    assert len(log.loss) == len(log.time_ms)
    assert np.isfinite(log.loss).all()


def test_early_stop_triggers(nest_kernel):
    log = build_small_personalizer(nest_kernel, loss_stop=1e9, max_steps=40).run()
    assert sum(1 for s in log.stage if s == "track") < 40