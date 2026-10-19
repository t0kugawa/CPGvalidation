import numpy as np

from persona.registry import Registry
from persona.agent import Agent
from persona.personalization import Personalizer, MocapTargetDriver
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
    agent = Agent(registry=reg)
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


def test_wall_clock_budget_stops_run(nest_kernel):
    log = build_small_personalizer(nest_kernel, max_steps=40).run(wall_clock_s=0.0)
    assert len(log.time_ms) <= 2


def test_pose_stage_is_capped(nest_kernel):
    pers = build_small_personalizer(nest_kernel, max_steps=20, eps_pose=1e-9,
                                    pose_hold_steps=999)
    log = pers.run(pose_max_steps=5)
    assert log.stage[0] == "pose"
    assert log.stage[5] == "track"


def test_driver_loops_trajectory():
    looped = MocapTargetDriver(ankle_L=[0.1, 0.2, 0.3], ankle_R=[0.1, 0.2, 0.3],
                               dt_ms=10.0, loop=True)
    assert np.isclose(looped.target(0.0)["target_ankle_L"], 0.1)
    assert np.isclose(looped.target(30.0)["target_ankle_L"], 0.1)
    assert np.isclose(looped.target(40.0)["target_ankle_L"], 0.2)

    clamped = MocapTargetDriver(ankle_L=[0.1, 0.2, 0.3], ankle_R=[0.1, 0.2, 0.3],
                                dt_ms=10.0, loop=False)
    assert np.isclose(clamped.target(100.0)["target_ankle_L"], 0.3)


def test_realize_every_k_holds_the_sample(nest_kernel):
    pers = build_small_personalizer(nest_kernel, max_steps=12, loss_stop=-1e9)
    key = "pop_rg_e_L_a"
    seen = []

    def hook(step, log, cpg_in, cpg_out, stage):
        rec = pers.registry.site_recorder(key)
        seen.append((step, rec.latest_sample(), rec.stats()["t"]))

    log = pers.run(realize_every_k=2, step_hook=hook)
    # strategy steps still run every dt (downstream predicates use the loss)
    assert len(log.loss) == len(log.time_ms)
    assert len(seen) >= 6
    by_step = {step: (sample_sigma, t) for step, sample_sigma, t in seen}
    for step, sample_sigma, t in seen:
        if step % 2 == 0:
            # fresh realization + record at this step's t_ms (kernel clock)
            assert t == step * 10.0
        else:
            # the odd step holds the even step's recorded (x, sigma, t) exactly;
            # even when a fresh draw is clipped to the same bound, no write happens
            assert (sample_sigma, t) == by_step[step - 1]


def test_t_ms_reaches_the_recorders(nest_kernel):
    pers = build_small_personalizer(nest_kernel, max_steps=6, loss_stop=-1e9)
    log = pers.run()
    n = len(log.time_ms)
    assert n >= 4
    rec = pers.registry.site_recorder("pop_rg_e_L_a")
    # the last apply landed at the loop's t_ms, i.e. (n-1)*dt on the CPG clock
    assert rec.stats()["t"] == (n - 1) * 10.0


def test_apply_errors_logged_once(nest_kernel, capsys):
    pers = build_small_personalizer(nest_kernel, max_steps=5, loss_stop=-1e9)

    class _Boom:
        def set(self, *args, **kwargs):
            raise RuntimeError("boom")

    pers.registry.population_cfgs["rg_e_L"].node_collection = _Boom()
    pers.run()
    out = capsys.readouterr().out
    assert out.count("[apply-error] pop_rg_e_L") == 1