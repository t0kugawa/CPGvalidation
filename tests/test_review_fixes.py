import numpy as np

from persona.parameter import PersonaParam
from persona.registry import Registry
from persona.personalization import RunLog
from t2xcpg.bs import BrainStem, BaseDrive
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.adapter import CPGOutput
from model.leg import Leg


# --- Important: the single-sigma rule must not diverge ---

def test_sigma_stays_bounded():
    p = PersonaParam(mean=0.0, sigma=1e-3)
    for _ in range(200):
        x = p.sample_exploration()
        p.update_sigma(sample=x, mu_at_sample=p.mean, sigma_at_sample=p.sigma,
                       advantage=0.5, lr_sigma=0.005)
    assert 1e-4 <= p.sigma < 0.05


# --- Critical 3: per-parameter independent RNG streams ---

def test_params_have_independent_rng():
    a = PersonaParam(mean=0.0, sigma=1.0)
    b = PersonaParam(mean=0.0, sigma=1.0)
    da = np.array([a.sample_exploration() for _ in range(50)])
    db = np.array([b.sample_exploration() for _ in range(50)])
    assert not np.allclose(da, db)
    assert abs(np.corrcoef(da, db)[0, 1]) < 0.9


# --- Important: natural-gradient step maps back losslessly ---
# (Superseded: unit-covariance of the persisted natural gradient is covered by
#  tests/test_strategy.py::test_step_is_unit_covariant.)


# --- Critical 2: STDP weight deltas reach the noise recorder during a step ---

def test_synaptic_parameters_reach_the_right_projection(nest_kernel):
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=6, bs_e_hz=20.0, bs_f_hz=20.0)
    ll = CPG_Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    lr = CPG_Leg("R", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    L2CPG(registry=reg, bs=bs, legs={"L": ll, "R": lr}, dt_ms=10.0)
    params = reg.get_all_optimizable_params()
    explored = {n: p.sample_exploration() for n, p in params.items()}
    reg.apply_explored_parameters(explored)
    for key, cfg in reg.connection_cfgs.items():
        col = reg.synapses(key)
        w = np.asarray(col.get("weight"), dtype=float)
        if cfg.model == "static_synapse":
            want = float(explored[f"conn_{key}_weight"])
            sigma = cfg.parameters["weight"].sigma
            # per-synapse realization around the explored sample (spec §5.2)
            assert abs(w.mean() - want) < 5.0 * sigma / np.sqrt(w.size) + 1e-9, \
                f"{key}: explored weight not realized around the sample"
            if cfg.parameters["weight"].min < cfg.parameters["weight"].max:
                # Frozen structural weights have min=max and are intentionally constant.
                assert w.std() > 0.0, f"{key}: no per-synapse dispersion"
            # recorded = applied: the recorder holds exactly the applied sample
            assert reg.site_recorder(f"conn_{key}_weight").latest_sample()[0] == want
        else:
            seed = round(float(cfg.extra_syn_spec["weight"]), 6)
            got = {round(float(x), 6) for x in w}
            assert got == {seed}, f"{key}: STDP weight must not be overwritten"


def test_base_drive_projections(nest_kernel):
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=5, bs_e_hz=40.0, bs_f_hz=40.0)
    base = BaseDrive(registry=reg, n_bs=5, base_hz=2.0)
    ll = CPG_Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    lr = CPG_Leg("R", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    L2CPG(registry=reg, bs=bs, legs={"L": ll, "R": lr}, base=base)
    keys = {k for k in reg.connection_cfgs if k.startswith("BASE_in_")}
    assert keys == {"BASE_in_rg_e_L", "BASE_in_rg_f_L",
                    "BASE_in_rg_e_R", "BASE_in_rg_f_R"}
    assert "pop_BASE_pg_rate" in reg.get_all_optimizable_params()


def test_stdp_weight_deltas_recorded_in_step(nest_kernel):
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=6, bs_e_hz=20.0, bs_f_hz=20.0)
    ll = CPG_Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    lr = CPG_Leg("R", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": ll, "R": lr}, dt_ms=10.0)
    cpg.step()
    stdp_recs = [n for n in reg.producer_recorders() if n.startswith("producer:stdp:")]
    assert stdp_recs, "every STDP projection must have a producer recorder"
    for name in stdp_recs:
        d = reg.noise_recorders[name].take_delta()
        assert d is not None and np.isfinite(d)     # fresh each step (0.0 if no updates)


# --- Important: plastic (STDP) seed weights must lie within their bounds ---

def test_plastic_seed_within_bounds(nest_kernel):
    reg = Registry()
    leg = CPG_Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    assert leg.plastic_projections, "expected at least one plastic projection"
    for k in leg.plastic_projections:
        cfg = reg.connection_cfgs[k]
        # STDP owns the weight; PERSONA optimizes only the hyper-parameters.
        assert "weight" not in cfg.parameters, f"{k} must not expose weight"
        assert "Wmax" in cfg.parameters, f"{k} must expose Wmax"
        seed = float(cfg.extra_syn_spec["weight"])
        assert 0.0 <= seed <= cfg.parameters["Wmax"].mean, f"{k} seed {seed} > Wmax"


# --- Important: treadmill surface velocity settable ---

def test_treadmill_surfacevel_settable():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    leg.set_treadmill_surfacevel(0.2)
    gid = leg._gid("treadmill")
    assert np.isclose(leg.model.geom_surfacevel[gid, 0], -0.2)


# --- Important: hip/knee servos follow the supplied target ---

def test_knee_servo_tracks_target():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    out = CPGOutput()
    for _ in range(60):
        ci = leg.step(out, target_ankle_L=0.3, target_ankle_R=0.3,
                      target_knee_L=0.9, target_knee_R=0.9)
    assert abs(ci.knee_angle_m_L - 0.9) < 0.1


# --- Important: acceptance uses both legs and the Stage-2 (track) window ---

def test_acceptance_uses_track_and_both_legs():
    from personalize import acceptance_from_log
    log = RunLog()
    log.stage = ["pose"] * 10 + ["track"] * 10
    log.ankle_model_L = [0.0] * 20
    log.ankle_target_L = [0.0] * 20
    log.ankle_model_R = [0.0] * 20
    log.ankle_target_R = [5.0] * 20  # R permanently wrong
    res = acceptance_from_log(log, stats=None, margin=0.05, coverage=0.70)
    assert res["angle_coverage"] == 0.0
    assert res["acceptable"] is False


def test_acceptance_ignores_pose_stage():
    from personalize import acceptance_from_log
    log = RunLog()
    log.stage = ["pose"] * 10 + ["track"] * 10
    log.ankle_model_L = [0.0] * 20
    log.ankle_target_L = [0.0] * 20
    log.ankle_model_R = [0.0] * 20
    log.ankle_target_R = [0.0] * 20
    log.ankle_target_R[0] = 5.0  # bad only during pose
    log.ankle_model_R[0] = 0.0
    res = acceptance_from_log(log, stats=None, margin=0.05, coverage=0.70)
    assert res["angle_coverage"] == 1.0


# --- Critical 1: entry point calls run_personalization with n_steps ---

def test_main_passes_n_steps(monkeypatch):
    import personalize
    captured = {}

    def fake(**kw):
        captured.update(kw)
        return {"acceptable": True, "n_steps_done": 0, "n_pose": 0, "n_track": 0,
                "final_loss": 0.0, "angle_coverage": 0.0, "output_dir": "."}

    monkeypatch.setattr(personalize, "run_personalization", fake)
    personalize.main(n_steps=7)
    assert captured.get("n_steps") == 7
    assert "max_steps" not in captured


# --- Target limb is a kinematic mirror: no collision, geometric phase ---

def _bid(model, name):
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_target_phase_ref_uses_actual_contact():
    """The target phase is the actual foot/treadmill contact, not a height epsilon."""
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    leg.model.opt.gravity[:] = 0.0
    pose = {s: {"hip": -0.2, "knee": 0.6, "ankle": 0.3} for s in ("L", "R")}
    targets = {
        "target_ankle_L": 0.3, "target_ankle_R": 0.3,
        "target_hip_L": -0.2, "target_knee_L": 0.6,
        "target_hip_R": -0.2, "target_knee_R": 0.6,
    }
    out = CPGOutput()

    leg.reset(pose=pose)
    low = leg._foot_lowest_z(leg.ref_foot_geoms["L"])

    leg.set_treadmill_height(low - 0.0015)
    leg.reset(pose=pose)
    ci = leg.step(out, **targets)
    assert ci.phase_ref_L == "swing"
    assert np.isclose(ci.contact_ref_L, 0.0)

    leg.set_treadmill_height(low + 0.0015)
    leg.reset(pose=pose)
    ci = leg.step(out, **targets)
    assert ci.phase_ref_L == "stance"
    assert np.isclose(ci.contact_ref_L, 1.0)


def test_target_has_contact_force():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    assert hasattr(leg, "_contact_force_ref")
    assert leg._contact_force_ref("L") >= 0.0


def test_joint_ranges_cover_loader_clamp():
    import mujoco
    from loader.mocap import JOINT_RANGES

    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    for s in ("L", "R"):
        for j, jid in (("hip", leg.j_hip_ref[s]), ("knee", leg.j_knee_ref[s]),
                       ("ankle", leg.j_ankle_ref[s]),
                       ("hip", leg.j_hip_m[s]), ("knee", leg.j_knee_m[s]),
                       ("ankle", leg.j_ankle_m[s])):
            lo, hi = (float(x) for x in leg.model.jnt_range[jid])
            assert lo <= JOINT_RANGES[j][0] + 1e-9, (s, j, lo)
            assert hi >= JOINT_RANGES[j][1] - 1e-9, (s, j, hi)


def test_noise_recorder_memory_is_constant_size():
    from persona.noise_recorder import NoiseRecorder

    nr = NoiseRecorder("producer:x")
    for i in range(10_000):
        nr.record_delta(float(i), 1.0)
    s = nr.stats()
    assert s["n"] == 10_000                      # statistics, not a growing sample list
    assert np.isclose(s["std"], 0.0) and np.isclose(s["latest"], 1.0)


def test_rg_recorders_are_read_and_cleared(nest_kernel):
    import nest

    reg = Registry()
    leg = CPG_Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    # A bare leg has no brainstem, so whether RG-E has fired within 50 ms is a
    # marginal stochastic transient (measured 0-11 spikes, depending on the
    # connection draw; 17-45 by 250 ms). This test is about the recorders being
    # read and cleared, not about excitability, so make the spiking precondition
    # deterministic rather than dependent on the RNG stream position.
    leg.rg_e.set({"I_e": 100.0})
    nest.Simulate(50.0)
    assert leg.rec_rge.get("events")["times"].size > 0, "RGE must have spiked"
    rg_e, rg_f = leg.get_rg_rates(10.0, 50.0)
    assert 0.0 <= rg_e <= 1.0 and 0.0 <= rg_f <= 1.0
    assert leg.rec_rge.get("events")["times"].size == 0
    assert leg.rec_rgf.get("events")["times"].size == 0


def test_target_foot_lowest_percentile():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    n = 20
    z = np.full(n, 0.05)
    a = np.linspace(0.0, 1.0, n)
    val = leg.target_foot_lowest_percentile(
        hip_x={"L": np.zeros(n), "R": np.zeros(n)},
        hip_z={"L": z, "R": z},
        hip={"L": a, "R": a}, knee={"L": a, "R": a}, ankle={"L": a, "R": a},
        percentile=5.0)
    assert np.isfinite(val)
    assert val < 0.05


def test_run_log_has_full_channel_set(nest_kernel):
    from tests.test_personalization import build_small_personalizer

    pers = build_small_personalizer(nest_kernel, max_steps=12)
    pers.stdp_every = 4
    log = pers.run()
    for name in ("cut_R", "ia_rate_E_R", "ia_len_F_R", "ib_force_F_L",
                 "ib_act_E_R", "adv_L", "adv_R", "hip_ref_R", "knee_model_R",
                 "rg_e_L", "rg_f_R"):
        assert hasattr(log, name), name
        assert len(getattr(log, name)) == len(log.time_ms), name
    assert log.stdp["delta"], "plastic projections must be logged"
    assert len(log.stdp["time_ms"]) == len(log.time_ms)
    assert log.stdp["sample_time_ms"].size
    for key, vals in log.stdp["weight_mean"].items():
        assert len(vals) == log.stdp["sample_time_ms"].size


def test_h5_round_trip_has_stdp_afferent_and_fisher(nest_kernel, tmp_path):
    from persona.h5io import save_run, load_run
    from tests.test_personalization import build_small_personalizer

    pers = build_small_personalizer(nest_kernel, max_steps=12)
    pers.stdp_every = 4
    log = pers.run()
    path = str(tmp_path / "run.h5")
    save_run(path, log, pers.agent, meta={"duration_s": 0.12})
    data, meta = load_run(path)
    assert data["log"]["cut_R"].shape == (len(log.time_ms),)
    assert set(log.stdp["delta"].keys()) <= set(data["stdp"]["delta"].keys())
    assert "final_weights" in data["stdp"]
    assert "fisher" in data["agent"]
    assert data["agent"]["means"]
    assert data["agent"]["sigmas"]
    assert float(meta["duration_s"]) == 0.12


# --- Output limb is scaled to the target's per-side segment lengths ---

def test_set_limb_lengths_matches_target():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    m = leg.model
    leg.set_limb_lengths("m", "L", thigh=0.030, shank=0.040, foot=0.025)
    assert np.isclose(np.linalg.norm(m.body_pos[_bid(m, "shank_m_L")]), 0.030)
    assert np.isclose(np.linalg.norm(m.body_pos[_bid(m, "foot_m_L")]), 0.040)
    assert np.isclose(np.linalg.norm(m.body_pos[_bid(m, "toe_m_L")]), 0.025)


def test_set_limb_lengths_preserves_rest_direction():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    m = leg.model
    before = m.body_pos[_bid(m, "shank_m_L")].copy()
    before /= np.linalg.norm(before)
    leg.set_limb_lengths("m", "L", thigh=0.030, shank=0.040, foot=0.025)
    after = m.body_pos[_bid(m, "shank_m_L")].copy()
    after /= np.linalg.norm(after)
    assert np.allclose(before, after)
