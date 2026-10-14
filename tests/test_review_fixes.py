import numpy as np

from persona.parameter import PersonaParam
from persona.strategy import FisherMatrix
from persona.registry import Registry
from persona.personalization import RunLog
from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.adapter import CPGOutput
from model.leg import Leg


# --- Critical 3: per-parameter independent RNG streams ---

def test_params_have_independent_rng():
    a = PersonaParam(mean=0.0, sigma_ctrl=1.0)
    b = PersonaParam(mean=0.0, sigma_ctrl=1.0)
    da = np.array([a.sample_exploration() for _ in range(50)])
    db = np.array([b.sample_exploration() for _ in range(50)])
    assert not np.allclose(da, db)
    assert abs(np.corrcoef(da, db)[0, 1]) < 0.9


# --- Important: standardized natural-gradient step maps back losslessly ---

def _hist_fisher(scales, seed=1, n=40):
    names = [f"p{i}" for i in range(len(scales))]
    fm = FisherMatrix(param_names=names, window_size=50)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        z = rng.standard_normal(len(scales))
        pert = {names[i]: float(z[i] * scales[i]) for i in range(len(scales))}
        fm.record_step(pert, np.array([0.0]), "stance")
    return fm


def test_step_is_scale_lossless():
    fm1 = _hist_fisher([1.0, 1.0])
    d1 = fm1.compute_natural_gradient_step(
        {"p0": 1.0, "p1": 1.0}, {"p0": 0.0, "p1": 0.0},
        {"p0": 1.0, "p1": 1.0}, np.array([1.0]))
    fm2 = _hist_fisher([10.0, 1.0])
    d2 = fm2.compute_natural_gradient_step(
        {"p0": 1.0, "p1": 1.0}, {"p0": 0.0, "p1": 0.0},
        {"p0": 1.0, "p1": 1.0}, np.array([1.0]))
    s1 = fm1.feature_scales()
    s2 = fm2.feature_scales()
    # The standardized step delta/scale is identical regardless of unit scale.
    assert np.isclose(d2["p0"] / s2[0], d1["p0"] / s1[0], atol=1e-9)
    assert np.isclose(d2["p1"] / s2[1], d1["p1"] / s1[1], atol=1e-9)


# --- Critical 2: STDP weight deltas reach the noise recorder during a step ---

def test_stdp_weight_deltas_recorded_in_step(nest_kernel):
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=6, bs_e_hz=20.0, bs_f_hz=20.0)
    ll = CPG_Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    lr = CPG_Leg("R", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": ll, "R": lr}, dt_ms=10.0)
    cpg.step()
    deltas = ll.noise_recorder.get_latest_step_deltas()
    assert any(k.endswith("_weight") for k in deltas)


# --- Important: ia_rg STDP seed must not exceed its plastic ceiling ---

def test_ia_rg_seed_within_wmax(nest_kernel):
    reg = Registry()
    leg = CPG_Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    for k in leg.plastic_projections:
        if k.startswith("ia_rg"):
            w = reg.connection_cfgs[k].parameters["weight"].mean
            assert w <= 10.0, f"{k} seed {w} exceeds Wmax"


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


# --- Important: target fitted into the ankle joint range ---

def test_fit_to_joint_range():
    from personalize import fit_to_joint_range
    a = np.linspace(-2.5, -0.5, 100)
    b = fit_to_joint_range(a)
    assert b.min() >= -0.8 - 1e-9 and b.max() <= 1.4 + 1e-9
    assert np.corrcoef(a, b)[0, 1] > 0.99


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