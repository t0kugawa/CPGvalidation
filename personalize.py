import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import nest

from persona.registry import Registry
from persona.agent import Agent
from persona.game import GameConfig
from persona.homeostasis import HomeostaticConfig
from persona.personalization import Personalizer, MocapTargetDriver
from persona.validation import ValidationReport
from t2xcpg.bs import BrainStem, BaseDrive
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.leg import Leg as MuJoCo_Leg
from loader.mocap import MocapLoader, treadmill_surfacevel
from loader.stats import StatsLoader, LocomotionMetrics
from model.model_cfg import ANKLE_RANGE_MARGIN_RAD, TREADMILL_PENETRATION_M
from persona.parameter import reset_rng_stream

DT_MS = 10.0

DEFAULT_REFERENCE = "data/zenodo/DatasetMotionCaputure/RatA/RatA_x1p50_Lfast/RatA_x1p50_Lfast_T1.csv"
DEFAULT_STATS = "data/zenodo/DatasetStatisticalAnalysis/StatisticalDataTable.csv"


def acceptance_from_log(log,
                        stats=None,
                        margin: float = 0.05,
                        coverage: float = 0.70) -> dict:
    """
    Biological acceptance using BOTH ankles and the Stage-2 ("track") window
    only. angle_coverage is the fraction of tracked steps where both ankles
    stay within `margin`.
    """
    from persona.validation import biological_acceptance

    stage = np.asarray(log.stage)
    track = stage == "track"
    if not track.any():
        track = np.ones(len(stage), dtype=bool)
    err_L = np.abs(np.asarray(log.ankle_model_L) - np.asarray(log.ankle_target_L))
    err_R = np.abs(np.asarray(log.ankle_model_R) - np.asarray(log.ankle_target_R))
    err = np.maximum(err_L, err_R)[track]
    return biological_acceptance({}, stats, err,
                                 phase_mask=np.ones(err.shape, dtype=bool),
                                 margin=margin, coverage=coverage)


def build_personalizer(reference_path: str = DEFAULT_REFERENCE,
                       n_rg: int = 20, n_mn: int = 20, n_int: int = 15,
                       n_aff: int = 20, n_bs: int = 30,
                       max_steps: int = 1000,
                       eps_pose: float = 0.08,
                       pose_hold_steps: int = 20,
                       loss_stop: float = 0.05,
                       loss_window: int = 30,
                       seed: int = 0,
                       w_scale: float = 1.0,
                       ref_hz: float = 100.0,
                       belt_mode: str = "dataset",
                       loop: bool = True,
                       tau_mem_s: float = 2.0,
                       resolution_ms: float = 0.2,
                       cut_multisite: bool = False,
                       homeostasis: HomeostaticConfig | None = None,
                       game: GameConfig | None = None
                       ) -> tuple[Personalizer, MocapLoader]:
    """
    Builds the full closed loop: bilateral CPG (via the PERSONA registry),
    bilateral MuJoCo rig, PERSONA agent, and the mocap-driven target limb.

    `cut_multisite` opts into one CUT Poisson relay and STDP projection per
    plantar site (Option B). It changes the PERSONA parameter dimension, so it
    defaults off until the policy is updated for the per-site keys.
    """
    if resolution_ms <= 0:
        raise ValueError(f"resolution_ms must be > 0, got {resolution_ms}")
    np.random.seed(seed)
    reset_rng_stream(seed)
    nest.ResetKernel()
    # 0.2 ms is the `cpg_2legs_fast` default. At 1.0 ms the izhikevich network
    # locks into a resolution-induced limit cycle (~500 Hz motor pools at zero
    # sensory input), which pins the flexor drive at 1.0 regardless of PERSONA.
    nest.SetKernelStatus({"resolution": float(resolution_ms),
                          "print_time": False,
                          "rng_seed": 1 + int(seed) % (2**32 - 2)})
    try:
        nest.verbosity = nest.VerbosityLevel.ERROR
    except Exception:
        pass

    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=n_bs, bs_e_hz=40.0, bs_f_hz=40.0)
    base = BaseDrive(registry=reg, n_bs=n_bs, base_hz=2.0)
    leg_l = CPG_Leg("L", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff,
                    w_scale=w_scale, ref_hz=ref_hz,
                    cut_multisite=cut_multisite)
    leg_r = CPG_Leg("R", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff,
                    w_scale=w_scale, ref_hz=ref_hz,
                    cut_multisite=cut_multisite)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": leg_l, "R": leg_r}, dt_ms=DT_MS,
                leading_leg="L", leading_gain=1.5, base=base)

    rig = MuJoCo_Leg(mujoco_cfg=os.path.join(BASE_DIR, "dual_rat_hindlimb.xml"),
                     dt_ms=DT_MS)

    loader = MocapLoader(reference_path, dt_ms=DT_MS)
    slow, fast = treadmill_surfacevel(loader.meta)
    belt = fast if loader.meta.condition != "T" else slow
    rig.set_treadmill_surfacevel(belt)
    angles = loader.resample()
    n_guard = min(int(max_steps), len(angles["hip_L"]))
    ankle_lo = min(float(np.min(angles[f"ankle_{s}"][:n_guard])) for s in ("L", "R")) - ANKLE_RANGE_MARGIN_RAD
    ankle_hi = max(float(np.max(angles[f"ankle_{s}"][:n_guard])) for s in ("L", "R")) + ANKLE_RANGE_MARGIN_RAD

    for s in ("L", "R"):
        thigh = float(np.mean(angles[f"seg_thigh_{s}"])) / 1000.0
        shank = float(np.mean(angles[f"seg_shank_{s}"])) / 1000.0
        foot = float(np.mean(angles[f"seg_foot_{s}"])) / 1000.0
        rig.set_limb_lengths("m", s, thigh, shank, foot)
        rig.set_limb_lengths("ref", s, thigh, shank, foot)
    # Physiological ankle stop: with the still-weak muscle drive the free ankle
    # would otherwise collapse onto the wide [-1.0, 2.6] range (V19/V21).
    rig.set_ankle_range(ankle_lo, ankle_hi)

    # Approach A: the target limb is driven straight from the dataset points —
    # absolute segment orientations (-> rig hinge angles, already in range) and a
    # mocap hip base tracking the GT marker. No range rescaling / zero-alignment.
    hip_x = {s: (angles[f"hip_x_{s}"] - np.mean(angles[f"hip_x_{s}"])) / 1000.0
             for s in ("L", "R")}
    hip_z = {s: angles[f"hip_z_{s}"] / 1000.0 for s in ("L", "R")}
    tgt = {
        "ankle_L": np.asarray(angles["ankle_L"], dtype=np.float64),
        "ankle_R": np.asarray(angles["ankle_R"], dtype=np.float64),
        "hip_L": np.asarray(angles["hip_L"], dtype=np.float64),
        "knee_L": np.asarray(angles["knee_L"], dtype=np.float64),
        "hip_R": np.asarray(angles["hip_R"], dtype=np.float64),
        "knee_R": np.asarray(angles["knee_R"], dtype=np.float64),
    }

    # The belt is one plane shared by both limbs. Two placements are supported:
    #   "dataset" — the dataset's ground (min foot-marker height over the trial);
    #   "model"   — the model-derived lowest surface of the reconstructed target
    #               foot (5th percentile over the trajectory), which keeps the
    #               reset pose reachable but leaves little controlled contact.
    if belt_mode == "model":
        belt_z = rig.target_foot_lowest_percentile(
            hip_x={s: hip_x[s] for s in ("L", "R")},
            hip_z={s: hip_z[s] for s in ("L", "R")},
            hip={s: angles[f"hip_{s}"] for s in ("L", "R")},
            knee={s: angles[f"knee_{s}"] for s in ("L", "R")},
            ankle={s: angles[f"ankle_{s}"] for s in ("L", "R")},
            percentile=5.0)
    else:
        n_win = len(angles["hip_L"]) if loop else min(int(max_steps), len(angles["hip_L"]))
        belt_z = min(float(np.min(angles["mh_z_L"][:n_win])),
                     float(np.min(angles["mh_z_R"][:n_win]))) / 1000.0
    rig.set_treadmill_height(belt_z + TREADMILL_PENETRATION_M)

    # Start both limbs at the target's first pose so the ankle angles match at
    # t=0 (the target is then reachable and moving), with the hips at the GT.
    rig.reset(pose={
        "L": {"hip": float(tgt["hip_L"][0]), "knee": float(tgt["knee_L"][0]),
              "ankle": float(tgt["ankle_L"][0])},
        "R": {"hip": float(tgt["hip_R"][0]), "knee": float(tgt["knee_R"][0]),
              "ankle": float(tgt["ankle_R"][0])},
    }, hip_pos={"L": (float(hip_x["L"][0]), float(hip_z["L"][0])),
                "R": (float(hip_x["R"][0]), float(hip_z["R"][0]))})
    driver = MocapTargetDriver(
        ankle_L=tgt["ankle_L"], ankle_R=tgt["ankle_R"],
        hip_L=tgt["hip_L"], knee_L=tgt["knee_L"],
        hip_R=tgt["hip_R"], knee_R=tgt["knee_R"], dt_ms=DT_MS,
        hip_x_L=hip_x["L"], hip_z_L=hip_z["L"],
        hip_x_R=hip_x["R"], hip_z_R=hip_z["R"],
        loop=loop,
    )

    agent = Agent(registry=reg, lr_mu=0.03, lr_sigma=0.005,
                  tau_mem_s=tau_mem_s, dt_ms=DT_MS,
                  homeostasis=homeostasis, game=game)

    personalizer = Personalizer(
        registry=reg, cpg=cpg, rig=rig, agent=agent, target_driver=driver,
        dt_ms=DT_MS, eps_pose=eps_pose, pose_hold_steps=pose_hold_steps,
        loss_stop=loss_stop, loss_window=loss_window, max_steps=max_steps,
    )
    return personalizer, loader


def run_personalization(reference_path: str = DEFAULT_REFERENCE,
                        stats_path: str = DEFAULT_STATS,
                        output_dir: str | None = None,
                        n_steps: int = 1000,
                        small: bool = False,
                        eps_pose: float = 0.08,
                        pose_hold_steps: int = 20,
                        loss_stop: float = 0.05,
                        loss_window: int = 30,
                        verbose: bool = True,
                        log_every: int = 20,
                        belt_mode: str = "dataset",
                        tau_mem_s: float = 2.0,
                        realize_every_k: int = 1,
                        cut_multisite: bool = False) -> dict:
    """
    Full experiment: build the loop, run the two-stage PERSONA optimization,
    write the validation figure, the CSV/npz logs, and return the acceptance dict.
    """
    output_dir = output_dir or BASE_DIR
    os.makedirs(output_dir, exist_ok=True)
    sizes = dict(n_rg=8, n_mn=8, n_int=6, n_aff=6, n_bs=8) if small \
        else dict(n_rg=20, n_mn=20, n_int=15, n_aff=20, n_bs=30)
    if small:
        pose_hold_steps = min(pose_hold_steps, 3)
        loss_window = min(loss_window, 5)

    personalizer, loader = build_personalizer(
        reference_path=reference_path, max_steps=n_steps,
        eps_pose=eps_pose, pose_hold_steps=pose_hold_steps,
        loss_stop=loss_stop, loss_window=loss_window, belt_mode=belt_mode,
        tau_mem_s=tau_mem_s, cut_multisite=cut_multisite, **sizes)
    personalizer.verbose = verbose
    personalizer.log_every = log_every

    log = personalizer.run(realize_every_k=realize_every_k)
    log.to_csv(os.path.join(output_dir, "run_log.csv"))
    n_pose = sum(1 for s in log.stage if s == "pose")
    n_track = sum(1 for s in log.stage if s == "track")

    stats = StatsLoader(stats_path) if os.path.exists(stats_path) else None

    contact_L = np.asarray([1.0 if p == "stance" else 0.0 for p in log.phase_L])
    contact_R = np.asarray([1.0 if p == "stance" else 0.0 for p in log.phase_R])
    foot_pos_L = np.column_stack([np.asarray(log.foot_x_m_L), np.zeros(len(log.foot_x_m_L)),
                                  np.asarray(log.foot_z_m_L)])
    foot_pos_R = np.column_stack([np.asarray(log.foot_x_m_R), np.zeros(len(log.foot_x_m_R)),
                                  np.asarray(log.foot_z_m_R)])
    metrics = LocomotionMetrics.from_run(
        np.asarray(log.ankle_model_L), np.asarray(log.ankle_model_R),
        contact_L, contact_R, DT_MS, foot_pos_L=foot_pos_L, foot_pos_R=foot_pos_R)

    acceptance = acceptance_from_log(log, stats, margin=0.05, coverage=0.70)

    report = ValidationReport()
    report.plot_tracking(log)
    report.plot_learning(log)
    report.save_all(output_dir)
    _save_combined_figure(log, output_dir)
    np.savez(os.path.join(output_dir, "run_log.npz"),
             time_ms=log.time_ms, ankle_model_L=log.ankle_model_L,
             ankle_model_R=log.ankle_model_R, ankle_target_L=log.ankle_target_L,
             ankle_target_R=log.ankle_target_R, loss=log.loss,
             advantage=log.advantage, mus_E_L=log.mus_E_L, mus_F_L=log.mus_F_L,
             cut_L=log.cut_L, ia_rate_E_L=log.ia_rate_E_L, ib_rate_E_L=log.ib_rate_E_L)

    return {
        "acceptable": acceptance["acceptable"],
        "duty_factor_in_distribution": acceptance["duty_factor_in_distribution"],
        "angle_coverage": acceptance["angle_coverage"],
        "final_loss": float(np.mean(log.loss[-20:])) if len(log.loss) else float("nan"),
        "n_steps_done": int(len(log.time_ms)),
        "n_pose": n_pose,
        "n_track": n_track,
        "metrics": metrics,
        "output_dir": output_dir,
    }


def _save_combined_figure(log, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    t = np.asarray(log.time_ms) / 1000.0
    axes[0].plot(t, log.ankle_target_L, "k--", label="target L")
    axes[0].plot(t, log.ankle_model_L, "r-", label="model L")
    axes[0].plot(t, log.ankle_target_R, "b--", label="target R")
    axes[0].plot(t, log.ankle_model_R, "g-", label="model R")
    axes[0].set_ylabel("ankle [rad]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, log.loss, color="tab:brown")
    axes[1].set_ylabel("loss")
    axes[1].grid(alpha=0.3)
    axes[2].plot(t, log.advantage, color="tab:blue")
    axes[2].axhline(0.0, color="k", ls=":", alpha=0.6)
    axes[2].set_ylabel("advantage")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "validation_results.png"), dpi=150)
    plt.close(fig)


def main(n_steps: int = 200, small: bool = False, output_dir: str | None = None,
         verbose: bool = True, log_every: int = 20,
         tau_mem_s: float = 2.0, realize_every_k: int = 1,
         cut_multisite: bool = False) -> dict:
    res = run_personalization(n_steps=n_steps, small=small, output_dir=output_dir,
                              verbose=verbose, log_every=log_every,
                              tau_mem_s=tau_mem_s, realize_every_k=realize_every_k,
                              cut_multisite=cut_multisite)
    print(f"Steps: {res['n_steps_done']} (pose {res['n_pose']}, track {res['n_track']})")
    print(f"Final loss (last-20 mean): {res['final_loss']:.4f}")
    print(f"Acceptable: {res['acceptable']} | angle coverage: {res['angle_coverage']:.2f}")
    print(f"Output: {res['output_dir']}")
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the PERSONA personalization experiment.")
    ap.add_argument("--tau-mem-s", type=float, default=2.0,
                    help="trace/credit memory time constant in seconds "
                         "(beta = 1 - dt/tau_mem)")
    ap.add_argument("--realize-every-k", type=int, default=1,
                    help="sample + realize + record the network every k steps "
                         "(strategy still updates every dt)")
    ap.add_argument("--cut-multisite", action="store_true",
                    help="Option B: one CUT Poisson relay and STDP projection "
                         "per plantar site (changes the PERSONA dimension)")
    args = ap.parse_args()
    main(tau_mem_s=args.tau_mem_s, realize_every_k=args.realize_every_k,
         cut_multisite=args.cut_multisite)