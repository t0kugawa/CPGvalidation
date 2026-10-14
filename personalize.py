import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import nest

from persona.registry import Registry
from persona.agent import Agent
from persona.personalization import Personalizer, MocapTargetDriver
from persona.validation import ValidationReport, biological_acceptance
from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.leg import Leg as MuJoCo_Leg
from loader.mocap import MocapLoader
from loader.stats import StatsLoader, LocomotionMetrics

DT_MS = 10.0
SIM_REST_ANKLE = 0.3

DEFAULT_REFERENCE = "data/zenodo/DatasetMotionCaputure/RatA/RatA_x1p50_Lfast/RatA_x1p50_Lfast_T1.csv"
DEFAULT_STATS = "data/zenodo/DatasetStatisticalAnalysis/StatisticalDataTable.csv"


def align_to_sim(ankle: np.ndarray, sim_rest: float = SIM_REST_ANKLE) -> np.ndarray:
    """
    Zero-aligns a mocap ankle trajectory to the MuJoCo rest pose: the trial's
    initial pose maps to the sim's rest ankle angle.
    """
    a = np.asarray(ankle, dtype=np.float64)
    return a - a[0] + sim_rest


def build_personalizer(reference_path: str = DEFAULT_REFERENCE,
                       n_rg: int = 20, n_mn: int = 20, n_int: int = 15,
                       n_aff: int = 20, n_bs: int = 30,
                       max_steps: int = 1000,
                       eps_pose: float = 0.08,
                       pose_hold_steps: int = 20,
                       loss_stop: float = 0.05,
                       loss_window: int = 30,
                       seed: int = 0) -> tuple[Personalizer, MocapLoader]:
    """
    Builds the full closed loop: bilateral CPG (via the PERSONA registry),
    bilateral MuJoCo rig, PERSONA agent, and the mocap-driven target limb.
    """
    np.random.seed(seed)
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})

    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=n_bs, bs_e_hz=50.0, bs_f_hz=50.0)
    leg_l = CPG_Leg("L", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff)
    leg_r = CPG_Leg("R", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": leg_l, "R": leg_r}, dt_ms=DT_MS)

    rig = MuJoCo_Leg(mujoco_cfg=os.path.join(BASE_DIR, "dual_rat_hindlimb.xml"),
                     dt_ms=DT_MS)

    loader = MocapLoader(reference_path, rest_pose=None, dt_ms=DT_MS)
    driver = MocapTargetDriver(
        ankle_L=align_to_sim(loader.ankle("L")),
        ankle_R=align_to_sim(loader.ankle("R")),
        dt_ms=DT_MS,
    )

    agent = Agent(registry=reg, lr_mu=0.03, lr_sigma_ctrl=0.005,
                  lr_sigma_param=0.001, window_size=max(30, loss_window))

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
                        loss_window: int = 30) -> dict:
    """
    Full experiment: build the loop, run the two-stage PERSONA optimization,
    write the validation figure and the run log, and return the acceptance dict.
    """
    output_dir = output_dir or BASE_DIR
    sizes = dict(n_rg=8, n_mn=8, n_int=6, n_aff=6, n_bs=8) if small \
        else dict(n_rg=20, n_mn=20, n_int=15, n_aff=20, n_bs=30)
    if small:
        pose_hold_steps = min(pose_hold_steps, 3)
        loss_window = min(loss_window, 5)

    personalizer, loader = build_personalizer(
        reference_path=reference_path, max_steps=n_steps,
        eps_pose=eps_pose, pose_hold_steps=pose_hold_steps,
        loss_stop=loss_stop, loss_window=loss_window, **sizes)

    log = personalizer.run()
    n_pose = sum(1 for s in log.stage if s == "pose")
    n_track = sum(1 for s in log.stage if s == "track")

    angle_errors = np.abs(np.asarray(log.ankle_model_L) - np.asarray(log.ankle_target_L))
    phase_mask = np.ones(len(angle_errors), dtype=bool)
    stats = StatsLoader(stats_path) if os.path.exists(stats_path) else None

    contact_L = np.asarray([1.0 if p == "stance" else 0.0 for p in log.phase_L])
    contact_R = np.asarray([1.0 if p == "stance" else 0.0 for p in log.phase_R])
    metrics = LocomotionMetrics.from_run(
        np.asarray(log.ankle_model_L), np.asarray(log.ankle_model_R),
        contact_L, contact_R, DT_MS)

    acceptance = biological_acceptance(
        metrics, stats, angle_errors, phase_mask, margin=0.05, coverage=0.70)

    report = ValidationReport()
    report.plot_tracking(log)
    report.plot_learning(log)
    report.save_all(output_dir)
    _save_combined_figure(log, output_dir)
    np.savez(os.path.join(output_dir, "run_log.npz"),
             time_ms=log.time_ms, ankle_model_L=log.ankle_model_L,
             ankle_model_R=log.ankle_model_R, ankle_target_L=log.ankle_target_L,
             ankle_target_R=log.ankle_target_R, loss=log.loss,
             advantage=log.advantage)

    return {
        "acceptable": acceptance["acceptable"],
        "global_in_distribution": acceptance["global_in_distribution"],
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


if __name__ == "__main__":
    res = run_personalization(max_steps=200)
    print(f"Steps: {res['n_steps_done']} (pose {res['n_pose']}, track {res['n_track']})")
    print(f"Final loss (last-20 mean): {res['final_loss']:.4f}")
    print(f"Acceptable: {res['acceptable']} | angle coverage: {res['angle_coverage']:.2f}")
    print(f"Output: {res['output_dir']}")