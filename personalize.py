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
from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.leg import Leg as MuJoCo_Leg
from loader.mocap import MocapLoader, treadmill_surfacevel

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


if __name__ == "__main__":
    personalizer, loader = build_personalizer(max_steps=200)
    print(f"Trial: {loader.meta}")
    print("Starting two-stage personalization...")
    log = personalizer.run()
    n_pose = sum(1 for s in log.stage if s == "pose")
    n_track = sum(1 for s in log.stage if s == "track")
    print(f"Steps: {len(log.time_ms)} (pose {n_pose}, track {n_track})")
    print(f"Final loss: {float(log.loss[-1]):.4f} | mean last-20: {float(np.mean(log.loss[-20:])):.4f}")