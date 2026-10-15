"""
Render the CPG-driven locomotion (startup + personalization) from three
camera views with a sensor HUD, and write GIFs.

Usage (headless):
    MUJOCO_GL=osmesa python render_locomotion.py --steps 400 --fps 40

Startup protocol:
  * t=0: both legs still, feet planted, CUT active, no muscle activity.
  * the brainstem fires at equal frequency (40 Hz) on both legs, but the
    leading (left) RG receives a larger synaptic weight so the first step
    emerges from the left leg.
  * the treadmill belt starts moving only after the brainstem drive begins.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import nest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import imageio

from persona.registry import Registry
from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.leg import Leg as MuJoCo_Leg
from model.adapter import CPGInput
from model.render import LegRenderer, hud_lines

DT_MS = 10.0
BS_HZ = 40.0
LEADING_LEG = "L"
LEADING_GAIN = 1.5
CUT_PLANTED_HZ = 120.0
TREADMILL_MPS = 0.2
BS_START_STEP = 0
BELT_START_STEP = 20


def build(n_rg, n_mn, n_int, n_aff, n_bs):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})
    try:
        nest.verbosity = nest.VerbosityLevel.ERROR
    except Exception:
        pass
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=n_bs, bs_e_hz=BS_HZ, bs_f_hz=BS_HZ)
    leg_l = CPG_Leg("L", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff,
                    w_scale=0.4, ref_hz=150.0)
    leg_r = CPG_Leg("R", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff,
                    w_scale=0.4, ref_hz=150.0)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": leg_l, "R": leg_r}, dt_ms=DT_MS,
                leading_leg=LEADING_LEG, leading_gain=LEADING_GAIN)
    rig = MuJoCo_Leg(mujoco_cfg=os.path.join(BASE_DIR, "dual_rat_hindlimb.xml"), dt_ms=DT_MS)
    rig.set_treadmill_surfacevel(0.0)  # belt off during startup
    return reg, cpg, rig


def run(duration_s, cams, width, height, fps, out_dir, fmt,
        n_rg, n_mn, n_int, n_aff, n_bs):
    reg, cpg, rig = build(n_rg, n_mn, n_int, n_aff, n_bs)
    # Startup (option 2a): start the controlled limbs from the target pose so
    # both ankle angles match at t=0; afferents are then derived from state.
    start_hip, start_knee, start_ankle = rig.get_reference_kinematics(0.0)
    rig.reset(pose={s: {"hip": start_hip, "knee": start_knee, "ankle": start_ankle}
                    for s in ("L", "R")})
    renderer = LegRenderer(rig.model, rig.data, width=width, height=height)

    n_frames = int(round(duration_s * fps))
    dt_s = DT_MS / 1000.0
    steps = int(round(duration_s / dt_s))
    # simulation step at which each output frame is captured (exact fps timing)
    frame_steps = {int(round(k / (fps * dt_s))) for k in range(n_frames)}

    ext = "mp4" if fmt == "mp4" else "gif"
    writers = {}
    for c in cams:
        path = os.path.join(out_dir, f"locomotion_{c}.{ext}")
        if fmt == "mp4":
            writers[c] = imageio.get_writer(path, fps=fps, codec="libx264",
                                            quality=8, macro_block_size=1)
        else:
            writers[c] = imageio.get_writer(path, mode="I", fps=fps)

    cpg_in = CPGInput()
    belt_on = False
    for step in range(steps):
        if not belt_on and step >= BELT_START_STEP:
            rig.set_treadmill_surfacevel(TREADMILL_MPS)
            belt_on = True
        if step == 0 and cpg_in.cut_L == 0.0 and cpg_in.cut_R == 0.0:
            cpg_in.cut_L = CUT_PLANTED_HZ
            cpg_in.cut_R = CUT_PLANTED_HZ

        cpg.update(cpg_in)
        cpg_out = cpg.step()
        # Target limbs follow the reference trajectory (distinct L/R); the
        # controlled ankle is muscle-driven by the CPG.
        cpg_in = rig.step(cpg_out)

        if step in frame_steps:
            stage = "startup" if step < BELT_START_STEP else "locomotion"
            lines = hud_lines(cpg_in, cpg_out, cpg.current_time_ms, stage)
            for c in cams:
                writers[c].append_data(renderer.frame(c, lines))
        if step % 100 == 0:
            print(f"[{cpg.current_time_ms:6.0f} ms] CUT {cpg_in.cut_L:5.1f} "
                  f"ankleL {cpg_in.ankle_angle_m_L:+.3f} SOL {cpg_out.mus_E_L:.2f} "
                  f"TA {cpg_out.mus_F_L:.2f}", flush=True)

    for w in writers.values():
        w.close()
    renderer.close()
    print(f"wrote {ext} ({n_frames} frames, {duration_s:.0f}s sim) to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--view", type=str, default="oblique_view")
    ap.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "renders"))
    ap.add_argument("--n-rg", type=int, default=12)
    ap.add_argument("--n-mn", type=int, default=10)
    ap.add_argument("--n-int", type=int, default=8)
    ap.add_argument("--n-aff", type=int, default=6)
    ap.add_argument("--n-bs", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args.duration_s, [args.view], args.width, args.height, args.fps,
        args.out, args.format, args.n_rg, args.n_mn, args.n_int, args.n_aff, args.n_bs)