"""
20-second full-framework validation run:
  * builds the closed loop (bilateral CPG + MuJoCo rig + mocap target),
  * runs PERSONA personalization inside the loop,
  * renders an oblique-view HUD movie (mp4),
  * writes the run log and the validation figures (tracking, learning,
    parameter distribution shift, Fisher matrix).

Usage (headless):
    MUJOCO_GL=osmesa python run_validation.py --duration-s 20 --view oblique_view
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
# OpenBLAS spawns many threads that thrash on WSL (a 322x322 solve goes from
# ~1.3 s to ~2 ms with a single thread). Force single-threaded BLAS.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import imageio

from personalize import build_personalizer, DEFAULT_REFERENCE, DT_MS
from persona.validation import ValidationReport
from persona.h5io import save_run
from model.render import LegRenderer, hud_lines
from t2xcpg.cfg import POPULATIONS_FINAL


def run(duration_s, view, width, height, fps, out_dir,
        w_scale, ref_hz, belt_mps, belt_start_s, belt_ramp_s,
        n_rg, n_mn, n_int, n_aff, n_bs, render=True, budget_s=None,
        pose_max_s=1.0):
    os.makedirs(out_dir, exist_ok=True)
    steps = int(round(duration_s / (DT_MS / 1000.0)))
    n_frames = int(round(duration_s * fps))
    frame_steps = {int(round(k / (fps * (DT_MS / 1000.0)))) for k in range(n_frames)}

    pers, loader = build_personalizer(
        reference_path=DEFAULT_REFERENCE,
        n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff, n_bs=n_bs,
        max_steps=steps, eps_pose=0.10, pose_hold_steps=3,
        loss_stop=1e-4, loss_window=20, w_scale=w_scale, ref_hz=ref_hz)
    pers.belt_mps = float(belt_mps)
    pers.belt_start_step = int(round(belt_start_s / (DT_MS / 1000.0)))
    pers.belt_ramp_steps = max(1, int(round(belt_ramp_s / (DT_MS / 1000.0))))
    pers.log_every = 100
    agent = pers.agent
    rig = pers.rig
    renderer = LegRenderer(rig.model, rig.data, width=width, height=height) if render else None
    writer = None
    if render:
        path = os.path.join(out_dir, f"locomotion_{view}.mp4")
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1)

    def _hook(step, log, cpg_in, cpg_out, stage):
        if render and step in frame_steps:
            lines = hud_lines(cpg_in, cpg_out, log.time_ms[-1], stage)
            writer.append_data(renderer.frame(view, lines))

    status = {}
    try:
        log = pers.run(step_hook=_hook, wall_clock_s=budget_s,
                       pose_max_steps=int(round(pose_max_s / (DT_MS / 1000.0))))
    finally:
        if writer is not None:
            try:
                writer.close()
                status["locomotion.mp4"] = "ok"
            except Exception as e:
                status["locomotion.mp4"] = f"ERROR {e}"
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    try:
        log.to_csv(os.path.join(out_dir, "run_log.csv"))
        status["run_log.csv"] = "ok"
    except Exception as e:
        status["run_log.csv"] = f"ERROR {e}"
    try:
        save_run(os.path.join(out_dir, "run.h5"), log, agent,
                 meta={"duration_s": float(duration_s), "dt_ms": float(DT_MS)})
        status["run.h5"] = "ok"
    except Exception as e:
        status["run.h5"] = f"ERROR {e}"

    rep = ValidationReport()
    for name, fn in (("tracking", rep.plot_tracking),
                     ("model_vs_target", rep.plot_model_vs_target),
                     ("learning", rep.plot_learning)):
        try:
            fn(log)
            status[f"{name}.png"] = "ok"
        except Exception as e:
            status[f"{name}.png"] = f"ERROR {e}"
    for name, fn in (("param_shift", lambda: rep.plot_param_shift(agent, top_n=10)),
                     ("fisher", lambda: rep.plot_fisher(agent)),
                     ("influence", lambda: rep.plot_influence(agent)),
                     ("metric", lambda: rep.plot_metric(agent))):
        try:
            fn()
            status[f"{name}.png"] = "ok"
        except Exception as e:
            status[f"{name}.png"] = f"ERROR {e}"
    try:
        rep.save_all(out_dir)
    except Exception as e:
        status["save_all"] = f"ERROR {e}"

    print("---- output status ----")
    for k, v in status.items():
        print(f"  {k}: {v}")
    print("files in", out_dir, ":", sorted(os.listdir(out_dir)))
    print("first10 loss", float(np.mean(log.loss[:10])), "last10", float(np.mean(log.loss[-10:])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=20.0)
    ap.add_argument("--view", type=str, default="oblique_view")
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "validation_run"))
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--ref-hz", type=float, default=100.0)
    ap.add_argument("--belt-mps", type=float, default=0.2)
    ap.add_argument("--belt-start-s", type=float, default=0.5)
    ap.add_argument("--belt-ramp-s", type=float, default=0.5)
    ap.add_argument("--n-rg", type=int, default=10)
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--n-int", type=int, default=8)
    ap.add_argument("--n-aff", type=int, default=6)
    ap.add_argument("--n-bs", type=int, default=8)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="stop after this many seconds of real time (server budget)")
    ap.add_argument("--pose-max-s", type=float, default=1.0,
                    help="cap on the settling (pose) stage, in seconds")
    ap.add_argument("--final", action="store_true",
                    help="full cpg_2legs_fast population sizes")
    ap.add_argument("--fast", action="store_true",
                    help="quick low-res short run for iteration")
    args = ap.parse_args()
    if args.fast:
        args.duration_s = min(args.duration_s, 4.0)
        args.width, args.height, args.fps = 480, 270, 20
        args.n_rg, args.n_mn, args.n_int, args.n_aff, args.n_bs = 6, 5, 5, 3, 5
    elif args.final:
        args.n_rg = POPULATIONS_FINAL["n_rg"]
        args.n_mn = POPULATIONS_FINAL["n_mn"]
        args.n_int = POPULATIONS_FINAL["n_int"]
        args.n_aff = POPULATIONS_FINAL["n_aff"]
        args.n_bs = POPULATIONS_FINAL["n_bs"]
    run(args.duration_s, args.view, args.width, args.height, args.fps, args.out,
        args.w_scale, args.ref_hz, args.belt_mps, args.belt_start_s, args.belt_ramp_s,
        args.n_rg, args.n_mn, args.n_int, args.n_aff, args.n_bs, render=not args.no_render,
        budget_s=args.budget_s, pose_max_s=args.pose_max_s)