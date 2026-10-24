#!/usr/bin/env python
"""Two-minute personalisation + validation run.

Runs the CPG + MuJoCo + PERSONA loop for `--duration-s` (default 120 s at
dt = 10 ms = 12000 steps) and writes `run.h5` (all data), `run_log.csv`, the
validation figures, and (unless `--no-render`) an mp4 into `--out`.

This always uses the full `cpg_2legs_fast` population sizes
(`POPULATIONS_FINAL`), matching `run_validation.py --final`. It used to hardcode
a 10-neuron network, which made a "2 min run" measure a mostly-disconnected CPG.

Usage:
    python run_2min.py --out validation_2min --no-render      # headless server
    python run_2min.py --out validation_2min                  # with mp4

Server notes:
    --snapshot-s 900        atomic run.h5/run_log.csv checkpoints every 15 min
    --resolution-ms 0.2     NEST resolution (default: the cpg_2legs_fast value).
                            Do not raise it to 1.0: the izhikevich network then
                            locks into a ~500 Hz limit cycle and the flexor drive
                            pins at 1.0 no matter what PERSONA samples.
    --realize-every-k K     sample+realize every K steps (1 = full fidelity).
                            At FINAL, K=1 costs several s/step on a laptop, so a
                            120 s run is many hours there and much less on a server.
    --budget-s N            stop after N seconds of wall clock (0/absent = off)
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402

import run_validation  # noqa: E402
from persona.game import GameConfig  # noqa: E402
from t2xcpg.cfg import POPULATIONS_FINAL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation_2min")
    ap.add_argument("--duration-s", type=float, default=120.0)
    ap.add_argument("--view", default="oblique_view")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the mp4 (headless server)")
    ap.add_argument("--ref-hz", type=float, default=run_validation.DEFAULT_REF_HZ,
                    help="motor rate mapped to full muscle drive")
    ap.add_argument("--resolution-ms", type=float,
                    default=run_validation.DEFAULT_RESOLUTION_MS,
                    help="NEST resolution in ms (default 0.2; 1.0 saturates the "
                         "izhikevich network and pins the flexor drive)")
    ap.add_argument("--belt", type=str, default="dataset",
                    choices=("dataset", "model"))
    ap.add_argument("--snapshot-s", type=float, default=900.0,
                    help="wall-clock interval between atomic checkpoints (0 disables)")
    ap.add_argument("--realize-every-k", type=int, default=1,
                    help="sample + realize + record the network every k steps")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="stop after this many seconds of real time")
    ap.add_argument("--pose-max-s", type=float, default=1.0,
                    help="cap on the settling (pose) stage, in seconds")
    ap.add_argument("--cut-multisite", action="store_true",
                    help="Option B: one CUT Poisson relay and STDP projection "
                         "per plantar site (requires the PERSONA update)")
    ap.add_argument("--threads", type=int, default=1,
                    help="independent CPG/MuJoCo replicas and NEST local "
                         "threads (>= 2 uses the parallel MultiAgent path; "
                         "the current thread path is experimental and slow)")
    ap.add_argument("--game", choices=("none", "selected"), default="selected",
                    help="conserved-stack game; 'selected' uses "
                         "GameConfig.current_best()")
    ap.add_argument("--seed", type=int, default=0,
                    help="run seed (also the NEST rng_seed)")
    ap.add_argument("--lr-mode", choices=("constant", "cosine"),
                    default="constant")
    ap.add_argument("--lr-start", type=float, default=0.03)
    ap.add_argument("--lr-end", type=float, default=None)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads must be >= 1")
    if args.lr_mode == "cosine" and args.lr_end is None:
        ap.error("--lr-end is required with --lr-mode cosine")
    run_validation.run(
        duration_s=args.duration_s, view=args.view,
        width=1920, height=1080, fps=40, out_dir=args.out,
        w_scale=1.0, ref_hz=args.ref_hz, belt_mps=0.2, belt_start_s=0.5,
        belt_ramp_s=0.5,
        n_rg=POPULATIONS_FINAL["n_rg"], n_mn=POPULATIONS_FINAL["n_mn"],
        n_int=POPULATIONS_FINAL["n_int"], n_aff=POPULATIONS_FINAL["n_aff"],
        n_bs=POPULATIONS_FINAL["n_bs"],
        render=not args.no_render, budget_s=args.budget_s,
        pose_max_s=args.pose_max_s, belt_mode=args.belt,
        snapshot_s=args.snapshot_s, realize_every_k=args.realize_every_k,
        resolution_ms=args.resolution_ms,
        cut_multisite=args.cut_multisite,
        game=(GameConfig.current_best() if args.game == "selected" else None),
        seed=args.seed, lr_mode=args.lr_mode, lr_start=args.lr_start,
        lr_end=args.lr_end, threads=args.threads)


if __name__ == "__main__":
    main()
