#!/usr/bin/env python
"""Two-minute personalisation + validation run.

Runs the CPG + MuJoCo + PERSONA loop for `--duration-s` (default 120 s at
dt = 10 ms = 12000 steps) and writes `run.h5` (all data), `run_log.csv`, the
validation figures, and (unless `--no-render`) an mp4 into `--out`.

Usage:
    python run_2min.py --out validation_2min
    python run_2min.py --out validation_2min --no-render      # headless server
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402

import run_validation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation_2min")
    ap.add_argument("--duration-s", type=float, default=120.0)
    ap.add_argument("--view", default="oblique_view")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the mp4 (headless server)")
    args = ap.parse_args()
    run_validation.run(
        duration_s=args.duration_s, view=args.view,
        width=1920, height=1080, fps=40, out_dir=args.out,
        w_scale=1.0, ref_hz=100.0, belt_mps=0.2, belt_start_s=0.5,
        belt_ramp_s=0.5, n_rg=10, n_mn=8, n_int=8, n_aff=6, n_bs=8,
        render=not args.no_render)


if __name__ == "__main__":
    main()