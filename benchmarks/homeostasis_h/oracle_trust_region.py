"""Feasibility oracle: direct parameter-space trust region, no H.

Each mean update is clipped to theta0 +/- R * (hi - lo). Sigma is untouched.
This is a probe, not a production controller.
"""
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench_homeostasis import PROD_SIZES, inflation_metrics


def run(radius, seed):
    from personalize import build_personalizer

    t0 = time.perf_counter()
    pers, _ = build_personalizer(
        max_steps=2000, eps_pose=0.10, pose_hold_steps=3,
        loss_stop=-1.0, loss_window=20, ref_hz=700.0, resolution_ms=0.2,
        seed=int(seed), homeostasis=None, **PROD_SIZES)
    agent = pers.agent
    pers.verbose = False
    pers.log_every = 0
    pers.stdp_every = 0
    pers.belt_mps = 0.2
    pers.belt_start_step = 50
    pers.belt_ramp_steps = 50

    initial = {name: float(p.mean) for name, p in agent.params.items()}
    bounds = {name: (float(p.min), float(p.max))
              for name, p in agent.params.items()}
    clipped = [0]
    active = [0]

    for name, param in agent.params.items():
        theta0 = initial[name]
        lo, hi = bounds[name]
        width = hi - lo
        if np.isfinite(width) and width > 0.0:
            allowed_lo = max(lo, theta0 - float(radius) * width)
            allowed_hi = min(hi, theta0 + float(radius) * width)
        else:
            allowed_lo, allowed_hi = lo, hi
        original = param.update_mean

        def update_mean(delta, p=param, orig=original,
                        alo=allowed_lo, ahi=allowed_hi):
            orig(delta)
            active[0] += 1
            clipped_value = float(np.clip(p.mean, alo, ahi))
            if clipped_value != p.mean:
                clipped[0] += 1
            p.mean = clipped_value

        param.update_mean = update_mean

    log = pers.run(pose_max_steps=100, realize_every_k=1)
    wall_s = time.perf_counter() - t0
    start = max(0, len(log.loss) - 1000)
    metrics = inflation_metrics(agent.history["means"], initial, bounds,
                                eval_start=start)
    return {
        "radius": float(radius),
        "seed": int(seed),
        "steps": int(len(log.loss)),
        "loss_eval": float(np.mean(log.loss[start:])),
        "loss_full": float(np.mean(log.loss[agent.warmup:])),
        "wall_s": float(wall_s),
        "clip_calls": int(clipped[0]),
        "update_calls": int(active[0]),
        "clip_fraction": float(clipped[0] / max(active[0], 1)),
        "n_params": int(metrics["n_params"]),
        **metrics,
    }


if __name__ == "__main__":
    radii = (0.40, 0.35, 0.30)
    seeds = (0, 1)
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [(r, s, pool.submit(run, r, s))
                   for s in seeds for r in radii]
        rows = []
        for r, s, future in futures:
            row = future.result()
            rows.append(row)
            print(f"[oracle] R={r:.2f} seed={s} p90={row['r_p90']:.4f} "
                  f"loss={row['loss_eval']:.4f} bound={row['at_bound']} "
                  f"q={row['q_gt_1']} clip={row['clip_fraction']:.3f} "
                  f"wall={row['wall_s']:.0f}s", flush=True)

    with open("benchmarks/homeostasis_h/real_compare_20s.csv",
              newline="", encoding="utf-8") as f:
        base = {(int(row["seed"])): row for row in csv.DictReader(f)
                if row["kind"] == "H_off"}
    for row in rows:
        b = base[row["seed"]]
        row["baseline_p90"] = float(b["r_p90"])
        row["baseline_loss"] = float(b["loss_eval"])
        row["baseline_q"] = int(b["q_gt_1"])
        row["p90_improvement"] = 1.0 - row["r_p90"] / row["baseline_p90"]
        row["loss_delta"] = row["loss_eval"] / row["baseline_loss"] - 1.0
        row["q_growth"] = row["q_gt_1"] - row["baseline_q"]

    for radius in radii:
        group = [r for r in rows if r["radius"] == radius]
        print(f"[oracle] R={radius:.2f} mean_p90="
              f"{100*np.mean([r['p90_improvement'] for r in group]):.1f}% "
              f"worst_p90={100*np.min([r['p90_improvement'] for r in group]):.1f}% "
              f"mean_loss={100*np.mean([r['loss_delta'] for r in group]):.1f}% "
              f"max_q_growth={max(r['q_growth'] for r in group)}", flush=True)

    out = "benchmarks/homeostasis_h/oracle_trust_region_20s"
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out + ".json and .csv", flush=True)
