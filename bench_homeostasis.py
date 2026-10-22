"""A/B benchmark harness for the PERSONA homeostatic regulator.

The synthetic arm is deliberately registry-free: twelve scalar-credit players
on a ring with two chords, the frozen plant from the approved design, and the
same fixed noise stream for H off and H on. It isolates the score/share/H math
and runs in seconds.

The production arms reuse ``personalize.build_personalizer``; they are opt-in
because each run builds NEST and MuJoCo.

Usage::

    python bench_homeostasis.py --synthetic --out benchmarks/homeostasis_h
    python bench_homeostasis.py --fast-smoke --jobs 5
    python bench_homeostasis.py --acceptance --jobs 5
    python bench_homeostasis.py --all --jobs 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# NEST/MuJoCo runs are single-threaded; BLAS thread pools otherwise oversubscribe
# parallel benchmark arms (the validation harness makes the same choice).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from persona.homeostasis import (
    HomeostaticConfig,
    HomeostaticRegulator,
    PhysicalMap,
    PlayerSpec,
)


SYNTHETIC_STEPS = 1000
SYNTHETIC_EVAL = 500
SYNTHETIC_SEED = 20261021
SYNTHETIC_PLAYERS = 12
SYNTHETIC_GAIN = 0.25
SYNTHETIC_DRIFT = 0.005
SYNTHETIC_NOISE = 0.02


def synthetic_map() -> tuple[PhysicalMap, list[str]]:
    specs = [
        PlayerSpec(f"p{i}", "population", (f"s{i}",), size=1, ptype="exc")
        for i in range(SYNTHETIC_PLAYERS)
    ]
    edges = [(i, (i + 1) % SYNTHETIC_PLAYERS) for i in range(SYNTHETIC_PLAYERS)]
    # Two chords: still connected, but with loops unlike a pure ring.
    edges += [(0, SYNTHETIC_PLAYERS // 2), (3, (3 + SYNTHETIC_PLAYERS // 2)
                                             % SYNTHETIC_PLAYERS)]
    pm = PhysicalMap.from_specs(specs, edges)
    return pm, [site for spec in specs for site in spec.param_names]


def run_synthetic_arm(h_on: bool, *, steps: int = SYNTHETIC_STEPS) -> dict:
    pm, sites = synthetic_map()
    cfg = HomeostaticConfig(
        enabled=True,
        h_burnin_steps=100,
        global_variance_mode="budget",
        lambda_H=1.0 if h_on else 0.0,
        tau_H_s=5.0,
    )
    reg = HomeostaticRegulator(pm, cfg, dt_s=0.01, beta=0.995,
                               site_names=sites)
    rng = np.random.default_rng(SYNTHETIC_SEED)
    c = np.zeros(SYNTHETIC_PLAYERS, dtype=np.float64)
    sigmas = np.full(SYNTHETIC_PLAYERS, 0.1, dtype=np.float64)
    F = np.eye(SYNTHETIC_PLAYERS, dtype=np.float64)
    index = np.arange(SYNTHETIC_PLAYERS, dtype=np.float64)
    e_rows, proj_rows, ell_rows = [], [], []
    u_norms = []
    for t in range(steps):
        u = reg.step_players(c, sigmas, 1.0, F, t + 1)
        if reg.stage == "active":
            e_rows.append(np.asarray(reg.last["e"], dtype=np.float64).copy())
            proj_rows.append((np.asarray(reg.last["projected_pi"])
                              - np.asarray(reg.last["a"])).copy())
            ell_rows.append(np.asarray(reg.last["ell"],
                                       dtype=np.float64).copy())
            u_norms.append(float(np.linalg.norm(u)))
        c = (c + SYNTHETIC_GAIN * u
             + SYNTHETIC_DRIFT * np.sin(2.0 * np.pi * t / 200.0 + index)
             + SYNTHETIC_NOISE * rng.normal(size=SYNTHETIC_PLAYERS))

    e = np.asarray(e_rows, dtype=np.float64)
    proj = np.asarray(proj_rows, dtype=np.float64)
    ell = np.asarray(ell_rows, dtype=np.float64)
    n_eval = min(SYNTHETIC_EVAL, len(e))
    e_eval = e[-n_eval:]
    proj_eval = proj[-n_eval:]
    ell_eval = ell[-n_eval:]
    return {
        "arm": "h_on" if h_on else "h_off",
        "share_rms": float(np.sqrt(np.mean(np.sum(e_eval ** 2, axis=1)))),
        "projected_rms": float(np.sqrt(np.mean(np.sum(proj_eval ** 2,
                                                      axis=1)))),
        "loss_proxy": float(-np.mean(ell_eval)),
        "u_norm_mean": float(np.mean(u_norms)) if u_norms else 0.0,
        "steps": int(steps),
        "eval_steps": int(n_eval),
        "mode": cfg.global_variance_mode,
        "lambda_H": cfg.lambda_H,
    }


def run_synthetic(*, steps: int = SYNTHETIC_STEPS) -> dict:
    off = run_synthetic_arm(False, steps=steps)
    on = run_synthetic_arm(True, steps=steps)
    reduction = 1.0 - on["share_rms"] / max(off["share_rms"], 1e-300)
    return {"h_off": off, "h_on": on,
            "share_reduction": float(reduction),
            "projected_reduction": float(
                1.0 - on["projected_rms"] / max(off["projected_rms"], 1e-300)),
            "passes_20pct": bool(reduction >= 0.20),
            "target_50pct": bool(reduction >= 0.50)}


def _write_synthetic(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_synthetic()
    rows = [result["h_off"], result["h_on"]]
    csv_path = out_dir / "synthetic_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "synthetic_summary.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"synthetic: H off RMS={result['h_off']['share_rms']:.6g} "
          f"H on RMS={result['h_on']['share_rms']:.6g} "
          f"reduction={100.0 * result['share_reduction']:.1f}%")
    print(f"projected reduction={100.0 * result['projected_reduction']:.1f}% "
          f"loss H off={result['h_off']['loss_proxy']:.5g} "
          f"H on={result['h_on']['loss_proxy']:.5g}")
    print(f"wrote {csv_path} and {json_path}")
    return result


# --- production A/B metrics -------------------------------------------------

FAST_SIZES = dict(n_rg=6, n_mn=5, n_int=5, n_aff=3, n_bs=5)
PROD_SIZES = dict(n_rg=10, n_mn=8, n_int=8, n_aff=6, n_bs=8)
FINAL_SIZES = dict(n_rg=100, n_mn=100, n_int=50, n_aff=100, n_bs=100)
MODES = ("off", "advantage", "budget", "fisher", "band")
CURRENT_BEST_MODE = "current_best"
FAST_STEPS = 400
ACCEPT_STEPS = 2000
EVAL_STEPS = 1000


def _homeostatic_config(mode, h_burnin_steps=100):
    """Build the config for one arm; ``current_best`` is experimental only."""
    if mode == "off":
        return None
    if mode == CURRENT_BEST_MODE:
        cfg = HomeostaticConfig.current_best()
        cfg.h_burnin_steps = int(h_burnin_steps)
        return cfg
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    return HomeostaticConfig(
        enabled=True,
        global_variance_mode=mode,
        h_burnin_steps=int(h_burnin_steps),
        lambda_H=1.0,
    )


def inflation_metrics(means, initial, bounds, *, eval_start=0):
    """Per-parameter inflation ratios on the evaluation window.

    ``r_p`` is range-normalized by the biological bound width; an infinite
    bound falls back to ``max(|theta0|, 1)`` and is flagged. ``q_p`` is the
    magnitude-normalized ratio reported as a count above 1.
    """
    r_values = []
    q_values = []
    at_bound = 0
    range_fallback = 0
    for name, values in means.items():
        arr = np.asarray(values[eval_start:], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0 or name not in initial:
            continue
        theta0 = float(initial[name])
        lo, hi = bounds.get(name, (-np.inf, np.inf))
        finite_bounds = np.isfinite(lo) and np.isfinite(hi)
        if finite_bounds:
            denom = float(hi - lo)
            if not np.isfinite(denom) or denom <= 0.0:
                denom = max(abs(theta0), 1.0)
                range_fallback += 1
        else:
            denom = max(abs(theta0), 1.0)
            range_fallback += 1
        delta = np.max(np.abs(arr - theta0))
        r_values.append(float(delta / denom))
        q_values.append(float(delta / max(abs(theta0), 1e-12)))
        if finite_bounds:
            at_bound += int(np.any(np.isclose(arr, lo, atol=1e-9,
                                              rtol=1e-9))
                            or np.any(np.isclose(arr, hi, atol=1e-9,
                                                rtol=1e-9)))
    r = np.asarray(r_values, dtype=np.float64)
    q = np.asarray(q_values, dtype=np.float64)
    return {
        "n_params": int(r.size),
        "r_median": float(np.median(r)) if r.size else 0.0,
        "r_p90": float(np.percentile(r, 90)) if r.size else 0.0,
        "r_max": float(np.max(r)) if r.size else 0.0,
        "r_gt_0_25": int(np.sum(r > 0.25)),
        "q_gt_1": int(np.sum(q > 1.0)),
        "at_bound": int(at_bound),
        "range_fallback": int(range_fallback),
    }


def type_sign_violations(means, bounds, ptypes=None):
    """Hard-bound and static-weight sign violations."""
    ptypes = ptypes or {}
    bad = []
    for name, values in means.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        lo, hi = bounds.get(name, (-np.inf, np.inf))
        violation = bool(np.any(arr < lo - 1e-9) or np.any(arr > hi + 1e-9))
        if "weight" in name:
            if lo >= 0.0 and np.any(arr < 0.0):
                violation = True
            if hi <= 0.0 and np.any(arr > 0.0):
                violation = True
        if violation:
            bad.append(name)
    return bad


def _phase_agreement(phases, reference):
    if not phases or not reference:
        return 1.0
    n = min(len(phases), len(reference))
    return float(np.mean(np.asarray(phases[:n]) == np.asarray(reference[:n])))


def production_arm(mode, seed, *, steps, eval_steps, sizes, h_burnin_steps=100):
    """One production-path run. Imports NEST/MuJoCo lazily."""
    from personalize import build_personalizer
    from persona.homeostasis import HomeostaticConfig

    if mode not in MODES and mode != CURRENT_BEST_MODE:
        raise ValueError(f"unknown mode {mode!r}")
    cfg = _homeostatic_config(mode, h_burnin_steps)
    pers, _ = build_personalizer(
        max_steps=int(steps), eps_pose=0.10, pose_hold_steps=3,
        loss_stop=-1.0, loss_window=20, ref_hz=700.0, resolution_ms=0.2,
        seed=int(seed), homeostasis=cfg, **sizes)
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
    ptypes = {}
    n_players = None
    if agent.homeostasis is not None:
        n_players = int(agent.homeostasis.physical_map.n_players)
        for spec in agent.homeostasis.physical_map.specs:
            for site in spec.param_names:
                ptypes[site] = spec.ptype

    t0 = time.perf_counter()
    log = pers.run(pose_max_steps=100, realize_every_k=1)
    wall_s = time.perf_counter() - t0
    n_steps = int(len(log.loss))
    eval_start = max(0, n_steps - int(eval_steps))
    metrics = inflation_metrics(agent.history["means"], initial, bounds,
                                eval_start=eval_start)
    bad = type_sign_violations(agent.history["means"], bounds, ptypes)
    hs = agent.history.get("homeostasis", [])
    ts = [float(d.get("T", np.nan)) for d in hs[eval_start:]]
    share_traj = [float(np.linalg.norm(np.asarray(d.get("e", []), dtype=float)))
                  for d in hs[eval_start:]]
    share = (float(np.sqrt(np.mean(np.square(share_traj))))
             if share_traj else None)
    return {
        "mode": mode,
        "seed": int(seed),
        "steps": n_steps,
        "eval_steps": n_steps - eval_start,
        "h_burnin_steps": 0 if mode == "off" else int(h_burnin_steps),
        "lift_mode": None if cfg is None else cfg.lift_mode,
        "h_delay_steps": None if cfg is None else cfg.h_delay_steps,
        "n_players": n_players,
        "loss_eval": float(np.mean(log.loss[eval_start:])),
        "loss_full": float(np.mean(log.loss[agent.warmup:])),
        "wall_s": float(wall_s),
        "step_ms": float(1000.0 * wall_s / max(n_steps, 1)),
        "violations": int(len(bad)),
        "violation_names": bad[:10],
        "share_rms": share,
        "share_traj": share_traj,
        "T_min": float(np.nanmin(ts)) if ts else None,
        "T_max": float(np.nanmax(ts)) if ts else None,
        "phases": [str(p) for p in agent.history["phase"]],
        "stage": str(log.stage[-1]) if log.stage else "unknown",
        "sizes": dict(sizes),
        **metrics,
    }


def _production_task(args):
    return production_arm(**args)


def _run_tasks(tasks, jobs):
    if jobs <= 1:
        out = []
        for task in tasks:
            out.append(_production_task(task))
            r = out[-1]
            print(f"[arm] seed={r['seed']} mode={r['mode']} "
                  f"loss_eval={r['loss_eval']:.5g} p90={r['r_p90']:.4g} "
                  f"wall={r['wall_s']:.1f}s", flush=True)
        return out
    out = []
    with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
        futures = [pool.submit(_production_task, task) for task in tasks]
        for fut in as_completed(futures):
            r = fut.result()
            out.append(r)
            print(f"[arm] seed={r['seed']} mode={r['mode']} "
                  f"loss_eval={r['loss_eval']:.5g} p90={r['r_p90']:.4g} "
                  f"wall={r['wall_s']:.1f}s", flush=True)
    return out


def _write_run_rows(path, rows):
    flat = []
    for row in rows:
        out = {k: v for k, v in row.items()
               if k not in ("phases", "violation_names", "share_traj")}
        out["violation_names"] = ";".join(row.get("violation_names", []))
        flat.append(out)
    if not flat:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _write_share_trajectories(path, rows):
    """Long-form per-step share error for the evaluated window."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=("seed", "mode", "eval_step", "share_norm"))
        writer.writeheader()
        for row in rows:
            for i, value in enumerate(row.get("share_traj") or []):
                writer.writerow({"seed": row["seed"], "mode": row["mode"],
                                 "eval_step": i + 1,
                                 "share_norm": float(value)})


def _paired_summary(rows, seeds, modes):
    by = {(r["seed"], r["mode"]): r for r in rows}
    summary = []
    for mode in modes:
        if mode == "off":
            continue
        improvements = []
        loss_deltas = []
        q_growth = []
        for seed in seeds:
            base = by[(seed, "off")]
            cur = by[(seed, mode)]
            improvements.append(1.0 - cur["r_p90"] / max(base["r_p90"], 1e-300))
            loss_deltas.append(cur["loss_eval"] / max(base["loss_eval"], 1e-300)
                               - 1.0)
            q_growth.append(cur["q_gt_1"] - base["q_gt_1"])
        finite = all(
            np.isfinite(by[(seed, mode)]["loss_eval"])
            and np.isfinite(by[(seed, mode)]["r_p90"])
            and (by[(seed, mode)]["T_min"] is None
                 or np.isfinite(by[(seed, mode)]["T_min"]))
            and (by[(seed, mode)]["T_max"] is None
                 or np.isfinite(by[(seed, mode)]["T_max"]))
            and (mode == "off"
                 or (by[(seed, mode)].get("share_rms") is not None
                     and np.isfinite(by[(seed, mode)]["share_rms"])))
            for seed in seeds)
        band_ok = True
        if mode == "band":
            band_ok = all((by[(seed, mode)]["T_min"] is not None
                           and by[(seed, mode)]["T_max"] is not None
                           and by[(seed, mode)]["T_min"] >= 0.5
                           and by[(seed, mode)]["T_max"] <= 2.0)
                          for seed in seeds)
        summary.append({
            "mode": mode,
            "mean_p90_improvement": float(np.mean(improvements)),
            "min_p90_improvement": float(np.min(improvements)),
            "mean_loss_delta": float(np.mean(loss_deltas)),
            "max_loss_delta": float(np.max(loss_deltas)),
            "q_growth_sum": int(np.sum(q_growth)),
            "q_growth_max": int(np.max(q_growth)),
            "finite": bool(finite),
            "band_ok": bool(band_ok),
            "accept_20pct": bool(np.mean(improvements) >= 0.20
                                 and np.min(improvements) >= -0.10),
            "target_50pct": bool(np.mean(improvements) >= 0.50),
            "accept_loss": bool(np.mean(loss_deltas) <= 0.10),
            "accept_q": bool(np.max(q_growth) <= 0),
            "passes": bool(np.mean(improvements) >= 0.20
                           and np.min(improvements) >= -0.10
                           and np.mean(loss_deltas) <= 0.10
                           and np.max(q_growth) <= 0 and finite and band_ok),
        })
    return summary


def run_fast_smoke(out_dir: Path, *, steps=FAST_STEPS, sizes=None,
                   seed=0, jobs=1):
    sizes = dict(FAST_SIZES if sizes is None else sizes)
    tasks = [dict(mode=mode, seed=seed, steps=steps, eval_steps=min(steps, 200),
                  sizes=sizes) for mode in MODES]
    rows = _run_tasks(tasks, jobs)
    for r in rows:
        if r["mode"] != "off":
            r["phase_agreement"] = _phase_agreement(
                r["phases"], next(x["phases"] for x in rows
                                  if x["mode"] == "off"))
        else:
            r["phase_agreement"] = 1.0
    result = {
        "layer": "fast_smoke",
        "steps": int(steps),
        "seed": int(seed),
        "sizes": sizes,
        "all_finite": bool(all(
            np.isfinite(r["loss_eval"]) and np.isfinite(r["r_p90"])
            and (r["mode"] == "off"
                 or (r["share_rms"] is not None
                     and np.isfinite(r["share_rms"])))
            for r in rows)),
        "zero_violations": bool(all(r["violations"] == 0 for r in rows)),
        "diagnostics": bool(all(r["mode"] == "off" or r["share_rms"] is not None
                                for r in rows)),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_run_rows(out_dir / "fast_smoke.csv", rows)
    _write_share_trajectories(out_dir / "share_trajectories.csv", rows)
    (out_dir / "fast_smoke_summary.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2),
        encoding="utf-8")
    print(f"fast smoke: finite={result['all_finite']} "
          f"violations={result['zero_violations']} "
          f"diagnostics={result['diagnostics']}", flush=True)
    return result


def run_acceptance(out_dir: Path, *, seeds=(0, 1), steps=ACCEPT_STEPS,
                   eval_steps=EVAL_STEPS, sizes=None, jobs=1):
    sizes = dict(PROD_SIZES if sizes is None else sizes)
    tasks = [dict(mode=mode, seed=seed, steps=steps, eval_steps=eval_steps,
                  sizes=sizes) for seed in seeds for mode in MODES]
    rows = _run_tasks(tasks, jobs)
    by_seed = {seed: next(r["phases"] for r in rows
                          if r["seed"] == seed and r["mode"] == "off")
               for seed in seeds}
    for r in rows:
        r["phase_agreement"] = (_phase_agreement(r["phases"],
                                                 by_seed[r["seed"]])
                                if r["mode"] != "off" else 1.0)
    summary = _paired_summary(rows, seeds, MODES)
    all_zero = bool(all(r["violations"] == 0 for r in rows))
    all_finite = bool(all(m["finite"] for m in summary))
    band_ok = bool(all(m["band_ok"] for m in summary if m["mode"] == "band"))
    passing = [m for m in summary if m["passes"]]
    best = (max(passing,
                key=lambda m: (m["mean_p90_improvement"],
                               -m["mean_loss_delta"]))
            if passing else None)
    result = {
        "layer": "acceptance",
        "steps": int(steps),
        "eval_steps": int(eval_steps),
        "seeds": list(seeds),
        "sizes": sizes,
        "modes": summary,
        "baseline_exact": True,
        "baseline_note": ("H off is the no-regulator path; exact parity is "
                          "covered by the H/strategy tests."),
        "same_seed_per_pair": True,
        "all_finite": all_finite,
        "all_zero_violations": all_zero,
        "band_in_band": band_ok,
        "modes_passing": [m["mode"] for m in passing],
        "chosen_mode": best["mode"] if best is not None else None,
        "passes": bool(all_finite and all_zero and band_ok and passing),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_run_rows(out_dir / "acceptance_runs.csv", rows)
    _write_share_trajectories(out_dir / "share_trajectories.csv", rows)
    with (out_dir / "acceptance_summary.csv").open("w", newline="",
                                                    encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "acceptance_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    for m in summary:
        print(f"[acceptance] {m['mode']}: "
              f"p90 improve={100*m['mean_p90_improvement']:.1f}% "
              f"loss delta={100*m['mean_loss_delta']:.1f}% "
              f"q growth={m['q_growth_max']} pass={m['passes']}", flush=True)
    print(f"acceptance pass={result['passes']} "
          f"zero violations={result['all_zero_violations']} "
          f"chosen={result['chosen_mode']}", flush=True)
    return result


def _write_figures(out_dir, pd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synthetic = out_dir / "synthetic_results.csv"
    if synthetic.exists():
        df = pd.read_csv(synthetic)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.bar(df["arm"], df["share_rms"], color=("#777777", "#2a7fb8"))
        ax.set_ylabel("RMS share error")
        ax.set_title("Frozen synthetic fixture")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "synthetic_share.png", dpi=150)
        plt.close(fig)

    fast = out_dir / "fast_smoke.csv"
    if fast.exists():
        df = pd.read_csv(fast)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
        x = np.arange(len(df))
        axes[0].bar(x, df["r_p90"], color="#2a7fb8")
        axes[1].bar(x, df["loss_eval"], color="#d07a2a")
        for ax, title, ylabel in (
                (axes[0], "Fast smoke: p90 inflation", "p90 r_p"),
                (axes[1], "Fast smoke: eval loss", "loss_eval")):
            ax.set_xticks(x)
            ax.set_xticklabels(df["mode"], rotation=25, ha="right")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "fast_smoke.png", dpi=150)
        plt.close(fig)

    summary = out_dir / "acceptance_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        x = np.arange(len(df))
        axes[0].bar(x, 100.0 * df["mean_p90_improvement"], color="#2a7fb8")
        axes[0].axhline(20.0, color="black", linestyle="--", linewidth=1.0,
                        label="20% acceptance")
        axes[0].set_ylabel("mean p90 improvement (%)")
        axes[0].set_title("Inflation reduction vs H off")
        axes[0].legend(fontsize=8)
        axes[1].bar(x, 100.0 * df["mean_loss_delta"], color="#d07a2a")
        axes[1].axhline(10.0, color="black", linestyle="--", linewidth=1.0,
                        label="10% loss limit")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("mean loss change (%)")
        axes[1].set_title("Eval loss vs H off")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(df["mode"], rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "acceptance_modes.png", dpi=150)
        plt.close(fig)

    runs = out_dir / "acceptance_runs.csv"
    if runs.exists():
        df = pd.read_csv(runs)
        modes = [m for m in MODES if m in set(df["mode"])]
        seeds = sorted(df["seed"].unique())
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        width = 0.8 / max(len(seeds), 1)
        x = np.arange(len(modes))
        for i, seed in enumerate(seeds):
            sub = df[df["seed"] == seed].set_index("mode").reindex(modes)
            ax.bar(x + i * width, sub["r_p90"], width=width,
                   label=f"seed {seed}")
        ax.set_xticks(x + width * (len(seeds) - 1) / 2.0)
        ax.set_xticklabels(modes, rotation=25, ha="right")
        ax.set_ylabel("p90 r_p (eval)")
        ax.set_title("Production p90 inflation by paired seed")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "acceptance_p90.png", dpi=150)
        plt.close(fig)

    traj = out_dir / "share_trajectories.csv"
    if traj.exists():
        df = pd.read_csv(traj)
        df = df[df["mode"] != "off"]
        if not df.empty:
            fig, ax = plt.subplots(figsize=(8.0, 4.0))
            for mode, sub in df.groupby("mode"):
                mean = sub.groupby("eval_step")["share_norm"].mean()
                ax.plot(mean.index, mean.to_numpy(), label=mode)
            ax.set_xlabel("evaluation control step")
            ax.set_ylabel("mean ||e||")
            ax.set_title("Share-error trajectory")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "share_trajectories.png", dpi=150)
            plt.close(fig)


def _write_artifacts(out_dir):
    """Write the combined XLSX and PNGs from the CSV layer."""
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional artifact layer
        print(f"[artifacts] pandas unavailable: {exc}", flush=True)
        return
    files = ("synthetic_results.csv", "fast_smoke.csv",
             "acceptance_runs.csv", "acceptance_summary.csv",
             "share_trajectories.csv")
    frames = {}
    for name in files:
        path = out_dir / name
        if path.exists():
            try:
                frames[path.stem] = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - corrupt partial file
                print(f"[artifacts] skip {name}: {exc}", flush=True)
    if frames:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except ImportError:
            try:
                import xlsxwriter  # noqa: F401
                engine = "xlsxwriter"
            except ImportError:
                engine = None
        if engine is not None:
            path = out_dir / "homeostasis_h_results.xlsx"
            with pd.ExcelWriter(path, engine=engine) as writer:
                for name, frame in frames.items():
                    frame.to_excel(writer, sheet_name=name[:31], index=False)
            print(f"[artifacts] wrote {path}", flush=True)
        else:  # pragma: no cover - optional engine
            print("[artifacts] no XLSX engine installed", flush=True)
    try:
        _write_figures(out_dir, pd)
        print(f"[artifacts] wrote PNGs to {out_dir}", flush=True)
    except Exception as exc:  # pragma: no cover - optional artifact layer
        print(f"[artifacts] PNGs unavailable: {exc}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="run the frozen synthetic closed-loop A/B")
    ap.add_argument("--fast-smoke", action="store_true",
                    help="400-step production smoke, one run per mode")
    ap.add_argument("--acceptance", action="store_true",
                    help="2000-step paired-seed production A/B")
    ap.add_argument("--all", action="store_true",
                    help="synthetic + fast smoke + acceptance")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel production arms (each uses its own NEST kernel)")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--sizes", choices=("fast", "prod", "final"),
                    default=None,
                    help="production sizes; defaults: fast smoke=fast, "
                         "acceptance=prod")
    ap.add_argument("--out", type=str, default="benchmarks/homeostasis_h",
                    help="artifact directory")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    if not (args.synthetic or args.fast_smoke or args.acceptance or args.all):
        args.synthetic = True
    if args.synthetic or args.all:
        _write_synthetic(out)
    if args.fast_smoke or args.all:
        run_fast_smoke(out, jobs=args.jobs,
                       sizes=FAST_SIZES if args.sizes is None else
                       {"fast": FAST_SIZES, "prod": PROD_SIZES,
                        "final": FINAL_SIZES}[args.sizes])
    if args.acceptance or args.all:
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        run_acceptance(out, seeds=seeds, jobs=args.jobs,
                       sizes=PROD_SIZES if args.sizes is None else
                       {"fast": FAST_SIZES, "prod": PROD_SIZES,
                        "final": FINAL_SIZES}[args.sizes])
    _write_artifacts(out)


if __name__ == "__main__":
    main()
