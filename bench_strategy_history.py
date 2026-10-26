"""Strategy-history and noise-recorder benchmark.

Synthetic mode is the logic evidence (no NEST): it compares an exact
applied-sample rebuild of ``F/Xbar/G`` against a post-step parameter trace, a
no-trace variant, and the current merge.  Fast-CPG mode is the production-path
confirmation and runs through ``build_personalizer``.

Usage::

    python bench_strategy_history.py --quick
    python bench_strategy_history.py --noise-config all
    python bench_strategy_history.py --fast-cpg --steps 400 --seeds 0 1
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from persona.noise_recorder import select_producer_names
from persona.strategy import FisherMatrix
from persona.strategy_trace import rebuild_strategy_from_trace

SYNTHETIC_SITES = 6
SYNTHETIC_NOISE = 2
PRODUCER_NAMES = (
    [f"producer:poisson:p{i}" for i in range(10)]
    + [f"producer:stdp:s{i}" for i in range(6)]
)
SYNTHETIC_NOISE_CONFIGS = ("all", "none", "poisson", "stdp",
                           "one-relay:p0")
CPG_NOISE_CONFIGS = ("all", "none", "poisson", "stdp",
                     "one-relay:cut_L")
FAST_EVAL_WINDOW = 50


def _synthetic_stream(steps: int, seed: int):
    rng = np.random.default_rng(seed)
    sigma0 = np.linspace(0.2, 0.8, SYNTHETIC_SITES)
    out = []
    for t in range(steps):
        sigma = sigma0 * (0.95 ** (t / 20.0))
        xhat = rng.normal(0.0, sigma)
        dlog = rng.normal(0.0, 0.01, size=SYNTHETIC_SITES)
        advantage = rng.normal(0.0, 0.4, size=2)
        noise = {f"n{j}": (float(rng.normal(0.0, 0.05)), 1.0)
                 for j in range(SYNTHETIC_NOISE)}
        out.append({"xhat": xhat, "sigma": sigma, "dlog": dlog,
                    "advantage": advantage, "noise": noise})
    return out


def _relative(left, right):
    return float(np.linalg.norm(left - right)
                 / max(float(np.linalg.norm(right)), 1e-12))


def _run_full(stream, *, beta, lambda_reg, warmup, refresh_every, lr_mu):
    names = [f"p{i}" for i in range(SYNTHETIC_SITES)]
    noise_names = [f"n{j}" for j in range(SYNTHETIC_NOISE)]
    fm = FisherMatrix(param_names=names, noise_names=noise_names,
                      beta=beta, lambda_reg=lambda_reg, warmup=warmup,
                      refresh_every=refresh_every, lr_mu=lr_mu)
    mu = np.zeros(SYNTHETIC_SITES)
    rows = []
    for item in stream:
        sites = {name: (item["xhat"][i] - mu[i], item["sigma"][i],
                        item["dlog"][i])
                 for i, name in enumerate(names)}
        delta = fm.step(sites, item["noise"], item["advantage"])
        applied_delta = np.array([delta[name] for name in names])
        rows.append({
            "dtheta": item["xhat"] - mu,
            "applied_delta": applied_delta,
            "sigma": item["sigma"].copy(),
            "dlogsigma": item["dlog"].copy(),
            "advantage": item["advantage"].copy(),
            "noise_delta": np.array([item["noise"][n][0]
                                     for n in noise_names]),
            "noise_scale": np.array([item["noise"][n][1]
                                     for n in noise_names]),
        })
        mu = mu + applied_delta
    return fm, rows, names, noise_names


def _trace_from_rows(rows, names, noise_names, *, beta, lambda_reg,
                     warmup, refresh_every, lr_mu):
    return {
        "dtheta": np.asarray([r["dtheta"] for r in rows]),
        "sigma": np.asarray([r["sigma"] for r in rows]),
        "dlogsigma": np.asarray([r["dlogsigma"] for r in rows]),
        "advantage": np.asarray([r["advantage"] for r in rows]),
        "noise_delta": np.asarray([r["noise_delta"] for r in rows]),
        "noise_scale": np.asarray([r["noise_scale"] for r in rows]),
        "param_names": list(names),
        "noise_names": list(noise_names),
        "beta": beta, "lambda_reg": lambda_reg, "warmup": warmup,
        "refresh_every": refresh_every, "lr_mu": lr_mu,
        "lr_sigma": 0.005, "dt_s": 0.01, "dtype": "float64",
    }


def _rebuild_from_parameter_trace(rows, names, noise_names, *,
                                  beta, lambda_reg, warmup, refresh_every,
                                  lr_mu):
    """Rebuild from what a P-trace actually stores: the applied mean/sigma
    history plus the recorded producer deltas. The sampling-time draw is
    missing, so this is expected to be wrong."""
    fm = FisherMatrix(param_names=names, noise_names=noise_names,
                      beta=beta, lambda_reg=lambda_reg, warmup=warmup,
                      refresh_every=refresh_every, lr_mu=lr_mu)
    for row in rows:
        sites = {
            name: (row["applied_delta"][i], row["sigma"][i],
                   row["dlogsigma"][i])
            for i, name in enumerate(names)
        }
        noise = {
            name: (row["noise_delta"][j], row["noise_scale"][j])
            for j, name in enumerate(noise_names)
        }
        fm._accumulate(*fm._score_vectors(sites, noise),
                       fm._pad_advantage(row["advantage"]))
    fm._refresh_inverse()
    return fm


def _run_no_trace(rows, names, noise_names, *, beta, lambda_reg, warmup,
                  refresh_every, lr_mu):
    fm = FisherMatrix(param_names=names, noise_names=noise_names,
                      beta=beta, lambda_reg=lambda_reg, warmup=warmup,
                      refresh_every=refresh_every, lr_mu=lr_mu)
    for row in rows:
        sites = {
            name: (row["dtheta"][i], row["sigma"][i],
                   row["dlogsigma"][i])
            for i, name in enumerate(names)
        }
        noise = {
            name: (row["noise_delta"][j], row["noise_scale"][j])
            for j, name in enumerate(noise_names)
        }
        score, applied = fm._score_vectors(sites, noise)
        advantage = fm._pad_advantage(row["advantage"])
        fm.F = beta * fm.F + (1.0 - beta) * np.outer(score, score)
        fm.Xbar = applied.copy()
        fm.G = beta * fm.G + (1.0 - beta) * np.outer(applied, advantage)
        fm.steps += 1
    fm._refresh_inverse()
    return fm


def _merge_errors(rows, full, names, noise_names, *, beta, lambda_reg,
                  warmup, refresh_every, lr_mu):
    chunks = np.array_split(np.arange(len(rows)), 3)
    segments = []
    for index in chunks:
        segment = FisherMatrix(
            param_names=names, noise_names=noise_names, beta=beta,
            lambda_reg=lambda_reg, warmup=warmup,
            refresh_every=refresh_every, lr_mu=lr_mu)
        for t in index:
            row = rows[t]
            sites = {name: (row["dtheta"][i], row["sigma"][i],
                            row["dlogsigma"][i])
                     for i, name in enumerate(names)}
            noise = {name: (row["noise_delta"][j], row["noise_scale"][j])
                     for j, name in enumerate(noise_names)}
            segment._accumulate(*segment._score_vectors(sites, noise),
                                segment._pad_advantage(row["advantage"]))
        segments.append(segment)
    merged = FisherMatrix(param_names=names, noise_names=noise_names,
                          beta=beta, lambda_reg=lambda_reg, warmup=warmup,
                          refresh_every=refresh_every, lr_mu=lr_mu)
    merged.merge(segments, [1.0, 1.0, 1.0])
    return {
        "merge_F_error": _relative(merged.F, full.F),
        "merge_Xbar_error": _relative(merged.Xbar, full.Xbar),
        "merge_G_error": _relative(merged.G, full.G),
    }


def run_synthetic(out_dir, *, steps: int = 120, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    beta, lambda_reg, warmup, refresh_every, lr_mu = (
        0.95, 1e-2, 3, 10, 0.1)
    stream = _synthetic_stream(int(steps), int(seed))
    full, rows, names, noise_names = _run_full(
        stream, beta=beta, lambda_reg=lambda_reg, warmup=warmup,
        refresh_every=refresh_every, lr_mu=lr_mu)
    trace = _trace_from_rows(rows, names, noise_names, beta=beta,
                             lambda_reg=lambda_reg, warmup=warmup,
                             refresh_every=refresh_every, lr_mu=lr_mu)
    rebuilt = rebuild_strategy_from_trace(trace)
    trace_error = max(
        _relative(rebuilt.F, full.F),
        _relative(rebuilt.Xbar, full.Xbar),
        _relative(rebuilt.G, full.G),
    )
    p_trace = _rebuild_from_parameter_trace(
        rows, names, noise_names, beta=beta, lambda_reg=lambda_reg,
        warmup=warmup, refresh_every=refresh_every, lr_mu=lr_mu)
    p_trace_error = max(
        _relative(p_trace.F, full.F),
        _relative(p_trace.Xbar, full.Xbar),
        _relative(p_trace.G, full.G),
    )
    no_trace = _run_no_trace(rows, names, noise_names, beta=beta,
                             lambda_reg=lambda_reg, warmup=warmup,
                             refresh_every=refresh_every, lr_mu=lr_mu)
    no_trace_error = max(
        _relative(no_trace.Xbar, full.Xbar),
        _relative(no_trace.G, full.G),
    )
    merge = _merge_errors(rows, full, names, noise_names, beta=beta,
                          lambda_reg=lambda_reg, warmup=warmup,
                          refresh_every=refresh_every, lr_mu=lr_mu)

    noise_rows = []
    for config in SYNTHETIC_NOISE_CONFIGS:
        selected = select_producer_names(PRODUCER_NAMES, config)
        started = time.perf_counter()
        fm = FisherMatrix(param_names=names, noise_names=selected,
                          beta=beta, lambda_reg=lambda_reg, warmup=warmup,
                          refresh_every=refresh_every, lr_mu=lr_mu)
        for item in stream:
            sites = {name: (item["xhat"][i], item["sigma"][i],
                            item["dlog"][i])
                     for i, name in enumerate(names)}
            noise = {name: (item["noise"]["n0"][0], 1.0)
                     for name in selected}
            fm.step(sites, noise, item["advantage"], accumulate_only=True)
        noise_rows.append({
            "config": config,
            "n_noise": len(selected),
            "steps": int(steps),
            "wall_s": float(time.perf_counter() - started),
            "F_noise_norm": float(np.linalg.norm(
                fm.F[2 * len(names):, :])),
            "G_noise_norm": float(np.linalg.norm(
                fm.G[2 * len(names):, :])),
        })

    summary = {
        "steps": int(steps),
        "seed": int(seed),
        "trace_rebuild_error": float(trace_error),
        "parameter_trace_error": float(p_trace_error),
        "no_trace_error": float(no_trace_error),
        **merge,
        "noise_configs": noise_rows,
    }
    (out_dir / "synthetic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "synthetic_runs.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(noise_rows[0]))
        writer.writeheader()
        writer.writerows(noise_rows)
    (out_dir / "synthetic_summary.md").write_text(
        "\n".join([
            "# Strategy-history synthetic probe",
            "",
            f"- steps: {steps}; seed: {seed}",
            f"- exact A-trace rebuild error: {trace_error:.3e}",
            f"- post-step P-trace rebuild error: {p_trace_error:.3e}",
            f"- no-Xbar trace error: {no_trace_error:.3e}",
            f"- merge F/Xbar/G errors: {merge['merge_F_error']:.3e} / "
            f"{merge['merge_Xbar_error']:.3e} / "
            f"{merge['merge_G_error']:.3e}",
            "",
            "The exact A-trace rebuild is the reference. The P-trace and "
            "no-Xbar variants are expected to be far worse; the merge errors "
            "show that the current merge is an ensemble average, not exact "
            "continuation.",
        ]) + "\n", encoding="utf-8")
    return summary


def _fast_cpg_task(args):
    from bench_homeostasis import FAST_SIZES
    from personalize import build_personalizer

    steps, seed, noise_config = args
    started = time.perf_counter()
    try:
        pers, _ = build_personalizer(
            max_steps=int(steps), eps_pose=0.10, pose_hold_steps=3,
            loss_stop=-1.0, loss_window=20, ref_hz=700.0,
            resolution_ms=0.2, seed=int(seed), game=None,
            homeostasis=None, noise_config=noise_config,
            trace_strategy=True, **FAST_SIZES)
        agent = pers.agent
        pers.verbose = False
        pers.log_every = 0
        pers.stdp_every = 0
        pers.belt_mps = 0.2
        pers.belt_start_step = 50
        pers.belt_ramp_steps = 50
        log = pers.run(pose_max_steps=100, realize_every_k=1)
        trace = agent.strategy_trace_arrays()
        rebuilt = rebuild_strategy_from_trace(trace)
        strategy = agent.strategy
        n_sites = strategy.n_sites
        n_noise = strategy.n_noise
        noise_F = float(np.linalg.norm(strategy.F[2 * n_sites:, :]))
        noise_G = float(np.linalg.norm(strategy.G[2 * n_sites:, :]))
        F_norm = float(np.linalg.norm(strategy.F))
        errors = {
            "F": _relative(rebuilt.F, strategy.F),
            "Xbar": _relative(rebuilt.Xbar, strategy.Xbar),
            "G": _relative(rebuilt.G, strategy.G),
        }
        eval_start = max(0, len(log.loss) - min(FAST_EVAL_WINDOW,
                                               len(log.loss)))
        return {
            "noise_config": noise_config,
            "seed": int(seed),
            "steps": int(len(log.loss)),
            "n_sites": int(n_sites),
            "n_noise": int(n_noise),
            "loss_eval": float(np.mean(log.loss[eval_start:])),
            "noise_F_norm": noise_F,
            "noise_F_rel": noise_F / max(F_norm, 1e-12),
            "noise_G_norm": noise_G,
            "trace_F_error": errors["F"],
            "trace_Xbar_error": errors["Xbar"],
            "trace_G_error": errors["G"],
            "recorders": len(agent.registry.producer_recorders()),
            "wall_s": float(time.perf_counter() - started),
            "error": None,
        }
    except Exception as exc:  # surfaced, never swallowed
        return {
            "noise_config": noise_config,
            "seed": int(seed),
            "steps": int(steps),
            "n_sites": None,
            "n_noise": None,
            "loss_eval": None,
            "noise_F_norm": None,
            "noise_F_rel": None,
            "noise_G_norm": None,
            "trace_F_error": None,
            "trace_Xbar_error": None,
            "trace_G_error": None,
            "recorders": None,
            "wall_s": float(time.perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_fast_cpg(out_dir, *, steps: int = 400, seeds=(0, 1),
                 noise_configs=CPG_NOISE_CONFIGS, fresh_process: bool = True
                 ) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(int(steps), int(seed), str(cfg))
             for seed in seeds for cfg in noise_configs]
    rows = []
    for task in tasks:
        if fresh_process:
            with ProcessPoolExecutor(max_workers=1) as pool:
                row = pool.submit(_fast_cpg_task, task).result()
        else:
            row = _fast_cpg_task(task)
        rows.append(row)
        print(f"[fast-cpg] seed={row['seed']} config={row['noise_config']} "
              f"loss={row['loss_eval']} n_noise={row['n_noise']} "
              f"trace_err={row['trace_F_error']} err={row['error']}",
              flush=True)
    (out_dir / "cpg_runs.csv").write_text(
        "\n".join([",".join(rows[0].keys())]
                  + [",".join("" if v is None else str(v)
                              for v in row.values()) for row in rows]) + "\n",
        encoding="utf-8")
    (out_dir / "cpg_summary.json").write_text(
        json.dumps({"steps": int(steps), "seeds": list(seeds),
                    "noise_configs": list(noise_configs),
                    "runs": rows}, indent=2), encoding="utf-8")
    return {"runs": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fast-cpg", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--noise-config", action="append", default=None,
                        help="repeatable; defaults to the full noise matrix")
    parser.add_argument("--out", type=Path,
                        default=Path("benchmarks/strategy_history"))
    args = parser.parse_args(argv)
    if args.quick:
        summary = run_synthetic(args.out, steps=20, seed=0)
        print(json.dumps(summary, indent=2))
        return summary
    if args.fast_cpg:
        configs = tuple(args.noise_config or CPG_NOISE_CONFIGS)
        return run_fast_cpg(args.out, steps=args.steps,
                            seeds=tuple(args.seeds), noise_configs=configs)
    summary = run_synthetic(args.out, steps=args.steps, seed=0)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
