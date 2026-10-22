"""A/B harness for the conserved-stack game.

Opt-in game arms plus the H-off baseline, same production path, seeds, sizes,
and 1000-step evaluation window as ``bench_homeostasis.py``:

    off          no regulator
    prop_none    proportional, no base seat (the first-comparison baseline)
    prop_uni_05  proportional, uniform base seat, liveness 0.05
    prop_uni_10  proportional, uniform base seat, liveness 0.10
    prop_uni_20  proportional, uniform base seat, liveness 0.20
    prop_unc_05  proportional, uncertainty base seat, liveness 0.05
    prop_unc_10  proportional, uncertainty base seat, liveness 0.10
    prop_unc_20  proportional, uncertainty base seat, liveness 0.20

Legacy names (``prop_pot``, ``inv_none``, ``inv_pot``) are still accepted.

``r_p90`` is a diagnostic here, not the objective. The comparison records
stability, loss, type/sign violations, active players, and the game's own stack
diagnostics.

Usage::

    python bench_game.py --smoke
    python bench_game.py --compare --jobs 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from bench_homeostasis import (
    FAST_SIZES,
    PROD_SIZES,
    _phase_agreement,
    inflation_metrics,
    type_sign_violations,
)
from persona.game import GameConfig
from persona.homeostasis import HomeostaticConfig


ARMS = ("off", "prop_none",
        "prop_uni_05", "prop_uni_10", "prop_uni_20",
        "prop_unc_05", "prop_unc_10", "prop_unc_20")
STEPS = 2000
EVAL_STEPS = 1000
POT_FEE = 0.05


_H_MODES = ("advantage", "budget", "fisher", "band")


def _kind(arm: str) -> str:
    return "homeostasis" if arm.startswith("h_") else "game"


def _h_config(arm: str) -> HomeostaticConfig | None:
    if arm == "h_best":
        cfg = HomeostaticConfig.current_best()
        cfg.h_delay_steps = 0          # activate within a fast run
        cfg.h_burnin_steps = 100
        return cfg
    if arm == "h_best_delay":
        cfg = HomeostaticConfig.current_best()
        cfg.h_burnin_steps = 100
        return cfg
    if arm.startswith("h_"):
        mode = arm[2:]
        if mode not in _H_MODES:
            raise ValueError(f"unknown H arm {arm!r}")
        return HomeostaticConfig(
            enabled=True,
            global_variance_mode=mode,
            h_burnin_steps=100,
            lambda_H=1.0,
        )
    raise ValueError(f"unknown H arm {arm!r}")


def _game_config(arm: str) -> GameConfig | None:
    if arm == "off":
        return None
    if arm in ("selected", "a25"):
        return GameConfig.current_best()
    parts = arm.split("_")
    mode = "proportional" if parts[0] == "prop" else "inverse"
    if len(parts) == 2 and parts[1] == "none":
        return GameConfig(enabled=True, throttle_mode=mode, burnin_steps=100)
    if parts[1] == "pot":
        return GameConfig(enabled=True, throttle_mode=mode,
                          movement_fee=POT_FEE, burnin_steps=100)
    seat = {"uni": "uniform", "unc": "uncertainty"}.get(parts[1])
    if seat is None:
        raise ValueError(f"unknown arm {arm!r}")
    kwargs = {}
    for token in parts[3:]:
        if token.startswith("a") and token[1:].isdigit():
            kwargs["throttle_exponent"] = int(token[1:]) / 10.0
        elif token.startswith("h") and token[1:].isdigit():
            kwargs["transfer_rate"] = int(token[1:]) / 100.0
        else:
            raise ValueError(f"unknown arm token {token!r} in {arm!r}")
    return GameConfig(
        enabled=True,
        throttle_mode=mode,
        base_seat=seat,
        liveness_fraction=int(parts[2]) / 100.0,
        burnin_steps=100,
        **kwargs,
    )


def run_arm(arm: str, seed: int, *, steps: int = STEPS,
            eval_steps: int = EVAL_STEPS, sizes=None,
            kind: str = "game") -> dict:
    """One production-path run. Imports NEST/MuJoCo lazily."""
    from personalize import build_personalizer

    sizes = dict(PROD_SIZES if sizes is None else sizes)
    game_cfg = _game_config(arm) if kind == "game" else None
    h_cfg = _h_config(arm) if kind == "homeostasis" else None
    pers, _ = build_personalizer(
        max_steps=int(steps), eps_pose=0.10, pose_hold_steps=3,
        loss_stop=-1.0, loss_window=20, ref_hz=700.0, resolution_ms=0.2,
        seed=int(seed), game=game_cfg, homeostasis=h_cfg, **sizes)
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
    regulator = (agent.game if agent.game is not None
                 else agent.homeostasis)
    if regulator is not None:
        for spec in regulator.physical_map.specs:
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

    gs = agent.history.get("game", [])
    window = gs[eval_start:]
    stacks = [np.asarray(d["stacks"], dtype=float) for d in window]
    if stacks:
        stack_entropy = float(np.mean([
            float(d["entropy_norm"]) for d in window]))
        stack_min = float(np.mean([float(s.min()) for s in stacks]))
        stack_max = float(np.mean([float(s.max()) for s in stacks]))
        throttle_min = float(np.mean([
            float(d["throttle_min"]) for d in window]))
        throttle_max = float(np.mean([
            float(d["throttle_max"]) for d in window]))
        pot_mean = float(np.mean([float(d["pot"]) for d in window]))
        transfer_abs = float(np.mean([
            float(np.abs(np.asarray(d["transfer"], dtype=float)).sum())
            for d in window]))
        active_players = int(np.sum(stacks[-1] > 1e-9))
    else:
        stack_entropy = stack_min = stack_max = None
        throttle_min = throttle_max = pot_mean = transfer_abs = None
        active_players = None

    hs = agent.history.get("homeostasis", [])
    hw = hs[eval_start:]
    if hw:
        share_traj = [float(np.linalg.norm(
            np.asarray(d.get("e", []), dtype=float))) for d in hw]
        share_rms = (float(np.sqrt(np.mean(np.square(share_traj))))
                     if share_traj else None)
        ts = [float(d.get("T", np.nan)) for d in hw]
        T_min = float(np.nanmin(ts)) if ts else None
        T_max = float(np.nanmax(ts)) if ts else None
    else:
        share_rms = T_min = T_max = None

    return {
        "mode": arm,
        "throttle": None if game_cfg is None else game_cfg.throttle_mode,
        "movement_fee": (None if game_cfg is None
                         else game_cfg.movement_fee),
        "h_delay_steps": None if h_cfg is None else h_cfg.h_delay_steps,
        "lift_mode": None if h_cfg is None else h_cfg.lift_mode,
        "seed": int(seed),
        "steps": n_steps,
        "eval_steps": n_steps - eval_start,
        "n_players": (None if regulator is None
                      else int(regulator.physical_map.n_players)),
        "loss_eval": float(np.mean(log.loss[eval_start:])),
        "loss_full": float(np.mean(log.loss[agent.warmup:])),
        "wall_s": float(wall_s),
        "step_ms": float(1000.0 * wall_s / max(n_steps, 1)),
        "violations": int(len(bad)),
        "violation_names": bad[:10],
        "phases": [str(p) for p in agent.history["phase"]],
        "stage": str(log.stage[-1]) if log.stage else "unknown",
        "sizes": sizes,
        "stack_entropy": stack_entropy,
        "stack_min": stack_min,
        "stack_max": stack_max,
        "throttle_min": throttle_min,
        "throttle_max": throttle_max,
        "pot_mean": pot_mean,
        "transfer_abs_mean": transfer_abs,
        "active_players": active_players,
        "share_rms": share_rms,
        "T_min": T_min,
        "T_max": T_max,
        "stack_final": ("" if not stacks else
                        ";".join(f"{v:.6g}" for v in stacks[-1])),
        **metrics,
    }


def _task(args):
    return run_arm(**args)


def _run_fresh(task):
    """Run one arm in a process that has not run another arm.

    NEST/MuJoCo leave enough state behind that reusing a worker changes the
    baseline by a few percent; one fresh process per arm removes that
    confound.
    """
    with ProcessPoolExecutor(max_workers=1) as pool:
        return pool.submit(_task, task).result()


def _run_tasks(tasks, jobs):
    out = []

    def record(row):
        out.append(row)
        print(f"[arm] seed={row['seed']} mode={row['mode']} "
              f"loss={row['loss_eval']:.5g} p90={row['r_p90']:.4g} "
              f"wall={row['wall_s']:.1f}s", flush=True)

    if jobs <= 1:
        for task in tasks:
            record(_run_fresh(task))
        return out
    with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
        futures = [pool.submit(_run_fresh, task) for task in tasks]
        for fut in as_completed(futures):
            record(fut.result())
    return out


def _summary(rows, seeds, arms):
    by = {(r["seed"], r["mode"]): r for r in rows}
    out = []
    for arm in arms:
        if arm == "off":
            continue
        p90 = []
        loss = []
        q = []
        for seed in seeds:
            base = by[(seed, "off")]
            cur = by[(seed, arm)]
            p90.append(1.0 - cur["r_p90"] / max(base["r_p90"], 1e-300))
            loss.append(cur["loss_eval"] / max(base["loss_eval"], 1e-300)
                        - 1.0)
            q.append(cur["q_gt_1"] - base["q_gt_1"])
        finite = all(
            np.isfinite(by[(seed, arm)]["loss_eval"])
            and np.isfinite(by[(seed, arm)]["r_p90"])
            for seed in seeds)
        out.append({
            "arm": arm,
            "mean_p90_improvement": float(np.mean(p90)),
            "min_p90_improvement": float(np.min(p90)),
            "mean_loss_delta": float(np.mean(loss)),
            "max_loss_delta": float(np.max(loss)),
            "q_growth_max": int(np.max(q)),
            "violations": int(sum(by[(seed, arm)]["violations"]
                                  for seed in seeds)),
            "finite": bool(finite),
            "loss_ok": bool(np.max(loss) <= 0.10),
            "p90_ok": bool(np.mean(p90) > 0.0),
            "viable": bool(finite and np.max(loss) <= 0.10
                           and np.mean(p90) > 0.0),
            "active_players": _mean_metric(by, seeds, arm, "active_players"),
            "stack_entropy": _mean_metric(by, seeds, arm, "stack_entropy"),
            "stack_min": _mean_metric(by, seeds, arm, "stack_min"),
            "stack_max": _mean_metric(by, seeds, arm, "stack_max"),
            "throttle_min": _mean_metric(by, seeds, arm, "throttle_min"),
            "throttle_max": _mean_metric(by, seeds, arm, "throttle_max"),
            "pot_mean": _mean_metric(by, seeds, arm, "pot_mean"),
            "transfer_abs_mean": _mean_metric(
                by, seeds, arm, "transfer_abs_mean"),
        })
    return out


def _mean_metric(by, seeds, arm, key):
    values = [by[(seed, arm)].get(key) for seed in seeds]
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _write_csv(path, rows):
    flat = []
    for row in rows:
        out = {k: v for k, v in row.items()
               if k not in ("phases", "violation_names", "sizes")}
        out["violation_names"] = ";".join(row.get("violation_names", []))
        out["sizes"] = json.dumps(row.get("sizes", {}))
        flat.append(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def _write_markdown(path, summary, rows, order=None):
    order = tuple(ARMS if order is None else order)
    lines = [
        "# Conserved-stack game, 20 s comparison",
        "",
        "p90 is a diagnostic, not the objective. `viable` means no seed loses",
        "more than 10% of `loss_eval` and mean p90 improves.",
        "",
        "| arm | mean p90 | min p90 | mean loss | max loss | q max | viol |"
        " active | entropy | stack min | stack max | pot | viable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in summary:
        lines.append(
            f"| {m['arm']} | {100*m['mean_p90_improvement']:.1f}% | "
            f"{100*m['min_p90_improvement']:.1f}% | "
            f"{100*m['mean_loss_delta']:.1f}% | "
            f"{100*m['max_loss_delta']:.1f}% | {m['q_growth_max']} | "
            f"{m['violations']} | {_fmt(m['active_players'], 4)} | "
            f"{_fmt(m['stack_entropy'])} | "
            f"{_fmt(m['stack_min'], 5)} | {_fmt(m['stack_max'], 5)} | "
            f"{_fmt(m['pot_mean'], 5)} | {m['viable']} |")
    lines += [
        "",
        "## Raw runs",
        "",
        "| seed | arm | steps | loss_eval | p90 | at_bound | q>1 | viol |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["seed"], order.index(x["mode"]))):
        lines.append(
            f"| {r['seed']} | {r['mode']} | {r['steps']} | "
            f"{r['loss_eval']:.5g} | {r['r_p90']:.4g} | {r['at_bound']} | "
            f"{r['q_gt_1']} | {r['violations']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}g}"


def run_compare(out_dir: Path, *, seeds=(0, 1), steps=STEPS,
                eval_steps=EVAL_STEPS, sizes=None, jobs=1, arms=None,
                prefix="real_compare_20s"):
    arms = tuple(ARMS if arms is None else arms)
    sizes = dict(PROD_SIZES if sizes is None else sizes)
    tasks = [dict(arm=arm, seed=seed, steps=steps, eval_steps=eval_steps,
                  sizes=sizes, kind=_kind(arm))
             for seed in seeds for arm in arms]
    rows = _run_tasks(tasks, jobs)
    by_seed = {seed: next(r["phases"] for r in rows
                          if r["seed"] == seed and r["mode"] == "off")
               for seed in seeds}
    for r in rows:
        r["phase_agreement"] = (
            _phase_agreement(r["phases"], by_seed[r["seed"]])
            if r["mode"] != "off" else 1.0)
    summary = _summary(rows, seeds, arms)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / f"{prefix}.csv", rows)
    (out_dir / f"{prefix}.json").write_text(
        json.dumps({"steps": int(steps), "eval_steps": int(eval_steps),
                    "seeds": list(seeds), "sizes": sizes,
                    "summary": summary,
                    "runs": [{k: v for k, v in r.items()
                              if k not in ("phases", "sizes")}
                             for r in rows]}, indent=2),
        encoding="utf-8")
    _write_markdown(out_dir / f"{prefix}.md", summary, rows, order=arms)

    for m in summary:
        print(f"[compare] {m['arm']}: "
              f"p90 mean={100*m['mean_p90_improvement']:.1f}% "
              f"min={100*m['min_p90_improvement']:.1f}% "
              f"loss mean={100*m['mean_loss_delta']:.1f}% "
              f"max={100*m['max_loss_delta']:.1f}% "
              f"q_max={m['q_growth_max']} viol={m['violations']} "
              f"entropy={_fmt(m['stack_entropy'])} "
              f"viable={m['viable']}", flush=True)
    return {"summary": summary, "runs": rows}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--prefix", default="real_compare_20s")
    parser.add_argument("--out", type=Path,
                        default=Path("benchmarks/game"))
    args = parser.parse_args(argv)

    if args.smoke:
        result = run_compare(args.out / "smoke", seeds=(0,), steps=400,
                             eval_steps=200, sizes=FAST_SIZES,
                             jobs=args.jobs, arms=args.arms,
                             prefix="smoke")
        print(json.dumps({"smoke": True,
                          "all_finite": all(m["finite"]
                                            for m in result["summary"]),
                          "violations": sum(m["violations"]
                                            for m in result["summary"])},
                         indent=2))
        return result
    result = run_compare(args.out, seeds=tuple(args.seeds),
                         steps=args.steps, eval_steps=args.eval_steps,
                         jobs=args.jobs, arms=args.arms,
                         prefix=args.prefix)
    return result


if __name__ == "__main__":
    main()
