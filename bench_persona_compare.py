"""Toy comparison harness: PERSONA/Evo-style game, NG variants, CMA-ES, surrogate.

The primary comparison is the 14-parameter toy in
``benchmarks/compare/toy_cpg.py``. Every arm gets 200 episodes x 100 steps =
20,000 environment steps and a common held-out tail evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from benchmarks.compare.cmaes_arm import run_cmaes
from benchmarks.compare.surrogate import surrogate_loss_and_grad
from benchmarks.compare.toy_cpg import (
    BOUNDS,
    INITIAL,
    PARAM_NAMES,
    ToyCPG,
    ToyConfig,
)
from benchmarks.compare.variants import VariantFisherMatrix
from persona.game import GameConfig, StackGame
from persona.homeostasis import PhysicalMap, PlayerSpec
from persona.parameter import PersonaParam, reset_rng_stream
from persona.strategy import FisherMatrix

PLAYER_GROUPS = {
    "rg_e": ("w_tonic_e", "w_rec_e", "w_rg_to_mn_e", "tau_m_e", "bias_e"),
    "rg_f": ("w_tonic_f", "w_rec_f", "w_rg_to_mn_f", "tau_m_f", "bias_f"),
    "output": ("gain_e", "gain_f"),
    "coupling": ("w_e_to_f", "w_f_to_e"),
}
ARMS = ("persona_game", "persona_nogame", "persona_diag", "persona_block",
        "cmaes", "surrogate")
TOY_LR_MU = 0.5
TOY_LOSS_SCALE = 20.0
CPG_ARMS = ("persona_manual", "persona_nogame", "persona_diag",
            "persona_block", "persona_evo", "cmaes_sub")


def _sigma(name: str) -> float:
    low, high = BOUNDS[name]
    return max(0.01, 0.05 * (high - low))


def _initial_for_seed(seed: int) -> dict:
    """Per-seed initial vector, paired across arms within a seed."""
    rng = np.random.default_rng(int(seed) ^ 0x5EED)
    initial = {}
    for name in PARAM_NAMES:
        low, high = BOUNDS[name]
        spread = 0.05 * (high - low)
        initial[name] = float(np.clip(
            INITIAL[name] + rng.uniform(-spread, spread), low, high))
    return initial


def _params(seed: int = 0, initial: dict | None = None) -> dict:
    reset_rng_stream(int(seed))
    values = dict(initial or _initial_for_seed(seed))
    return {
        name: PersonaParam(mean=values[name], sigma=_sigma(name),
                           min=BOUNDS[name][0], max=BOUNDS[name][1])
        for name in PARAM_NAMES
    }


def _blocks():
    index = {name: i for i, name in enumerate(PARAM_NAMES)}
    n = len(PARAM_NAMES)
    return [np.asarray([index[name] for name in group]
                       + [n + index[name] for name in group], dtype=int)
            for group in PLAYER_GROUPS.values()]


def _strategy(arm: str, seed: int):
    game = None
    if arm == "persona_game":
        specs = [PlayerSpec(name, "population", tuple(group), size=1)
                 for name, group in PLAYER_GROUPS.items()]
        physical_map = PhysicalMap.from_specs(specs)
        game = StackGame(physical_map, GameConfig.current_best(),
                         dt_s=0.01, beta=0.995, site_names=list(PARAM_NAMES))
    if arm == "persona_diag":
        return VariantFisherMatrix(
            param_names=list(PARAM_NAMES), noise_names=[], beta=0.995,
            lambda_reg=1e-3, warmup=3, lr_mu=TOY_LR_MU,
            preconditioner="diag")
    if arm == "persona_block":
        return VariantFisherMatrix(
            param_names=list(PARAM_NAMES), noise_names=[], beta=0.995,
            lambda_reg=1e-3, warmup=3, lr_mu=TOY_LR_MU,
            preconditioner="block", blocks=_blocks())
    return FisherMatrix(
        param_names=list(PARAM_NAMES), noise_names=[], beta=0.995,
        lambda_reg=1e-3, warmup=3, refresh_every=10, lr_mu=TOY_LR_MU, game=game)


def _run_online(arm: str, seed: int, episodes: int = 200) -> dict:
    started = time.perf_counter()
    toy = ToyCPG(ToyConfig(steps=100, seed=seed))
    initial = _initial_for_seed(seed)
    params = _params(seed, initial)
    strategy = _strategy(arm, seed)
    curve = []
    best_curve = float("inf")
    best_params = None
    window_e = deque(maxlen=10)
    window_f = deque(maxlen=10)
    for episode in range(int(episodes)):
        toy.reset(seed=seed + episode)
        episode_errors = []
        episode_torques = []
        for _ in range(toy.config.steps):
            snapshot = {name: (float(param.mean), float(param.sigma))
                        for name, param in params.items()}
            actions = {name: float(param.sample_exploration())
                       for name, param in params.items()}
            row = toy.step(actions)
            error = float(row["error2"])
            episode_torques.append(float(row["torque"]))
            if row["phase"] == "E":
                window_e.append(error)
            else:
                window_f.append(error)
            loss_vec = TOY_LOSS_SCALE * np.asarray([
                float(np.mean(window_e)) if window_e else 0.0,
                float(np.mean(window_f)) if window_f else 0.0,
            ])
            advantage = strategy.compute_advantage(loss_vec, phase=("E", "F"))
            apply_updates = strategy.steps + 1 > strategy.warmup
            sites = {}
            for name, param in params.items():
                mean_at, sigma_at = snapshot[name]
                sample = actions[name]
                dlog = 0.0
                if apply_updates:
                    dlog = param.sigma_step_applied(
                        sample=sample, mu_at_sample=mean_at,
                        sigma_at_sample=sigma_at,
                        advantage=float(np.mean(advantage)),
                        lr_sigma=0.005)
                sites[name] = (sample - mean_at, sigma_at, dlog)
            deltas = strategy.step(sites, {}, advantage)
            if apply_updates:
                for name, param in params.items():
                    mean_at, sigma_at = snapshot[name]
                    param.update_mean(deltas[name])
                    param.update_sigma(
                        sample=actions[name], mu_at_sample=mean_at,
                        sigma_at_sample=sigma_at,
                        advantage=float(np.mean(advantage)),
                        lr_sigma=0.005)
            episode_errors.append(error)
        tail = slice(len(episode_errors) // 2, len(episode_errors))
        episode_loss = float(np.mean(episode_errors[tail])
                             + 0.01 * np.mean(
                                 np.asarray(episode_torques[tail]) ** 2))
        curve.append(episode_loss)
        if episode_loss < best_curve:
            best_curve = episode_loss
            best_params = {name: float(param.mean)
                           for name, param in params.items()}
    final = {name: float(param.mean) for name, param in params.items()}
    held_final = float(toy.held_out_loss(final))
    held_best = (float(toy.held_out_loss(best_params))
                 if best_params is not None else held_final)
    return {
        "arm": arm, "seed": int(seed), "episodes": int(episodes),
        "held_out_loss": held_final,
        "final_held_out_loss": held_final,
        "best_held_out_loss": held_best,
        "final_curve": float(curve[-1]) if curve else float("nan"),
        "best_curve": float(best_curve),
        "curve": curve, "wall_s": float(time.perf_counter() - started),
        "final_params": final, "best_params": best_params,
        "initial_params": initial,
    }


def _run_offline(arm: str, seed: int, episodes: int = 200) -> dict:
    started = time.perf_counter()
    toy = ToyCPG(ToyConfig(steps=100, seed=seed))
    initial = _initial_for_seed(seed)
    if arm == "cmaes":
        result = run_cmaes(
            lambda params: float(toy.rollout(params)["tail_loss"]),
            BOUNDS, initial, n_trials=int(episodes), seed=seed)
        final = dict(result["best_params"])
        best_params = dict(result["best_params"])
        best_loss = float(result["best_value"])
        values = result["values"]
        best = []
        current = float("inf")
        for value in values:
            current = min(current, float(value))
            best.append(current)
        curve = best
    elif arm == "surrogate":
        params = dict(initial)
        m = {name: 0.0 for name in PARAM_NAMES}
        v = {name: 0.0 for name in PARAM_NAMES}
        curve = []
        best_loss = float("inf")
        best_params = dict(params)
        for epoch in range(int(episodes)):
            config = ToyConfig(steps=100, seed=seed + epoch)
            loss, grad = surrogate_loss_and_grad(params, config)
            for index, name in enumerate(PARAM_NAMES):
                g = float(grad[index])
                m[name] = 0.9 * m[name] + 0.1 * g
                v[name] = 0.999 * v[name] + 0.001 * g * g
                m_hat = m[name] / (1.0 - 0.9 ** (epoch + 1))
                v_hat = v[name] / (1.0 - 0.999 ** (epoch + 1))
                params[name] -= 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
                params[name] = float(np.clip(params[name], *BOUNDS[name]))
            curve.append(float(loss))
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_params = dict(params)
        final = params
    else:
        raise ValueError(f"unknown offline arm {arm!r}")
    held_final = float(toy.held_out_loss(final))
    held_best = float(toy.held_out_loss(best_params))
    return {
        "arm": arm, "seed": int(seed), "episodes": int(episodes),
        "held_out_loss": held_final,
        "final_held_out_loss": held_final,
        "best_held_out_loss": held_best,
        "final_curve": float(curve[-1]) if curve else float("nan"),
        "best_curve": float(best_loss),
        "curve": curve, "wall_s": float(time.perf_counter() - started),
        "final_params": final, "best_params": best_params,
        "initial_params": initial,
    }


def run_toy(seeds=(0, 1, 2, 3, 4), arms=ARMS, episodes: int = 200,
            out_dir=None) -> dict:
    rows = []
    curves = {}
    for seed in seeds:
        for arm in arms:
            if arm in ("cmaes", "surrogate"):
                result = _run_offline(arm, seed, episodes)
            elif arm in ARMS:
                result = _run_online(arm, seed, episodes)
            else:
                raise ValueError(f"unknown arm {arm!r}")
            rows.append({key: value for key, value in result.items()
                         if key not in ("curve", "final_params")})
            curves[f"{arm}:{seed}"] = result["curve"]
            print(f"[toy] seed={seed} arm={arm} "
                  f"held={result['held_out_loss']:.5g} "
                  f"wall={result['wall_s']:.1f}s", flush=True)
    summary = _summarize(rows)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs(out_dir, rows, summary, curves)
    return {"rows": rows, "summary": summary, "curves": curves}


def _summarize(rows):
    summary = []
    for arm in sorted({row["arm"] for row in rows}):
        group = [row for row in rows if row["arm"] == arm]
        summary.append({
            "arm": arm,
            "n": len(group),
            "mean_best_held_out_loss": float(np.mean(
                [row["best_held_out_loss"] for row in group])),
            "std_best_held_out_loss": float(np.std(
                [row["best_held_out_loss"] for row in group])),
            "mean_final_held_out_loss": float(np.mean(
                [row["final_held_out_loss"] for row in group])),
            "mean_final_curve": float(np.mean(
                [row["final_curve"] for row in group])),
            "mean_best_curve": float(np.mean(
                [row["best_curve"] for row in group])),
            "mean_wall_s": float(np.mean([row["wall_s"] for row in group])),
        })
    return summary


def _write_outputs(out_dir, rows, summary, curves):
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(rows[0])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields}
                          for row in rows])
    with (out_dir / "summary.csv").open("w", newline="",
                                       encoding="utf-8") as fh:
        fields = list(summary[0])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "summary.json").write_text(
        json.dumps({"runs": rows, "summary": summary}, indent=2,
                   default=str), encoding="utf-8")
    (out_dir / "curves.json").write_text(
        json.dumps(curves, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for key, curve in curves.items():
            ax.plot(curve, label=key, linewidth=1.0)
        ax.set_xlabel("episode / trial / epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _cpg_player_blocks(registry, agent):
    from persona.physo_reference import level2_labels
    from persona.segments import SegmentMatrix

    matrix = SegmentMatrix.from_registry(registry)
    groups: dict[str, list[int]] = {}
    for segment, label in level2_labels(matrix).items():
        groups.setdefault(label, []).append(segment)
    index = {name: i for i, name in enumerate(agent.params)}
    n = len(agent.params)
    blocks = []
    for group in groups.values():
        indices = []
        for segment in group:
            for site in matrix.segments[segment].param_names:
                if site in index:
                    indices.extend([index[site], n + index[site]])
        if indices:
            blocks.append(np.asarray(indices, dtype=int))
    return blocks


def _cpg_build(arm: str, seed: int, steps: int, player_map=None,
               run: bool = True):
    from bench_homeostasis import FAST_SIZES
    from persona.game import GameConfig
    from personalize import build_personalizer

    game = (GameConfig.current_best()
            if arm in ("persona_manual", "persona_evo") else None)
    pers, _ = build_personalizer(
        max_steps=int(steps), eps_pose=0.10, pose_hold_steps=3,
        loss_stop=-1.0, loss_window=20, ref_hz=700.0, resolution_ms=0.2,
        seed=int(seed), game=game, homeostasis=None, player_map=player_map,
        **FAST_SIZES)
    agent = pers.agent
    if arm in ("persona_diag", "persona_block"):
        blocks = _cpg_player_blocks(pers.registry, agent) \
            if arm == "persona_block" else None
        agent.strategy = VariantFisherMatrix(
            param_names=list(agent.params), noise_names=[],
            beta=agent.beta, lambda_reg=1e-3, warmup=agent.warmup,
            lr_mu=0.02, preconditioner=("diag" if arm == "persona_diag"
                                        else "block"), blocks=blocks)
        agent.game = None
    pers.verbose = False
    pers.log_every = 0
    pers.stdp_every = 0
    pers.belt_mps = 0.2
    pers.belt_start_step = 50
    pers.belt_ramp_steps = 50
    log = pers.run(pose_max_steps=100, realize_every_k=1) if run else None
    return pers, agent, log


def bio_table(metrics: dict) -> dict:
    """Per-criterion bio table; sole-up is not logged by RunLog."""
    return {
        "phase_gate": metrics.get("phase_passed"),
        "biological_acceptable": metrics.get("biological_acceptable"),
        "angle_coverage": metrics.get("angle_coverage"),
        "duty_factor_in_distribution": metrics.get(
            "duty_factor_in_distribution"),
        "violations": metrics.get("violations"),
        "at_bound": metrics.get("at_bound"),
        "q_gt_1": metrics.get("q_gt_1"),
        "sole_up": "not measured (not logged)",
    }


def _cpg_metrics(agent, log) -> dict:
    from bench_homeostasis import inflation_metrics, type_sign_violations
    from loader.stats import LocomotionMetrics
    from persona.phase_metrics import phase_gate_metrics
    from personalize import acceptance_from_log

    initial = {name: float(param.mean) for name, param in agent.params.items()}
    bounds = {name: (float(param.min), float(param.max))
              for name, param in agent.params.items()}
    ptypes = {}
    if agent.game is not None:
        for spec in agent.game.physical_map.specs:
            for site in spec.param_names:
                ptypes[site] = spec.ptype
    eval_start = max(0, len(log.loss) - 50)
    metrics = {
        "loss_eval": float(np.mean(log.loss[eval_start:])),
        "loss_full": float(np.mean(log.loss)),
        "n_steps": int(len(log.loss)),
    }
    metrics.update(inflation_metrics(agent.history["means"], initial, bounds,
                                     eval_start=eval_start))
    metrics["violations"] = int(len(type_sign_violations(
        agent.history["means"], bounds, ptypes)))
    game = agent.history.get("game", [])
    if game:
        metrics["active_players"] = int(np.sum(
            np.asarray(game[-1]["stacks"]) > 1e-12))
        metrics["stack_entropy"] = float(game[-1].get("entropy_norm", 0.0))
    else:
        metrics["active_players"] = None
        metrics["stack_entropy"] = None
    try:
        log_map = {name: getattr(log, name)
                   for name in type(log)._NUMERIC + type(log)._STR}
        phase = phase_gate_metrics(log_map, dt_ms=10.0)
        metrics.update({
            "phase_passed": bool(phase.passed),
            "phase_failures": list(phase.failures),
            "phase_cycles": phase.metrics.get("cycles"),
            "phase_period_cv": phase.metrics.get("period_cv"),
            "phase_ef": phase.metrics.get("ef_phase"),
            "phase_lr_drift": phase.metrics.get("lr_drift"),
            "phase_angle_corr": phase.metrics.get("angle_corr"),
        })
        acceptance = acceptance_from_log(log, None, margin=0.05,
                                         coverage=0.70)
        metrics.update({
            "biological_acceptable": bool(acceptance["acceptable"]),
            "angle_coverage": float(acceptance["angle_coverage"]),
            "duty_factor_in_distribution": bool(
                acceptance["duty_factor_in_distribution"]),
        })
        contact_L = np.asarray(
            [1.0 if value == "stance" else 0.0 for value in log.phase_L])
        contact_R = np.asarray(
            [1.0 if value == "stance" else 0.0 for value in log.phase_R])
        foot_L = np.column_stack([log.foot_x_m_L,
                                  np.zeros(len(log.foot_x_m_L)),
                                  log.foot_z_m_L])
        foot_R = np.column_stack([log.foot_x_m_R,
                                  np.zeros(len(log.foot_x_m_R)),
                                  log.foot_z_m_R])
        metrics.update(LocomotionMetrics.from_run(
            log.ankle_model_L, log.ankle_model_R, contact_L, contact_R,
            10.0, foot_pos_L=foot_L, foot_pos_R=foot_R))
    except Exception as exc:
        metrics["phase_error"] = f"{type(exc).__name__}: {exc}"
    metrics["sole_up"] = "not measured (not logged)"
    return metrics


def _run_cpg_arm(arm: str, seed: int, steps: int, player_map=None) -> dict:
    started = time.perf_counter()
    pers, agent, log = _cpg_build(arm, seed, steps, player_map)
    metrics = _cpg_metrics(agent, log)
    agent.close()
    return {"arm": arm, "seed": int(seed), "steps": int(len(log.loss)),
            "players": (len(player_map.groups) if player_map is not None
                        else None),
            "wall_s": float(time.perf_counter() - started), **metrics}


def run_cpg(seeds=(0,), arms=("persona_manual", "persona_nogame",
                              "persona_diag", "persona_block"),
            steps: int = 400, out_dir=None, player_map=None,
            cmaes_trials: int = 4, cmaes_sub_k: int = 16) -> dict:
    from persona.partition import PlayerPartition

    if isinstance(player_map, (str, Path)):
        with open(player_map, "r", encoding="utf-8") as fh:
            player_map = PlayerPartition.from_json(json.load(fh))
    rows = []
    for seed in seeds:
        for arm in arms:
            if arm == "cmaes_sub":
                row = _run_cmaes_cpg(seed, steps, cmaes_trials,
                                     cmaes_sub_k)
            else:
                if arm == "persona_evo" and player_map is None:
                    raise ValueError("persona_evo requires --player-map")
                row = _run_cpg_arm(
                    arm, seed, steps,
                    player_map if arm == "persona_evo" else None)
            rows.append(row)
            print(f"[cpg] seed={seed} arm={arm} loss={row.get('loss_eval')} "
                  f"phase={row.get('phase_passed')} "
                  f"bio={row.get('biological_acceptable')}", flush=True)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            with (out_dir / "cpg_runs.csv").open(
                    "w", newline="", encoding="utf-8") as fh:
                fields = list(rows[0])
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows([{key: row.get(key) for key in fields}
                                  for row in rows])
        (out_dir / "cpg_summary.json").write_text(
            json.dumps(rows, indent=2, default=str), encoding="utf-8")
        lines = [
            "# PERSONA CPG comparison",
            "",
            "| arm | loss | p90 | at_bound | q>1 | viol | entropy | "
            "phase | bio | duty in dist |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['arm']} | {_fmt_metric(row.get('loss_eval'))} | "
                f"{_fmt_metric(row.get('r_p90'))} | "
                f"{_fmt_metric(row.get('at_bound'), 0)} | "
                f"{_fmt_metric(row.get('q_gt_1'), 0)} | "
                f"{_fmt_metric(row.get('violations'), 0)} | "
                f"{_fmt_metric(row.get('stack_entropy'))} | "
                f"{row.get('phase_passed')} | "
                f"{row.get('biological_acceptable')} | "
                f"{row.get('duty_factor_in_distribution')} |")
        lines += ["", "Sole-up: not measured (not logged; `RunLog` does not",
                  "record `foot_sole_z_*`)."]
        (out_dir / "cpg_bio.md").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    return {"rows": rows}


def _run_cmaes_cpg(seed: int, steps: int, trials: int, k: int) -> dict:
    from bench_homeostasis import FAST_SIZES
    from persona.game import GameConfig
    from personalize import build_personalizer

    pilot, agent, _ = _cpg_build("persona_nogame", seed, steps)
    influence = np.asarray(agent.strategy.influence(), dtype=float)
    n = len(agent.params)
    combined = influence[:n] + influence[n:2 * n]
    names = list(agent.params)
    selected = [names[i] for i in np.argsort(-combined)[:k]]
    bounds = {name: (float(agent.params[name].min),
                     float(agent.params[name].max)) for name in selected}
    x0 = {name: float(agent.params[name].mean) for name in selected}
    agent.close()

    def objective(params):
        pers, cma_agent, _ = _cpg_build("persona_nogame", seed, steps,
                                        run=False)
        for name, value in params.items():
            cma_agent.params[name].mean = float(value)
        log = pers.run(pose_max_steps=100, realize_every_k=1)
        cma_agent.close()
        return float(np.mean(log.loss[-50:]))

    result = run_cmaes(objective, bounds, x0, n_trials=trials, seed=seed)
    return {"arm": "cmaes_sub", "seed": int(seed), "steps": int(steps),
            "players": None, "wall_s": float(result["wall_s"]),
            "loss_eval": float(result["best_value"]), "k": int(k),
            "trials": int(trials)}


def _fmt_metric(value, digits=3):
    return "—" if value is None else f"{value:.{digits}g}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--cpg", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--cpg-steps", type=int, default=400)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--player-map", default=None)
    parser.add_argument("--cmaes-trials", type=int, default=4)
    parser.add_argument("--cmaes-sub-k", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/compare"))
    args = parser.parse_args(argv)
    if args.cpg:
        return run_cpg(
            seeds=tuple(args.seeds),
            arms=tuple(args.arms or CPG_ARMS),
            steps=args.cpg_steps, out_dir=args.out,
            player_map=args.player_map, cmaes_trials=args.cmaes_trials,
            cmaes_sub_k=args.cmaes_sub_k)
    if args.smoke:
        return run_toy(seeds=(0,), arms=("persona_game", "persona_diag",
                                        "cmaes", "surrogate"),
                       episodes=2, out_dir=args.out / "smoke")
    return run_toy(seeds=tuple(args.seeds),
                   arms=tuple(args.arms or ARMS),
                   episodes=args.episodes, out_dir=args.out / "toy")


if __name__ == "__main__":
    main()
