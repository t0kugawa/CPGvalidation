"""PERSONA-Evo harness: compare initialization and stop rules on the fast CPG.

Usage::

    python bench_evo.py --smoke
    python bench_evo.py --compare --jobs 2
    python bench_evo.py --players 24 --seeds 0 --generations 2 --steps 400
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from persona.evo import EvoConfig, EvoTrainer, GenerationResult, write_evo_outputs
from persona.partition import PlayerPartition
from persona.physo_reference import (
    global_drive_fate,
    level2_labels,
    partition_metrics,
)
from persona.segments import Segment


class _DummyMatrix:
    """Small synthetic matrix for fake-runner tests; never used for a real run."""

    def __init__(self, n: int = 16):
        self.n_segments = int(n)
        self.n_sites = int(n)
        self.segments = tuple(
            Segment(index, f"seg_{index}", "population",
                    (f"site_{index}",), 1, "exc")
            for index in range(self.n_segments))
        self.adjacency = np.zeros((n, n), dtype=float)
        for index in range(n - 1):
            self.adjacency[index, index + 1] = 1.0
            self.adjacency[index + 1, index] = 1.0
        self.distance = shortest_path(csr_matrix(self.adjacency > 0.0),
                                      directed=False, unweighted=True)
        self.edge_types = {
            (index, index + 1): frozenset({"incidence"})
            for index in range(n - 1)
        }

    def validate_partition(self, partition):
        seen = set()
        for group in partition.groups:
            if not group:
                raise ValueError("empty group")
            for segment in group:
                if segment in seen:
                    raise ValueError("duplicate")
                seen.add(segment)
        if seen != set(range(self.n_segments)):
            raise ValueError("missing segment")

    def player_link(self, segment, members):
        members = list(members)
        return (float(np.min(self.distance[segment, members])),
                float(self.adjacency[segment, members].sum()))


def _real_matrix():
    from bench_homeostasis import FAST_SIZES
    from personalize import build_personalizer
    from persona.segments import SegmentMatrix

    scratch, _ = build_personalizer(
        max_steps=1, game=None, homeostasis=None, seed=0, **FAST_SIZES)
    matrix = SegmentMatrix.from_registry(scratch.registry)
    scratch.agent.close()
    return matrix


def _reference_partition(matrix) -> PlayerPartition:
    groups: dict[str, list[int]] = {}
    for segment, label in level2_labels(matrix).items():
        groups.setdefault(label, []).append(segment)
    return PlayerPartition(tuple(tuple(sorted(group))
                                 for group in groups.values()))


def evaluate_partition(partition: PlayerPartition, matrix, *,
                       seed: int = 0, steps: int = 400) -> dict:
    """Run one fixed partition through the normal game-on generation path."""
    from persona.evo_runner import run_generation

    config = EvoConfig(
        n_players=len(partition.groups), generations=1,
        steps_per_generation=int(steps), eval_window=50, seed=int(seed),
        fresh_process=False, final_run=False)
    result = run_generation(partition, config, 0, seed)
    labels = level2_labels(matrix)
    metrics = dict(result.metrics)
    metrics.update(partition_metrics(partition, labels))
    metrics["global_drive"] = global_drive_fate(partition, matrix)[
        "classification"]
    metrics["residual"] = float(result.replicator_residual)
    return metrics


def evaluate_partitions(out_dir, partitions: dict, *, seeds=(0,),
                        steps: int = 400) -> dict:
    """Compare named partitions on the fast CPG with the common metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = _real_matrix()
    rows = []
    for name, partition in partitions.items():
        for seed in seeds:
            metrics = evaluate_partition(partition, matrix, seed=seed,
                                         steps=steps)
            rows.append({
                "arm": name, "seed": int(seed),
                "players": len(partition.groups), **{
                    key: metrics.get(key) for key in (
                        "loss_eval", "loss_full", "r_p90", "at_bound",
                        "q_gt_1", "violations", "active_players",
                        "stack_entropy", "ari", "nmi", "purity", "coverage",
                        "v_measure", "pairwise_f1", "global_drive",
                        "residual", "phase_passed", "phase_cycles",
                        "phase_period_cv", "phase_ef", "phase_lr_drift",
                        "phase_angle_corr", "phase_failures",
                        "biological_acceptable", "angle_coverage",
                        "duty_factor_in_distribution", "duty_factor_L",
                        "duty_factor_R", "double_support", "step_length_L",
                        "step_length_R", "stride_length_L", "stride_length_R",
                        "phase_error")},
            })
            print(f"[eval] {name} seed={seed} loss={metrics.get('loss_eval')} "
                  f"phase={metrics.get('phase_passed')} "
                  f"bio={metrics.get('biological_acceptable')}", flush=True)
    (out_dir / "evaluation.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")
    if rows:
        fields = list(rows[0])
        (out_dir / "evaluation.csv").write_text(
            "\n".join([
                ",".join(fields),
                *[",".join("" if row.get(key) is None else str(row.get(key))
                           for key in fields) for row in rows],
            ]) + "\n", encoding="utf-8")
    lines = [
        "# PERSONA-Evo partition evaluation",
        "",
        "| arm | players | loss | p90 | at_bound | q>1 | viol | entropy |"
        " ARI | NMI | phase | bio | duty in dist |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['players']} | "
            f"{_fmt(row.get('loss_eval'))} | {_fmt(row.get('r_p90'))} | "
            f"{_fmt(row.get('at_bound'), 0)} | {_fmt(row.get('q_gt_1'), 0)} | "
            f"{_fmt(row.get('violations'), 0)} | "
            f"{_fmt(row.get('stack_entropy'))} | {_fmt(row.get('ari'))} | "
            f"{_fmt(row.get('nmi'))} | {row.get('phase_passed')} | "
            f"{row.get('biological_acceptable')} | "
            f"{row.get('duty_factor_in_distribution')} |")
    (out_dir / "evaluation.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    return {"rows": rows}


def run_compare(out_dir, *, seeds=(0,), players=(24,),
                inits=("territory",), stops=("both",), steps: int = 400,
                generations: int = 6, jobs: int = 1, prefix: str = "smoke",
                runner=None, matrix=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if matrix is None:
        matrix = (_DummyMatrix(max(16, max(int(p) for p in players)))
                  if runner is not None else _real_matrix())
    rows = []
    for seed in seeds:
        for n_players in players:
            for init in inits:
                for stop in stops:
                    config = EvoConfig(
                        n_players=int(n_players),
                        generations=int(generations),
                        steps_per_generation=int(steps),
                        init=str(init), stop=str(stop), seed=int(seed),
                        fresh_process=bool(runner is None),
                        final_run=False)
                    if runner is not None:
                        arm_runner = runner
                    else:
                        from persona.evo_runner import (
                            run_generation,
                            run_generation_fresh,
                        )
                        if config.fresh_process:
                            arm_runner = lambda p, g, s, c=config: \
                                run_generation_fresh(p, c, g, s)
                        else:
                            arm_runner = lambda p, g, s, c=config: \
                                run_generation(p, c, g, s)
                    result = EvoTrainer(matrix, config, arm_runner).run()
                    arm = f"{prefix}_{seed}_{n_players}_{init}_{stop}"
                    arm_dir = out_dir / arm
                    write_evo_outputs(arm_dir, result, matrix, config)
                    last = result.history[-1] if result.history else {}
                    metrics = dict(last.get("metrics", {})
                                   if isinstance(last, dict) else {})
                    labels = level2_labels(matrix)
                    metrics.update(partition_metrics(
                        result.final_partition, labels))
                    metrics["global_drive"] = global_drive_fate(
                        result.final_partition, matrix)["classification"]
                    metrics["residual"] = float(
                        last.get("replicator_residual", float("nan")))
                    metrics["movement"] = float(
                        last.get("movement", float("nan")))
                    metrics["fraction_stability"] = last.get(
                        "fraction_stability")
                    metrics["wall_total"] = float(sum(
                        float(record.get("wall_s", 0.0))
                        for record in result.history))
                    row = {
                        "prefix": prefix, "seed": int(seed),
                        "players": int(n_players), "init": str(init),
                        "stop": str(stop),
                        "stop_reason": result.stop_reason,
                        "generations": len(result.history),
                        "final_players": len(result.final_partition.groups),
                        **{key: metrics.get(key) for key in (
                            "loss_eval", "loss_full", "r_p90", "at_bound",
                            "q_gt_1", "violations", "active_players",
                            "stack_entropy", "ari", "nmi", "purity",
                            "coverage", "v_measure", "pairwise_f1",
                            "global_drive", "residual", "movement",
                            "fraction_stability", "wall_total",
                            "phase_passed", "phase_cycles", "phase_period_cv",
                            "phase_ef", "phase_lr_drift",
                            "phase_angle_corr", "phase_failures",
                            "biological_acceptable", "angle_coverage",
                            "duty_factor_in_distribution", "duty_factor_L",
                            "duty_factor_R", "double_support",
                            "step_length_L", "step_length_R",
                            "stride_length_L", "stride_length_R",
                            "phase_error")},
                    }
                    rows.append(row)
                    print(f"[evo] seed={seed} P={n_players} {init}/{stop} "
                          f"stop={result.stop_reason} "
                          f"players={len(result.final_partition.groups)} "
                          f"ari={metrics.get('ari')} "
                          f"loss={metrics.get('loss_eval')}", flush=True)
    summary = _summarize(rows)
    _write_outputs(out_dir, prefix, rows, summary)
    return {"rows": rows, "summary": summary}


def _summarize(rows):
    keys = sorted({(row["players"], row["init"], row["stop"])
                   for row in rows})
    summary = []
    for players, init, stop in keys:
        group = [row for row in rows
                 if (row["players"], row["init"], row["stop"])
                 == (players, init, stop)]
        entry = {"players": players, "init": init, "stop": stop,
                 "n": len(group)}
        for key in ("loss_eval", "ari", "nmi", "purity", "violations",
                    "final_players", "residual", "wall_total",
                    "phase_passed", "biological_acceptable",
                    "angle_coverage", "q_gt_1", "at_bound"):
            values = [float(row[key]) for row in group
                      if row.get(key) is not None
                      and np.isfinite(float(row[key]))]
            entry[f"mean_{key}"] = float(np.mean(values)) if values else None
        entry["stop_reasons"] = sorted({row["stop_reason"] for row in group})
        summary.append(entry)
    return summary


def _write_outputs(out_dir: Path, prefix: str, rows, summary):
    (out_dir / f"{prefix}_runs.csv").write_text(
        "\n".join([
            ",".join(rows[0].keys()),
            *[",".join("" if row.get(key) is None else str(row.get(key))
                       for key in rows[0]) for row in rows],
        ]) + "\n", encoding="utf-8")
    (out_dir / f"{prefix}_summary.json").write_text(
        json.dumps({"runs": rows, "summary": summary}, indent=2,
                   default=str), encoding="utf-8")
    lines = [
        f"# PERSONA-Evo {prefix}",
        "",
        "| P | init | stop | n | mean ARI | mean NMI | mean loss | "
        "phase pass | bio accept | violations | final players | stop reasons |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in summary:
        lines.append(
            f"| {entry['players']} | {entry['init']} | {entry['stop']} | "
            f"{entry['n']} | {_fmt(entry['mean_ari'])} | "
            f"{_fmt(entry['mean_nmi'])} | {_fmt(entry['mean_loss_eval'])} | "
            f"{_fmt(entry['mean_phase_passed'])} | "
            f"{_fmt(entry['mean_biological_acceptable'])} | "
            f"{_fmt(entry['mean_violations'], 0)} | "
            f"{_fmt(entry['mean_final_players'], 1)} | "
            f"{', '.join(entry['stop_reasons'])} |")
    (out_dir / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}g}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-partition", action="append", default=[],
                        help="repeatable final_partition.json to evaluate")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--players", type=int, nargs="+", default=[24, 137])
    parser.add_argument("--inits", nargs="+", default=["territory", "random"])
    parser.add_argument("--stops", nargs="+", default=["partition", "both"])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/evo"))
    parser.add_argument("--prefix", default="compare")
    args = parser.parse_args(argv)
    if args.evaluate:
        matrix = _real_matrix()
        partitions = {
            "manual": PlayerPartition(tuple(
                (index,) for index in range(matrix.n_segments))),
            "reference": _reference_partition(matrix),
        }
        for path in args.eval_partition:
            path = Path(path)
            with path.open("r", encoding="utf-8") as fh:
                partitions[path.parent.name] = PlayerPartition.from_json(
                    json.load(fh))
        return evaluate_partitions(
            args.out / "evaluation", partitions,
            seeds=tuple(args.eval_seeds), steps=args.eval_steps)
    if args.smoke:
        return run_compare(
            args.out / "smoke", seeds=(0,), players=(24,),
            inits=("territory", "random"), stops=("partition",),
            steps=400, generations=1, jobs=args.jobs, prefix="smoke")
    return run_compare(
        args.out, seeds=tuple(args.seeds), players=tuple(args.players),
        inits=tuple(args.inits), stops=tuple(args.stops), steps=args.steps,
        generations=args.generations, jobs=args.jobs, prefix=args.prefix)


if __name__ == "__main__":
    main()
