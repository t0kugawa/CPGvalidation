"""Production runner for one PERSONA-Evo generation.

Kept separate from :mod:`persona.evo` so the trainer and its tests never import
NEST. The fresh-process wrapper mirrors ``bench_game._run_fresh``.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from persona.evo import EvoConfig, GenerationResult
from persona.partition import PlayerPartition


def run_generation(partition: PlayerPartition, config: EvoConfig,
                   generation: int, seed: int) -> GenerationResult:
    """Run the normal game-on personalization with one partition."""
    from bench_homeostasis import (
        FAST_SIZES,
        inflation_metrics,
        type_sign_violations,
    )
    from personalize import build_personalizer
    from persona.game import GameConfig

    started = time.perf_counter()
    game = GameConfig.current_best()
    game.share_floor = 0.0
    sizes = dict(config.sizes or FAST_SIZES)
    pers, _ = build_personalizer(
        max_steps=int(config.steps_per_generation),
        eps_pose=0.10, pose_hold_steps=3, loss_stop=-1.0, loss_window=20,
        ref_hz=700.0, resolution_ms=0.2, seed=int(seed),
        game=game, homeostasis=None, player_map=partition,
        **sizes,
    )
    agent = pers.agent
    initial = {name: float(param.mean)
               for name, param in agent.params.items()}
    bounds = {name: (float(param.min), float(param.max))
              for name, param in agent.params.items()}
    ptypes = {}
    if agent.game is not None:
        for spec in agent.game.physical_map.specs:
            for site in spec.param_names:
                ptypes[site] = spec.ptype
    pers.verbose = False
    pers.log_every = 0
    pers.stdp_every = 0
    pers.belt_mps = 0.2
    pers.belt_start_step = 50
    pers.belt_ramp_steps = 50
    try:
        log = pers.run(pose_max_steps=100, realize_every_k=1)
    finally:
        agent.close()

    game_history = list(agent.history.get("game", []))
    if not game_history:
        raise RuntimeError("Evo generation produced no game diagnostics")
    window = game_history[-int(config.eval_window):]
    stacks = np.asarray([entry["stacks"] for entry in window], dtype=float)
    fractions = np.asarray(game_history[-1]["stacks"], dtype=float)
    fraction_mean = stacks.mean(axis=0)
    contribution = np.asarray(
        [entry["contribution"] for entry in window], dtype=float).mean(axis=0)
    transfer = np.asarray(
        [entry["transfer"] for entry in window], dtype=float).mean(axis=0)
    replicator_residual = float(np.max(np.abs(transfer)))
    movement = float(np.mean([
        float(np.max(np.abs(np.asarray(entry["transfer"], dtype=float))))
        for entry in window
    ]))
    eval_start = max(0, len(log.loss) - int(config.eval_window))
    metrics = {
        "loss_eval": float(np.mean(log.loss[eval_start:])),
        "loss_full": float(np.mean(log.loss)),
        "n_steps": int(len(log.loss)),
        "active_players": int(np.sum(fractions > 1e-12)),
        "stack_entropy": float(game_history[-1].get("entropy_norm", 0.0)),
        "n_players": int(len(partition.groups)),
    }
    metrics.update(inflation_metrics(agent.history["means"], initial, bounds,
                                     eval_start=eval_start))
    metrics["violations"] = int(len(type_sign_violations(
        agent.history["means"], bounds, ptypes)))
    try:
        from loader.stats import LocomotionMetrics
        from persona.phase_metrics import phase_gate_metrics
        from personalize import acceptance_from_log

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
    except Exception as exc:  # diagnostic only; never hide the main result
        metrics["phase_error"] = f"{type(exc).__name__}: {exc}"
    return GenerationResult(
        partition=partition,
        generation=int(generation),
        seed=int(seed),
        fractions=fractions,
        fraction_mean=fraction_mean,
        contribution=contribution,
        transfer=transfer,
        replicator_residual=replicator_residual,
        movement=movement,
        metrics=metrics,
        wall_s=float(time.perf_counter() - started),
    )


def _run_generation_task(partition, config, generation, seed):
    return run_generation(partition, config, generation, seed)


def run_generation_fresh(partition: PlayerPartition, config: EvoConfig,
                         generation: int, seed: int) -> GenerationResult:
    """Run one generation in a fresh process."""
    with ProcessPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_generation_task, partition, config,
                           generation, seed).result()
