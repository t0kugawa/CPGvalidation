"""Optuna CMA-ES arm for the toy comparison."""
from __future__ import annotations

import time


def run_cmaes(objective, bounds: dict, x0: dict, n_trials: int,
              seed: int = 0) -> dict:
    """Minimize ``objective(params)`` with Optuna's CMA-ES sampler."""
    import optuna
    from optuna.samplers import CmaEsSampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = CmaEsSampler(
        seed=int(seed),
        x0={name: float(x0[name]) for name in bounds},
        sigma0=0.1,
        n_startup_trials=0,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    started = time.perf_counter()

    def _objective(trial):
        params = {
            name: trial.suggest_float(name, float(low), float(high))
            for name, (low, high) in bounds.items()
        }
        value = float(objective(params))
        trial.set_user_attr("params", params)
        return value

    study.optimize(_objective, n_trials=int(n_trials))
    values = [float(trial.value) for trial in study.trials
              if trial.value is not None]
    best = study.best_trial
    return {
        "best_params": dict(best.user_attrs["params"]),
        "best_value": float(best.value),
        "values": values,
        "n_trials": int(n_trials),
        "wall_s": float(time.perf_counter() - started),
    }
