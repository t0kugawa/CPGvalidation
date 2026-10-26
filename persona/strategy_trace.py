"""Exact replay of a PERSONA strategy from an applied-sample trace.

The trace stores the sampling-time policy state, not the post-step parameter
means: ``dtheta = xhat - mu_at``, ``sigma_at``, the applied ``dlogsigma``, the
advantage vector, and the producer deltas/scales.  Replaying those through the
same ``F/Xbar/G`` recurrences reproduces the strategy exactly in float64.

This module is deliberately separate from :mod:`persona.strategy` so importing
it does not pull in the game/homeostasis wiring.
"""
from __future__ import annotations

import numpy as np

from persona.strategy import FisherMatrix

__all__ = ["rebuild_strategy_from_trace", "trace_arrays"]


def trace_arrays(trace: dict) -> dict:
    """Return a trace with list-valued fields converted to arrays.

    Agent stores rows in lists while a run is live; HDF5 and replay consume
    arrays.  Existing arrays are returned unchanged.
    """
    out = dict(trace)
    for key in ("dtheta", "sigma", "dlogsigma", "advantage",
                "noise_delta", "noise_scale"):
        value = out.get(key)
        if value is None:
            continue
        out[key] = np.asarray(value, dtype=out.get("dtype", "float64"))
    return out


def rebuild_strategy_from_trace(trace: dict,
                                FisherMatrix_cls=FisherMatrix
                                ) -> FisherMatrix:
    """Replay an applied-sample trace into a fresh strategy.

    The trace must contain ``param_names``, ``noise_names``, ``beta``,
    ``lambda_reg``, ``warmup``, ``refresh_every``, ``dtheta``, ``sigma``,
    ``dlogsigma``, ``advantage``, ``noise_delta`` and ``noise_scale``.
    ``dtype`` is used for the row conversion; float32 traces are approximate.
    """
    arrays = trace_arrays(trace)
    param_names = list(arrays["param_names"])
    noise_names = list(arrays["noise_names"])
    strategy = FisherMatrix_cls(
        param_names=param_names,
        noise_names=noise_names,
        beta=float(arrays["beta"]),
        lambda_reg=float(arrays["lambda_reg"]),
        warmup=int(arrays["warmup"]),
        refresh_every=int(arrays["refresh_every"]),
        lr_mu=float(arrays.get("lr_mu", 0.02)),
        lr_sigma=float(arrays.get("lr_sigma", 0.005)),
        dt_s=float(arrays.get("dt_s", 0.01)),
    )
    dtheta = arrays["dtheta"]
    sigma = arrays["sigma"]
    dlogsigma = arrays["dlogsigma"]
    advantage = arrays["advantage"]
    noise_delta = arrays["noise_delta"]
    noise_scale = arrays["noise_scale"]
    for t in range(len(dtheta)):
        sites = {
            name: (dtheta[t, i], sigma[t, i], dlogsigma[t, i])
            for i, name in enumerate(param_names)
        }
        noise = {
            name: (noise_delta[t, j], noise_scale[t, j])
            for j, name in enumerate(noise_names)
        }
        score, applied = strategy._score_vectors(sites, noise)
        strategy._accumulate(score, applied,
                             strategy._pad_advantage(advantage[t]))
    strategy._refresh_inverse()
    return strategy
