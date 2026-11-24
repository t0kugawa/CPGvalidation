"""Delay, STDP-synapse and small numeric helpers for the CPG."""
from typing import Any

import nest
import numpy as np

from persona.parameter import PersonaParam
from t2xcpg.params import PARAMS


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def copy_syn_model(name: str, params: dict, wr=None):
    p = dict(params)
    if wr is not None:
        p["weight_recorder"] = wr
    nest.CopyModel("stdp_synapse", name, p)


def stdp_param_spec(syn_type: str) -> dict:
    prefix = {"ia": "stdp_ia_", "bs": "stdp_bs_", "cut": "stdp_cut_"}.get(syn_type, "stdp_default_")
    return {
        "tau_plus": float(PARAMS[f"{prefix}tau_plus"]),
        "lambda": float(PARAMS[f"{prefix}lambda"]),
        "alpha": float(PARAMS[f"{prefix}alpha"]),
        "mu_plus": float(PARAMS[f"{prefix}mu_plus"]),
        "mu_minus": float(PARAMS[f"{prefix}mu_minus"]),
        "Wmax": float(PARAMS[f"{prefix}wmax"]),
    }


def create_stdp_synapse(registry, syn_type: str, postfix: str) -> tuple[str, Any]:
    namespace = getattr(registry, "namespace", "")
    name = f"stdp_{syn_type}_{namespace}{postfix}"
    wr = registry.create_weight_recorder()
    copy_syn_model(name, stdp_param_spec(syn_type), wr)
    return name, wr


STDP_POSTFIX = {"cut_rge": "cut", "ia_rg": "ia", "ia_rge": "ia", "ia_rgf": "ia"}


def get_jitter_delay(jitter_ms: float | None = None) -> float:
    jitter = PARAMS["delay_jitter_ms"] if jitter_ms is None else float(jitter_ms)
    delay = PARAMS["syn_ms"] + (
        PARAMS["length_m"] / PARAMS["velocity_mps"]) * 1000.0
    if jitter > 0.0:
        delay += float(np.random.normal(0.0, jitter))
    return float(max(1.0, delay))


def delay_param(mean_ms: float) -> PersonaParam:
    """One optimizable synaptic delay, seeded at the computed mean."""
    lo = float(PARAMS["delay_min_ms"])
    hi = float(PARAMS["delay_max_ms"])
    mean = clamp(mean_ms, lo, hi)
    sigma = max(mean * float(PARAMS["delay_sigma_fraction"]),
                float(PARAMS["delay_sigma_floor_ms"]))
    return PersonaParam(mean=mean, sigma=sigma, min=lo, max=hi)
