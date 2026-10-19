from typing import Any
import nest
import numpy as np

try:
    from t2xcpg.cfg import DT_MS
except ImportError:
    DT_MS = 10.0


INIT_IZHI_PARAMS = {
    "a": 0.02,
    "b": 0.2,
    "c": -65.0,
    "d": 8.0,
    "V_th": 30.0,
    "V_min": -120.0,
}

DELAY_SCALE: float = 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def copy_syn_model(name: str, params: dict, wr=None):
    p = dict(params)
    if wr is not None:
        p["weight_recorder"] = wr
    nest.CopyModel("stdp_synapse", name, p)


STDP_DEFAULT = {
    "tau_plus": 20.0,
    "lambda": 0.001,
    "alpha": 0.95,
    "mu_plus": 0.4,
    "mu_minus": 0.4,
    "Wmax": 120.0,
}

STDP_IA = {
    "tau_plus": 20.0,
    "lambda": 0.001,
    "alpha": 0.95,
    "mu_plus": 0.4,
    "mu_minus": 0.4,
    "Wmax": 10.0,
}

STDP_BS = {
    "tau_plus": 20.0,
    "lambda": 0.001,
    "alpha": 0.95,
    "mu_plus": 0.4,
    "mu_minus": 0.4,
    "Wmax": 30.0,
}


def stdp_param_spec(syn_type: str) -> dict:
    """
    The optimizable STDP hyper-parameters (means) for a synapse family. PERSONA
    optimizes these, never the synaptic `weight` (which STDP itself owns).
    """
    match syn_type:
        case "ia":
            return dict(STDP_IA)
        case "bs":
            return dict(STDP_BS)
        case _:
            return dict(STDP_DEFAULT)


def create_stdp_synapse(registry, syn_type: str, postfix: str) -> tuple[str, Any]:
    """
    Defines one plastic projection's STDP synapse model. Creating the weight
    recorder goes through the registry (single NEST route); this module keeps
    only the `nest.CopyModel` call that must run before the registry connects.
    """
    name = f"stdp_{syn_type}_{postfix}"
    wr = registry.create_weight_recorder()
    copy_syn_model(name, stdp_param_spec(syn_type), wr)
    return name, wr


SYN_MS: float = 1.0
LENGTH_M: float = 0.005
VELOCITY_MPS: float = 5.45


# STDP is used on the sensory input pathways (spec D12): CUT -> RGE and Ia -> RG.
# All other projections are static synapses.
STDP_POSTFIX = {"cut_rge": "default", "ia_rg": "ia", "ia_rge": "ia", "ia_rgf": "ia"}


def get_jitter_delay(jitter_ms: float = 0.2) -> float:
    delay = SYN_MS + (LENGTH_M / VELOCITY_MPS) * 1000.0
    if jitter_ms > 0.0:
        delay += float(np.random.normal(0.0, jitter_ms))
    # Must be at least 1.0 ms in NEST
    return float(max(1.0, delay))
