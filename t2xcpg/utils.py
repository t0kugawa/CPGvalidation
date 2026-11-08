from typing import Any
import nest
import numpy as np

from t2xcpg.params import PARAMS

try:
    from t2xcpg.cfg import DT_MS
except ImportError:
    DT_MS = PARAMS["dt_ms"]


INIT_IZHI_PARAMS = {
    "a": PARAMS["izh_init_a"],
    "b": PARAMS["izh_init_b"],
    "c": PARAMS["izh_init_c"],
    "d": PARAMS["izh_init_d"],
    "V_th": PARAMS["izh_init_v_th"],
    "V_min": PARAMS["izh_init_v_min"],
}

DELAY_SCALE: float = PARAMS["delay_scale"]


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def copy_syn_model(name: str, params: dict, wr=None):
    p = dict(params)
    if wr is not None:
        p["weight_recorder"] = wr
    nest.CopyModel("stdp_synapse", name, p)


STDP_DEFAULT = {
    "tau_plus": PARAMS["stdp_default_tau_plus"],
    "lambda": PARAMS["stdp_default_lambda"],
    "alpha": PARAMS["stdp_default_alpha"],
    "mu_plus": PARAMS["stdp_default_mu_plus"],
    "mu_minus": PARAMS["stdp_default_mu_minus"],
    "Wmax": PARAMS["stdp_default_wmax"],
}

STDP_IA = {
    "tau_plus": PARAMS["stdp_ia_tau_plus"],
    "lambda": PARAMS["stdp_ia_lambda"],
    "alpha": PARAMS["stdp_ia_alpha"],
    "mu_plus": PARAMS["stdp_ia_mu_plus"],
    "mu_minus": PARAMS["stdp_ia_mu_minus"],
    "Wmax": PARAMS["stdp_ia_wmax"],
}

STDP_BS = {
    "tau_plus": PARAMS["stdp_bs_tau_plus"],
    "lambda": PARAMS["stdp_bs_lambda"],
    "alpha": PARAMS["stdp_bs_alpha"],
    "mu_plus": PARAMS["stdp_bs_mu_plus"],
    "mu_minus": PARAMS["stdp_bs_mu_minus"],
    "Wmax": PARAMS["stdp_bs_wmax"],
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
    namespace = getattr(registry, "namespace", "")
    name = f"stdp_{syn_type}_{namespace}{postfix}"
    wr = registry.create_weight_recorder()
    copy_syn_model(name, stdp_param_spec(syn_type), wr)
    return name, wr


SYN_MS: float = PARAMS["syn_ms"]
LENGTH_M: float = PARAMS["length_m"]
VELOCITY_MPS: float = PARAMS["velocity_mps"]


# STDP is used on the sensory input pathways (spec D12): CUT -> RGE and Ia -> RG.
# All other projections are static synapses.
STDP_POSTFIX = {"cut_rge": "default", "ia_rg": "ia", "ia_rge": "ia", "ia_rgf": "ia"}


def get_jitter_delay(jitter_ms: float | None = None) -> float:
    jitter_ms = PARAMS["delay_jitter_ms"] if jitter_ms is None else jitter_ms
    delay = SYN_MS + (LENGTH_M / VELOCITY_MPS) * 1000.0
    if jitter_ms > 0.0:
        delay += float(np.random.normal(0.0, jitter_ms))
    # Must be at least 1.0 ms in NEST
    return float(max(1.0, delay))
