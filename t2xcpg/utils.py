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


def poisson_to_parrot(poisson_n: int, parrot_n: int, hz: float) -> Any:
    assert poisson_n == parrot_n, f"one_to_one connection requires equal poisson_n ({poisson_n}) and parrot_n ({parrot_n})"
    poisson = nest.Create("poisson_generator", poisson_n)
    parrot = nest.Create("parrot_neuron", parrot_n)
    nest.Connect(poisson, parrot, conn_spec={"rule": "one_to_one"})
    poisson.set({"rate": float(max(0.0, hz))})
    return poisson, parrot


def create_izhi_population(size: int, params=None) -> Any:
    p = dict(INIT_IZHI_PARAMS)
    if params:
        p.update(params)
    return nest.Create("izhikevich", size, params=p)


def create_spike_recorder(target):
    rec = nest.Create("spike_recorder", params={"record_to": "memory"})
    nest.Connect(target, rec)
    return rec


def create_weight_recorder():
    return nest.Create("weight_recorder")


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


def create_stdp_synapse(syn_type: str, postfix: str):
    name = f"stdp_{syn_type}_{postfix}"
    wr = create_weight_recorder()
    match syn_type:
        case "default":
            copy_syn_model(name, STDP_DEFAULT, wr)
        case "ia":
            copy_syn_model(name, STDP_IA, wr)
        case "bs":
            copy_syn_model(name, STDP_BS, wr)
        case _:
            copy_syn_model(name, STDP_DEFAULT, wr)
    return name, wr


SYN_MS: float = 1.0
LENGTH_M: float = 0.005
VELOCITY_MPS: float = 5.45


# STDP is used on the sensory input pathways (spec D12): CUT -> RGE and Ia -> RG.
# All other projections are static synapses.
STDP_POSTFIX = {"cut_rge": "default", "ia_rg": "ia"}


def get_jitter_delay(jitter_ms: float = 0.2) -> float:
    delay = SYN_MS + (LENGTH_M / VELOCITY_MPS) * 1000.0
    if jitter_ms > 0.0:
        delay += float(np.random.normal(0.0, jitter_ms))
    # Must be at least 1.0 ms in NEST
    return float(max(1.0, delay))
