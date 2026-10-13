from typing import Any

import nest
from cfg import *

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
    assert poisson_n == parrot_n, f"one_to_one connection requires equal poisson_n (provided {poisson_n}) and parrot_n (provided {parrot_n})"

    poisson = nest.Create("poisson_generator", poisson_n)
    parrot = nest.Create("parrot_neuron", parrot_n)
    nest.Connect(poisson, parrot, conn_spec={"rule": "one_to_one"})
    nest.SetStatus(poisson, {"rate": hz})
    return poisson, parrot


def create_izhi_population(size: int, params=INIT_IZHI_PARAMS) -> Any:
    return nest.Create("izhikevich", size, params=params)


def create_spike_recorder(target):
    rec = nest.Create("spike_recorder", params={"record_to": "memory"})
    nest.Connect(target, rec)
    return rec


def create_weight_recorder():
    return nest.Create("weight_recorder")


def copy_syn_model(name, params, wr):
    if wr is not None:
        nest.CopyModel("stdp_synapse", name, {**params, "weight_recorder": wr})
    else:
        nest.CopyModel("stdp_synapse", name, params)


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

def create_stdp_synapse(type: str, postfix: str):
    match type:
        case "default":
            name = f"stdp_default_{postfix}"
            copy_syn_model(name, STDP_DEFAULT, create_weight_recorder())
            return name
        case "ia":
            name = f"stdp_ia_{postfix}"
            copy_syn_model(f"stdp_ia_{postfix}", STDP_IA, create_weight_recorder())
            return name
        case "bs":
            name = f"stdp_bs_{postfix}"
            copy_syn_model(name, STDP_BS, create_weight_recorder())
            return name




SYN_MS: float = 1.0
LENGTH_M: float = 0.005
VELOCITY_MPS: float = 5.45

def get_jitter_delay(jitter_ms: float=0.2) -> float:
    delay = SYN_MS + (LENGTH_M / VELOCITY_MPS) * 1000.0
    jm = float(max(0.0, jitter_ms))
    if jm > 0.0:
        delay += nest.random.normal(mean=0.0, std=jm)

    delay = float(max(float(delay), float(max(1e-9, DT_MS))))
    return delay



