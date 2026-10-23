"""NEURON kernel lifecycle facade.

Raw NEURON access is intentionally confined to this package and to
``persona/neuron_adapter.py``.  Model code calls the registry, never ``h``.
"""
from __future__ import annotations

from neuron import h


_GENERATION = 0
_DEFAULT_DT = 0.025
_DEFAULT_V = -65.0


def generation() -> int:
    return int(_GENERATION)


def reset_kernel(
    *,
    resolution_ms: float | None = None,
    v_init: float | None = None,
    cvode: bool = False,
    maxstep_ms: float | None = None,
) -> None:
    """Reset NEURON's process-global state and start a new kernel generation."""
    global _GENERATION
    _GENERATION += 1

    # Drop prior sections (and therefore the point processes attached to
    # them).  Creating a new registry after this call is the supported run
    # pattern.
    try:
        for sec in list(h.allsec()):
            h.delete_section(sec=sec)
    except Exception:
        try:
            h("forall delete_section()")
        except Exception:
            pass

    dt = float(resolution_ms if resolution_ms is not None else _DEFAULT_DT)
    if not (dt > 0.0):
        raise ValueError("resolution_ms must be > 0")
    h.dt = dt
    try:
        h.CVode().active(1 if cvode else 0)
        if cvode and maxstep_ms is not None:
            h.CVode().maxstep = float(maxstep_ms)
    except Exception:
        pass
    h.t = 0.0
    h.finitialize(float(v_init if v_init is not None else _DEFAULT_V))
    h.fcurrent()


def set_resolution(resolution_ms: float) -> None:
    h.dt = float(resolution_ms)


def initialize(v_init: float | None = None) -> None:
    h.finitialize(float(v_init if v_init is not None else _DEFAULT_V))
    h.fcurrent()


def simulate(dt_ms: float) -> float:
    target = float(h.t) + float(dt_ms)
    guard = 0
    while float(h.t) < target - 1e-9:
        h.fadvance()
        guard += 1
        if guard > 10_000_000:
            raise RuntimeError("NEURON simulation did not advance")
    return float(h.t)


def time_ms() -> float:
    return float(h.t)


def dt_ms() -> float:
    return float(h.dt)


def set_maxstep(maxstep_ms: float) -> None:
    try:
        h.CVode().maxstep = float(maxstep_ms)
    except Exception:
        pass


def set_neuron_seed(seed: int) -> object:
    rng = h.Random()
    rng.Random123(int(seed), 0, 0)
    return rng


def parallel_context():
    return h.ParallelContext()
