"""Common abstract registry interface for PERSONA.

This module deliberately has **no** simulator imports and no shared mutable
state.  ``Adapter`` is a pure abstract interface; the two concrete adapters
(``persona.nest_adapter.NestAdapter`` and
``persona.neuron_adapter.NeuronAdapter``) own their state and implement every
method below.  The few pure helpers in this module are module-level functions,
not a mixin, so there is no ``SiteBook``-style abstraction layer.

The contract is documented in
``docs/superpowers/specs/2026-10-23-neuron-registry-adapter-design.md``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from fnmatch import fnmatchcase
from typing import Any, Dict, Iterable, List

import numpy as np

from persona.connection import Connection
from persona.noise_recorder import NoiseRecorder
from persona.parameter import PersonaParam
from persona.population import Population
from persona.spring import SpringSpec


# ---------------------------------------------------------------------------
# Pure helpers shared by both concrete adapters.
# ---------------------------------------------------------------------------

def clamp_weight_sign(param: PersonaParam) -> PersonaParam:
    """Enforce the sign identity of a static synaptic weight.

    An excitatory weight can never cross below zero and an inhibitory weight
    can never cross above zero, so the two cannot replace each other through
    learning. Existing stricter bounds are preserved; a zero-mean weight is
    ambiguous and left untouched. This is a type/sign constraint, not a
    stability setpoint.
    """
    mean = float(param.mean)
    if mean > 0.0:
        param.min = max(float(param.min), 0.0)
    elif mean < 0.0:
        param.max = min(float(param.max), 0.0)
    return param


def enforce_synaptic_sign(param: PersonaParam, kind: str) -> PersonaParam:
    """Normalize a weight site to the identity of its synapse type.

    ``kind`` is ``"inh"``, ``"exc"`` or ``"signed"``.  An inhibitory type is
    always represented by a negative site value and an excitatory type by a
    positive one, even when a model supplied a positive magnitude with an
    explicitly inhibitory type.  This is the hard invariant: the hardware
    point process may only ever receive ``abs(value)``, so a positive
    magnitude is not allowed to turn an inhibitory projection excitatory.

    The existing stricter bound on the correct side is preserved.  A zero
    value is allowed (it has no effect) but its type identity is not changed.
    """
    mode = str(kind).lower()
    if mode not in {"inh", "exc", "signed"}:
        raise ValueError(f"unknown synapse kind {kind!r}")

    mean = float(param.mean)
    lo = float(param.min)
    hi = float(param.max)

    if mode == "inh":
        if mean > 0.0:
            mean = -mean
        # Keep the site strictly on the negative half-line.  A positive
        # magnitude supplied with ``min=0, max=W`` becomes ``[-W, 0]`` rather
        # than an empty interval around a negative mean.
        param.mean = mean
        param.max = min(hi, 0.0)
        param.min = min(lo, -abs(hi), mean, param.max)
    elif mode == "exc":
        if mean < 0.0:
            mean = -mean
        param.mean = mean
        param.min = max(lo, 0.0)
        param.max = max(hi, abs(lo), mean, param.min)
    else:
        # Signed static names: keep the seed's side, exactly as NEST does.
        return clamp_weight_sign(param)
    return param


def plastic_delta_from_events(events: dict, last_w: dict, seed_weight: float,
                              n_syn: int) -> tuple[float, dict]:
    """Per-synapse mean weight increment of one drained recorder batch.

    ``Δw = w_event − last_w[(sender, target)]`` (first sighting → seed weight)
    and ``δ = Σ Δw / N_syn``.  Mutates and returns ``last_w``.
    """
    weights = np.asarray(events.get("weights", []), dtype=np.float64)
    senders = np.asarray(events.get("senders", []), dtype=np.int64)
    targets = np.asarray(events.get("targets", []), dtype=np.int64)
    delta_sum = 0.0
    for s, tgt, w in zip(senders, targets, weights):
        key = (int(s), int(tgt))
        prev = float(last_w.get(key, seed_weight))
        delta_sum += float(w) - prev
        last_w[key] = float(w)
    return delta_sum / max(1, int(n_syn)), last_w


def key_patterns(keys) -> List[str]:
    if isinstance(keys, str):
        return [keys]
    return [str(k) for k in keys]


def attrs_tuple(attrs) -> tuple[str, ...]:
    values = (attrs,) if isinstance(attrs, str) else tuple(attrs)
    values = tuple(str(a) for a in values)
    if not values:
        raise ValueError("at least one attribute must be given")
    bad = [a for a in values if a not in ("mean", "sigma")]
    if bad:
        raise ValueError(
            f"freezable attributes are 'mean' and 'sigma', got {bad}")
    return values


def resolve_site_keys(params: Dict[str, PersonaParam], keys) -> tuple[str, ...]:
    """Resolve exact canonical keys or fnmatch globs to sorted site keys.

    Exact keys are checked first.  This is required for bracketed NEURON
    section paths such as ``dend[0].g_pas``, because ``fnmatchcase`` treats
    ``[0]`` as a character class.  A glob still works as before.
    """
    resolved: List[str] = []
    for pattern in key_patterns(keys):
        if pattern in params:
            matches = [pattern]
        else:
            matches = [k for k in params if fnmatchcase(k, pattern)]
        if not matches:
            raise KeyError(f"no PERSONA site matches {pattern!r}")
        for key in matches:
            if key not in resolved:
                resolved.append(key)
    return tuple(sorted(resolved))


# ---------------------------------------------------------------------------
# The common interface.
# ---------------------------------------------------------------------------

class Adapter(ABC):
    """Pure abstract interface implemented by every PERSONA registry adapter."""

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    # --- kernel lifecycle (simulator-owned) --------------------------------

    @classmethod
    @abstractmethod
    def reset_kernel(
        cls,
        *,
        resolution_ms: float | None = None,
        v_init: float | None = None,
        cvode: bool = False,
        maxstep_ms: float | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_resolution(self, resolution_ms: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, v_init: float | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def simulate(self, dt_ms: float) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def time_ms(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def dt_ms(self) -> float:
        raise NotImplementedError

    # --- noise recorders (simulator-independent) ----------------------------

    @abstractmethod
    def register_noise_recorder(self, recorder: NoiseRecorder) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_noise_recorder(self, name: str) -> NoiseRecorder | None:
        raise NotImplementedError

    @abstractmethod
    def site_recorder(self, full_key: str) -> NoiseRecorder:
        raise NotImplementedError

    @abstractmethod
    def producer_recorder(
        self, name: str, scale: float | None = None) -> NoiseRecorder:
        raise NotImplementedError

    @abstractmethod
    def producer_recorders(self) -> Dict[str, NoiseRecorder]:
        raise NotImplementedError

    # --- creation defaults, freeze, springs --------------------------------

    @abstractmethod
    def recapture_defaults(self, keys=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def freeze(self, keys, attrs=("mean",)) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self, keys, attrs=None) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def frozen_attrs(self, site_key: str) -> frozenset[str]:
        raise NotImplementedError

    @abstractmethod
    def is_frozen(self, site_key: str, attr: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def defaults(self, site_key: str) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def create_spring(self, name: str, keys, *, power: float | None = None,
                      gain: float = 1.0,
                      max_power: float = 0.5) -> SpringSpec:
        raise NotImplementedError

    @abstractmethod
    def remove_spring(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def springs(self) -> Dict[str, SpringSpec]:
        raise NotImplementedError

    @abstractmethod
    def spring_state(self, means: Dict[str, float],
                     sigmas: Dict[str, float],
                     lr_t: float) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    # --- nodes, populations, connections, relays ---------------------------

    @abstractmethod
    def register_nodes(self, name: str, nodes: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_parrot_nodes(self, name: str, n: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_spike_recorder(self, target: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_weight_recorder(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_population(
        self, cfg: Population,
        explored_means: Dict[str, float] | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_connection(
        self, cfg: Connection,
        explored_params: Dict[str, float] | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_relay(
        self, pg_key: str, in_key: str, n: int, *,
        rate: float = 0.0,
        rate_param: PersonaParam | None = None,
        pop_name: str | None = None,
        noise_name: str | None = None,
        noise_scale: float | None = None,
    ) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_relay_rate(self, pg_key: str, rate: float) -> None:
        raise NotImplementedError

    # --- synapse views and plasticity --------------------------------------

    @abstractmethod
    def synapses(self, conn_key: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def synapse_count(self, conn_key: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def finalize_connections(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_plastic_events(
        self, clear: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def sample_plastic_weight_deltas(
        self, t_before: float, t_now: float) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def last_plastic_counts(self) -> Dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def sample_plastic_weight_stats(self) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def final_plastic_weights(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    # --- parameters, sampling, apply ---------------------------------------

    @abstractmethod
    def get_all_optimizable_params(self) -> Dict[str, PersonaParam]:
        raise NotImplementedError

    @abstractmethod
    def sample_all_actions(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def apply_explored_parameters(
        self, sampled_values: Dict[str, float]) -> None:
        raise NotImplementedError

    # --- introspection ------------------------------------------------------

    @abstractmethod
    def get(self, name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_population_cfg(self, name: str) -> Population:
        raise NotImplementedError
