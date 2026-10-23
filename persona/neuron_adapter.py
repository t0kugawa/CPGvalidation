"""Concrete NEURON adapter for PERSONA.

The adapter owns the NEURON lifecycle and the PERSONA site bookkeeping.  All
raw ``h`` access is confined to this module and ``persona/neuron/``; model code
only sees duck-compatible groups and recorders.

Public compatibility: ``persona.neuron.NeuronRegistry`` is an alias of
``NeuronAdapter``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from neuron import h

from persona.adapter import (
    Adapter,
    attrs_tuple,
    clamp_weight_sign,
    enforce_synaptic_sign,
    key_patterns,
    plastic_delta_from_events,
    resolve_site_keys,
)
from persona.connection import Connection
from persona.noise_recorder import NoiseRecorder
from persona.parameter import PersonaParam
from persona.population import Population
from persona.spring import SpringSpec
from persona.neuron import kernel
from persona.neuron.cells import (
    AfferentFactory,
    BurstingInterneuronFactory,
    InterneuronFactory,
    MotoneuronFactory,
    MuscleFactory,
    SectionCell,
    SpikeSource,
)
from persona.neuron.groups import CellGroup, RateGroup, RelayGroup, SynapseGroup
from persona.neuron.recorders import SpikeRecorder, WeightRecorder
from persona.neuron.synapses import (
    STDPSynapseFactory,
    SynapseContext,
    TapSynapseFactory,
    builtin_synapse_factories,
)

__all__ = ["NeuronAdapter", "NeuronRegistry"]


class NeuronAdapter(Adapter):
    """PERSONA registry implemented directly on NEURON."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._generation = kernel.generation()
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._neuron_rng = kernel.set_neuron_seed(self._seed)

        self.population_cfgs: Dict[str, Population] = {}
        self.population_nodes: Dict[str, Any] = {}
        self.connection_cfgs: Dict[str, Connection] = {}
        self.connection_groups: Dict[str, SynapseGroup] = {}
        self.noise_recorders: Dict[str, NoiseRecorder] = {}

        self._defaults: Dict[str, Dict[str, float]] = {}
        self._frozen: Dict[str, set[str]] = {}
        self._springs: Dict[str, SpringSpec] = {}
        self.last_apply_errors: List[tuple[str, str]] = []

        self._syn_counts: Dict[str, int] = {}
        self._plastic_n_syn: Dict[str, int] = {}
        self._plastic_last_w: Dict[str, Dict[Tuple[int, int], float]] = {}
        self._plastic_counts: Dict[str, int] = {}
        self._plastic_cache_key: float | None = None
        self._plastic_cache: Dict[str, float] = {}

        self._relay_noise: Dict[str, str] = {}
        self._relay_last: Dict[str, float] = {}

        self._next_id = 0
        self._object_ids: Dict[int, int] = {}
        self._initialized = False

        self._cell_types: Dict[str, Any] = {
            "interneuron": InterneuronFactory(),
            "bursting_interneuron": BurstingInterneuronFactory(),
            "motoneuron": MotoneuronFactory(),
            "muscle": MuscleFactory(),
            "afferent": AfferentFactory(),
        }
        self._synapse_types: Dict[str, Any] = builtin_synapse_factories()
        self._recorders: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # kernel lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def reset_kernel(
        cls,
        *,
        resolution_ms: float | None = None,
        v_init: float | None = None,
        cvode: bool = False,
        maxstep_ms: float | None = None,
    ) -> None:
        kernel.reset_kernel(
            resolution_ms=resolution_ms,
            v_init=v_init,
            cvode=cvode,
            maxstep_ms=maxstep_ms,
        )

    def _check_alive(self) -> None:
        if kernel.generation() != self._generation:
            raise RuntimeError(
                "this NeuronAdapter belongs to an older NEURON kernel; "
                "create a new adapter after reset_kernel()")

    def set_resolution(self, resolution_ms: float) -> None:
        self._check_alive()
        if self._initialized:
            raise RuntimeError(
                "cannot change NEURON resolution after initialize()")
        value = float(resolution_ms)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("resolution_ms must be finite and > 0")
        kernel.set_resolution(value)

    def initialize(self, v_init: float | None = None) -> None:
        self._check_alive()
        kernel.initialize(v_init)
        self._initialized = True

    def simulate(self, dt_ms: float) -> float:
        self._check_alive()
        value = float(dt_ms)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("dt_ms must be finite and >= 0")
        if not self._initialized:
            self.initialize()
        return kernel.simulate(value)

    @property
    def time_ms(self) -> float:
        self._check_alive()
        return kernel.time_ms()

    @property
    def dt_ms(self) -> float:
        self._check_alive()
        return kernel.dt_ms()

    def _now_ms(self) -> float:
        return self.time_ms

    # ------------------------------------------------------------------
    # noise recorders
    # ------------------------------------------------------------------
    def register_noise_recorder(self, recorder: NoiseRecorder) -> None:
        self.noise_recorders[recorder.name] = recorder

    def get_noise_recorder(self, name: str) -> NoiseRecorder | None:
        return self.noise_recorders.get(name)

    def site_recorder(self, full_key: str) -> NoiseRecorder:
        name = f"site:{full_key}"
        rec = self.noise_recorders.get(name)
        if rec is None:
            rec = NoiseRecorder(name)
            self.register_noise_recorder(rec)
        return rec

    def producer_recorder(self, name: str,
                          scale: float | None = None) -> NoiseRecorder:
        full = name if name.startswith("producer:") else f"producer:{name}"
        rec = self.noise_recorders.get(full)
        if rec is None:
            rec = NoiseRecorder(full, scale=scale)
            self.register_noise_recorder(rec)
        elif scale is not None:
            rec.scale = float(scale) if float(scale) > 0.0 else None
        return rec

    def producer_recorders(self) -> Dict[str, NoiseRecorder]:
        return {n: r for n, r in self.noise_recorders.items()
                if n.startswith("producer:")}

    # ------------------------------------------------------------------
    # creation defaults and freeze
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_param_name(conn: Connection, name: str) -> str:
        """Canonical NEURON STDP site key (bare mechanism attribute)."""
        if not (conn.weight_recorder is not None
                or "stdp" in str(conn.model).lower()):
            return str(name)
        short = str(name)[5:] if str(name).startswith("stdp.") else str(name)
        return {
            "Wmax": "wmax",
            "tau_plus": "tauhebb",
            "tau_minus": "tauanti",
            "lambda": "hebbwt",
            "alpha": "antiwt",
            "mu_plus": "hebbwt",
            "mu_minus": "antiwt",
        }.get(short, short)

    def _record_defaults(self, site_key: str, param: PersonaParam) -> None:
        self._defaults[site_key] = {
            "mean": float(param.mean),
            "sigma": float(param.sigma),
            "min": float(param.min),
            "max": float(param.max),
            "sigma_min": float(param.sigma_min),
            "sigma_max": float(param.sigma_max),
        }

    @staticmethod
    def _key_patterns(keys) -> List[str]:
        return key_patterns(keys)

    @staticmethod
    def _attrs_tuple(attrs) -> tuple[str, ...]:
        return attrs_tuple(attrs)

    def _resolve_site_keys(self, keys) -> tuple[str, ...]:
        return resolve_site_keys(self.get_all_optimizable_params(), keys)

    def recapture_defaults(self, keys=None) -> None:
        params = self.get_all_optimizable_params()
        resolved = tuple(params) if keys is None else self._resolve_site_keys(keys)
        for key in resolved:
            self._record_defaults(key, params[key])

    def freeze(self, keys, attrs=("mean",)) -> tuple[str, ...]:
        resolved = self._resolve_site_keys(keys)
        attrs = self._attrs_tuple(attrs)
        params = self.get_all_optimizable_params()
        for key in resolved:
            if key not in self._defaults:
                raise KeyError(
                    f"no creation default recorded for PERSONA site {key!r}")
            for attr in attrs:
                setattr(params[key], attr, self._defaults[key][attr])
            self._frozen.setdefault(key, set()).update(attrs)
        return resolved

    def unfreeze(self, keys, attrs=None) -> tuple[str, ...]:
        resolved = self._resolve_site_keys(keys)
        attrs = None if attrs is None else self._attrs_tuple(attrs)
        for key in resolved:
            if attrs is None:
                self._frozen.pop(key, None)
                continue
            current = self._frozen.get(key)
            if current is not None:
                current.difference_update(attrs)
                if not current:
                    self._frozen.pop(key, None)
        return resolved

    def frozen_attrs(self, site_key: str) -> frozenset[str]:
        return frozenset(self._frozen.get(site_key, ()))

    def is_frozen(self, site_key: str, attr: str) -> bool:
        return attr in self._frozen.get(site_key, ())

    def defaults(self, site_key: str) -> Dict[str, float]:
        return dict(self._defaults[site_key])

    # ------------------------------------------------------------------
    # homeostatic springs
    # ------------------------------------------------------------------
    def create_spring(self, name: str, keys, *, power: float | None = None,
                      gain: float = 1.0,
                      max_power: float = 0.5) -> SpringSpec:
        if name in self._springs:
            raise ValueError(f"spring {name!r} already exists")
        max_power = float(max_power)
        gain = float(gain)
        if not (0.0 < max_power <= 1.0):
            raise ValueError("max_power must be in (0, 1]")
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("gain must be finite and non-negative")
        if power is not None:
            power = float(power)
            if not math.isfinite(power) or not 0.0 <= power <= max_power:
                raise ValueError(
                    f"power must be finite and in [0, {max_power}]")
        resolved = self._resolve_site_keys(keys)
        missing = [key for key in resolved if key not in self._defaults]
        if missing:
            raise KeyError(
                f"no creation default recorded for PERSONA sites {missing}")
        for spec in self._springs.values():
            overlap = set(resolved) & set(spec.keys)
            if overlap:
                raise ValueError(
                    f"spring {name!r} overlaps existing spring {spec.name!r} "
                    f"on {sorted(overlap)}")
        spec = SpringSpec(name=name, keys=resolved, power=power,
                          gain=gain, max_power=max_power)
        self._springs[name] = spec
        return spec

    def remove_spring(self, name: str) -> None:
        del self._springs[name]

    def springs(self) -> Dict[str, SpringSpec]:
        return dict(self._springs)

    def spring_state(self, means: Dict[str, float],
                     sigmas: Dict[str, float],
                     lr_t: float) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for spec in self._springs.values():
            for key in spec.keys:
                if key not in means or key not in sigmas:
                    continue
                if self.is_frozen(key, "mean"):
                    continue
                anchor = self._defaults[key]["mean"]
                sigma_ref = self._defaults[key]["sigma"]
                sigma_min = self._defaults[key]["sigma_min"]
                power = spec.power_for(sigmas[key], sigma_ref, lr_t, sigma_min)
                out[key] = {
                    "force": power * (anchor - float(means[key])),
                    "power": power,
                    "anchor": anchor,
                    "mean": float(means[key]),
                    "lr_mu": float(lr_t),
                }
        return out

    # ------------------------------------------------------------------
    # ids and group helpers
    # ------------------------------------------------------------------
    @property
    def seed(self) -> int:
        return self._seed

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def h(self):
        """The NEURON ``h`` object (adapter-only escape hatch)."""
        return h

    def set_maxstep(self, maxstep_ms: float) -> None:
        value = float(maxstep_ms)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("maxstep_ms must be finite and > 0")
        kernel.set_maxstep(value)

    def set_neuron_seed(self, seed: int) -> None:
        self._neuron_rng = kernel.set_neuron_seed(int(seed))

    @property
    def parallel_context(self):
        return kernel.parallel_context()

    def cell_type(self, name: str) -> Any:
        return self._cell_types[name]

    def synapse_type(self, name: str) -> Any:
        return self._synapse_types[name]

    def _next_cell_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _stable_id(self, cell: Any) -> int:
        if isinstance(cell, SectionCell):
            return int(cell.ident)
        key = id(cell)
        if key not in self._object_ids:
            self._object_ids[key] = self._next_cell_id()
        return int(self._object_ids[key])

    def _cells(self, group: Any) -> List[Any]:
        if isinstance(group, (CellGroup, RelayGroup, RateGroup)):
            return list(group)
        if isinstance(group, (list, tuple)):
            return list(group)
        raise TypeError(f"{type(group).__name__} is not an iterable group")

    def _spike_source(self, group: Any, cell: Any) -> SpikeSource:
        if isinstance(group, CellGroup):
            return group.cell_type.spike_source(cell)
        if isinstance(group, RelayGroup):
            return group.spike_source(cell)
        raise TypeError(
            f"{type(group).__name__} cannot be used as a spike source")

    def _set_nodes(self, nodes: Any, update_dict: Dict[str, Any]) -> None:
        if hasattr(nodes, "set"):
            nodes.set(update_dict)
            return
        for i, node in enumerate(nodes):
            per = {}
            for key, val in update_dict.items():
                arr = np.asarray(val, dtype=np.float64).ravel()
                per[key] = float(arr[0] if arr.size == 1 else arr[i])
            node.set(per)

    # ------------------------------------------------------------------
    # node registration, recorders, relays
    # ------------------------------------------------------------------
    def register_nodes(self, name: str, nodes: Any) -> Any:
        self._check_alive()
        self.population_nodes[name] = nodes
        return nodes

    def create_parrot_nodes(self, name: str, n: int) -> RelayGroup:
        self._check_alive()
        count = int(n)
        if count < 0:
            raise ValueError("n must be >= 0")
        cells = [h.IntFire4() for _ in range(count)]
        ids = [self._next_cell_id() for _ in cells]
        for cell, ident in zip(cells, ids):
            self._object_ids[id(cell)] = ident
        group = RelayGroup(name, cells, kind="tap", ids=ids)
        self.register_nodes(name, group)
        return group

    def create_spike_recorder(self, target: Any) -> SpikeRecorder:
        self._check_alive()
        rec = SpikeRecorder()
        if isinstance(target, (CellGroup, RelayGroup)):
            for cell in target:
                rec.attach(self._spike_source(target, cell),
                           self._stable_id(cell))
        elif isinstance(target, SectionCell):
            found = False
            for group in self.population_nodes.values():
                if isinstance(group, CellGroup) and target in group.cells:
                    rec.attach(self._spike_source(group, target),
                               self._stable_id(target))
                    found = True
                    break
            if not found:
                raise KeyError("SectionCell is not registered in this adapter")
        elif isinstance(target, (list, tuple)):
            for cell in target:
                ident = self._stable_id(cell)
                rec.attach(SpikeSource(obj=cell, threshold=-20.0), ident)
        elif hasattr(target, "spike_source"):
            source = target.spike_source()
            ident = self._stable_id(target)
            if isinstance(source, list):
                raise TypeError(
                    "pass the group, not a multi-cell source list, to "
                    "create_spike_recorder")
            rec.attach(source, ident)
        else:
            rec.attach(SpikeSource(obj=target, threshold=-20.0),
                       self._stable_id(target))
        return rec

    def create_weight_recorder(self) -> WeightRecorder:
        self._check_alive()
        return WeightRecorder()

    def create_relay(self, pg_key: str, in_key: str, n: int, *,
                     rate: float = 0.0,
                     rate_param: PersonaParam | None = None,
                     pop_name: str | None = None,
                     noise_name: str | None = None,
                     noise_scale: float | None = None) -> tuple[RateGroup, RelayGroup]:
        self._check_alive()
        count = int(n)
        if count < 0:
            raise ValueError("n must be >= 0")
        base_rate = float(max(0.0, rate))
        stims = []
        ids = [self._next_cell_id() for _ in range(count)]
        for ident, stim in zip(ids, [h.NetStim() for _ in range(count)]):
            stim.start = 0.0
            # memcpg's Poisson-like reference recipe; a model may override
            # `noise`/`number` through RateGroup or Population.neuron.
            stim.noise = 1.0
            try:
                stim.noiseFromRandom(self._neuron_rng)
            except Exception:
                pass
            if base_rate > 0.0:
                stim.interval = 1000.0 / base_rate
                stim.number = 1_000_000_000
            else:
                stim.number = 0
            self._object_ids[id(stim)] = ident
            stims.append(stim)

        pg = RateGroup(pg_key, stims, ids=ids)
        relay = RelayGroup(in_key, stims, kind="forward", ids=ids,
                           rate_group=pg)
        self.register_nodes(pg_key, pg)
        self.register_nodes(in_key, relay)

        if rate_param is not None:
            name = pop_name or pg_key
            pop = Population(model="poisson_generator", n=count,
                             params={"rate": rate_param}, name=name)
            pop.node_collection = pg
            self.population_cfgs[name] = pop
            self._record_defaults(f"pop_{name}_rate", rate_param)
            self.site_recorder(f"pop_{name}_rate").record_sample(
                self._now_ms(), rate_param.mean, rate_param.sigma)
            try:
                self._set_nodes(
                    pg, {"rate": rate_param.sample_population(
                        count, center=rate_param.mean)})
            except Exception as exc:
                self.last_apply_errors.append((f"pop_{name}", str(exc)))

        if noise_name is not None:
            rec_name = f"producer:poisson:{noise_name}"
            value = (float(noise_scale) if noise_scale is not None
                     else base_rate)
            self.producer_recorder(f"poisson:{noise_name}", scale=value)
            self._relay_noise[pg_key] = rec_name
            self._relay_last[pg_key] = base_rate

        return pg, relay

    def set_relay_rate(self, pg_key: str, rate: float) -> None:
        self._check_alive()
        r = float(max(0.0, rate))
        self.get(pg_key).set({"rate": r})
        rec_name = self._relay_noise.get(pg_key)
        if rec_name is not None:
            prev = self._relay_last.get(pg_key, r)
            self._relay_last[pg_key] = r
            self.noise_recorders[rec_name].record_delta(
                self._now_ms(), r - prev)

    # ------------------------------------------------------------------
    # population creation
    # ------------------------------------------------------------------
    def create_population(
        self,
        cfg: Population,
        explored_means: Dict[str, float] | None = None,
    ) -> CellGroup:
        self._check_alive()
        if cfg.model not in self._cell_types:
            raise KeyError(
                f"unknown NEURON cell type {cfg.model!r}; registered: "
                f"{sorted(self._cell_types)}")
        factory = self._cell_types[cfg.model]
        group = factory.create(self, cfg)
        bad = [name for name in cfg.params if not group.has_param(name)]
        if bad:
            raise ValueError(
                f"Population {cfg.name!r} has unsupported NEURON parameter "
                f"path(s): {bad}")
        cfg.node_collection = group
        self.population_cfgs[cfg.name] = cfg
        self.population_nodes[cfg.name] = group

        for p_name, param_obj in cfg.params.items():
            key = f"pop_{cfg.name}_{p_name}"
            self._record_defaults(key, param_obj)
            self.site_recorder(key).record_sample(
                self._now_ms(), param_obj.mean, param_obj.sigma)

        sampled = cfg.sample_matrix(explored_means=explored_means)
        if sampled:
            try:
                self._set_nodes(group, sampled)
            except Exception as exc:
                self.last_apply_errors.append((f"pop_{cfg.name}", str(exc)))
        return group

    # ------------------------------------------------------------------
    # connection realization
    # ------------------------------------------------------------------
    @staticmethod
    def _model_for(
        cfg: Connection,
        explored_params: Dict[str, float] | None = None,
    ) -> tuple[str, Any | None, float]:
        model = str(cfg.model)
        weight = cfg.parameters.get("weight")
        if weight is None:
            seed = float(cfg.extra_syn_spec.get("weight", 0.0))
        elif explored_params is not None and "weight" in explored_params:
            seed = float(explored_params["weight"])
        else:
            seed = float(weight.mean)
        if model == "static_synapse":
            if seed == 0.0:
                raise ValueError(
                    "static_synapse weight is zero; the excitatory/"
                    "inhibitory identity is ambiguous")
            model = "static_exc" if seed > 0.0 else "static_inh"
        elif model == "stdp_synapse":
            model = "stdp_exc"
        return model, weight, seed

    def _pairs(self, source: Any, target: Any,
               cfg: Connection) -> List[Tuple[int, int]]:
        n_src = len(self._cells(source))
        n_tgt = len(self._cells(target))
        if n_src == 0 or n_tgt == 0:
            return []
        allow_autapses = bool(cfg.neuron.get(
            "allow_autapses", cfg.extra_syn_spec.get("allow_autapses", True)))
        allow_multapses = bool(cfg.neuron.get(
            "allow_multapses", cfg.extra_syn_spec.get("allow_multapses", True)))

        def allowed(i: int, j: int) -> bool:
            if not allow_autapses and source is target and i == j:
                return False
            return True

        rule = str(cfg.rule)
        pairs: List[Tuple[int, int]] = []
        if rule == "all_to_all":
            pairs = [(i, j) for i in range(n_src) for j in range(n_tgt)
                     if allowed(i, j)]
        elif rule == "one_to_one":
            pairs = [(i, i) for i in range(min(n_src, n_tgt))
                     if allowed(i, i)]
        elif rule == "pairwise_bernoulli":
            p = float(np.clip(cfg.get_effective_p(), 0.001, 1.0))
            pairs = [(i, j) for i in range(n_src) for j in range(n_tgt)
                     if allowed(i, j) and self._rng.random() < p]
        elif rule == "fixed_indegree":
            indegree = int(cfg.neuron.get(
                "indegree", cfg.extra_syn_spec.get("indegree", 1)))
            jitter = max(0, int(cfg.neuron.get(
                "indegree_jitter", cfg.extra_syn_spec.get(
                    "indegree_jitter", 0))))
            for j in range(n_tgt):
                k = indegree
                if jitter:
                    k += int(self._rng.integers(-jitter, jitter + 1))
                k = max(0, min(n_src if allow_multapses else n_src, k))
                candidates = [i for i in range(n_src) if allowed(i, j)]
                if not candidates or k == 0:
                    continue
                chosen = self._rng.choice(
                    candidates, size=min(k, len(candidates)),
                    replace=allow_multapses and k > len(candidates),
                )
                for i in np.atleast_1d(chosen):
                    pairs.append((int(i), j))
        elif rule == "fixed_outdegree":
            outdegree = int(cfg.neuron.get(
                "outdegree", cfg.extra_syn_spec.get("outdegree", 1)))
            jitter = max(0, int(cfg.neuron.get(
                "outdegree_jitter", cfg.extra_syn_spec.get(
                    "outdegree_jitter", 0))))
            for i in range(n_src):
                k = outdegree
                if jitter:
                    k += int(self._rng.integers(-jitter, jitter + 1))
                k = max(0, min(n_tgt, k))
                candidates = [j for j in range(n_tgt) if allowed(i, j)]
                if not candidates or k == 0:
                    continue
                chosen = self._rng.choice(
                    candidates, size=min(k, len(candidates)),
                    replace=allow_multapses and k > len(candidates),
                )
                for j in np.atleast_1d(chosen):
                    pairs.append((i, int(j)))
        else:
            raise ValueError(f"unsupported NEURON connection rule {rule!r}")

        if not allow_multapses:
            seen = set()
            unique = []
            for pair in pairs:
                if pair not in seen:
                    seen.add(pair)
                    unique.append(pair)
            pairs = unique
        return pairs

    def create_connection(
        self,
        cfg: Connection,
        explored_params: Dict[str, float] | None = None,
    ) -> str:
        self._check_alive()
        src_group = self.get(cfg.source)
        tgt_group = self.get(cfg.target)

        model, weight_param, seed = self._model_for(cfg, explored_params)
        factory = self._synapse_types.get(model)
        if factory is None:
            raise KeyError(
                f"unknown NEURON synapse type {model!r}; registered: "
                f"{sorted(self._synapse_types)}")
        is_plastic = bool(getattr(factory, "plastic", False))
        kind = getattr(factory, "kind", "signed")
        if kind == "signed" and not isinstance(factory, TapSynapseFactory):
            # Explicit static names are normalized; a resolved static_synapse
            # has already chosen its side above.
            kind = "inh" if model.endswith("_inh") else "exc"

        # Hard identity invariant before any point process is created.
        if weight_param is not None and not is_plastic:
            enforce_synaptic_sign(weight_param, kind)
            if explored_params is not None and "weight" in explored_params:
                seed = float(explored_params["weight"])
            else:
                seed = float(weight_param.mean)
        if not is_plastic:
            if kind == "inh" and seed > 0.0:
                seed = -abs(seed)
            elif kind == "exc" and seed < 0.0:
                seed = abs(seed)

        if isinstance(tgt_group, RelayGroup):
            if not tgt_group.can_target:
                raise ValueError(
                    f"relay {cfg.target!r} is a source-only forward relay; "
                    "connect into a tap relay or a CellGroup")
            if is_plastic:
                raise ValueError("STDP into a readout relay is not supported")
            factory = TapSynapseFactory()
            kind = "signed"

        if not isinstance(tgt_group, (CellGroup, RelayGroup)):
            raise TypeError(
                f"NEURON cannot connect into {type(tgt_group).__name__}")
        if isinstance(src_group, RateGroup):
            raise TypeError(
                f"rate group {cfg.source!r} is not a spike source; connect "
                "from its RelayGroup instead")
        if isinstance(src_group, RelayGroup) and not src_group.can_source:
            raise ValueError(
                f"relay {cfg.source!r} is a readout tap and cannot be a "
                "connection source")

        delay_param = cfg.parameters.get("delay")
        if (explored_params is not None and "delay" in explored_params):
            delay_seed = float(explored_params["delay"])
        elif delay_param is not None:
            delay_seed = float(delay_param.mean)
        else:
            delay_seed = float(cfg.extra_syn_spec.get("delay", 1.0))
        if delay_seed <= 0.0:
            raise ValueError("synaptic delay must be > 0")

        weight_cv = float(cfg.neuron.get(
            "weight_cv", cfg.extra_syn_spec.get("weight_cv", 0.0)))
        delay_cv = float(cfg.neuron.get(
            "delay_cv", cfg.extra_syn_spec.get("delay_cv", 0.0)))
        if weight_cv < 0.0 or delay_cv < 0.0:
            raise ValueError("weight_cv and delay_cv must be >= 0")
        threshold_override = cfg.neuron.get(
            "threshold", cfg.extra_syn_spec.get("threshold", None))
        param_values = {name: float(
            explored_params[name] if explored_params and name in explored_params
            else param.mean) for name, param in cfg.parameters.items()}
        # Creation-only static values (e.g. syn.e/tau) are not PERSONA sites,
        # but they still belong on the point process / NetCon.
        _reserved = {
            "weight", "delay", "weight_cv", "delay_cv", "threshold",
            "indegree", "outdegree", "indegree_jitter", "outdegree_jitter",
            "allow_autapses", "allow_multapses",
        }
        for key, value in cfg.extra_syn_spec.items():
            if key in _reserved:
                continue
            try:
                param_values[str(key)] = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"extra_syn_spec[{key!r}] must be numeric for NEURON")

        src_cells = self._cells(src_group)
        tgt_cells = self._cells(tgt_group)
        pairs = self._pairs(src_group, tgt_group, cfg)
        handles = []
        for i, j in pairs:
            src_cell = src_cells[i]
            tgt_cell = tgt_cells[j]
            presyn = self._spike_source(src_group, src_cell)
            postsyn = self._spike_source(tgt_group, tgt_cell)

            weight = float(seed)
            if weight_cv > 0.0:
                weight = float(self._rng.normal(
                    seed, abs(seed) * weight_cv))
                if kind == "inh":
                    weight = -abs(weight)
                else:
                    weight = abs(weight)

            delay = float(delay_seed)
            if delay_cv > 0.0:
                delay = float(self._rng.normal(
                    delay_seed, abs(delay_seed) * delay_cv))
                delay = max(float(h.dt), delay)

            if isinstance(tgt_group, CellGroup):
                pool = tgt_group.cell_type.allocate_synapse(
                    tgt_cell, getattr(factory, "pool_kind", kind))
            else:
                pool = None

            ctx = SynapseContext(
                registry=self,
                conn_key=(cfg.key if cfg.key is not None
                          else f"{cfg.source}_{cfg.target}"),
                source=cfg.source,
                target=cfg.target,
                presyn_cell=src_cell,
                postsyn_cell=tgt_cell,
                presyn_source=presyn,
                postsyn_pool=pool,
                weight=weight,
                delay=delay,
                params=param_values,
                weight_recorder=cfg.weight_recorder,
                threshold=(presyn.threshold if threshold_override is None
                           else float(threshold_override)),
                postsyn_source=postsyn,
                presyn_id=self._stable_id(src_cell),
                postsyn_id=self._stable_id(tgt_cell),
            )
            handle = factory.create(self, ctx)
            if cfg.weight_recorder is not None:
                cfg.weight_recorder.attach(handle)
            handles.append(handle)

        conn_key = cfg.key if cfg.key is not None else f"{cfg.source}_{cfg.target}"
        self.connection_cfgs[conn_key] = cfg
        self.connection_groups[conn_key] = SynapseGroup(
            key=conn_key,
            source=cfg.source,
            target=cfg.target,
            model=model,
            handles=handles,
            factory=factory,
            weight_recorder=cfg.weight_recorder,
        )
        self._syn_counts[conn_key] = len(handles)
        if cfg.weight_recorder is not None:
            self._plastic_n_syn[conn_key] = len(handles)

        for p_name, param_obj in cfg.parameters.items():
            site_name = self._canonical_param_name(cfg, p_name)
            key = f"conn_{conn_key}_{site_name}"
            self._record_defaults(key, param_obj)
            self.site_recorder(key).record_sample(
                self._now_ms(), param_obj.mean, param_obj.sigma)

        if cfg.weight_recorder is not None:
            wmax = None
            for candidate in ("wmax", "Wmax", "stdp.wmax", "stdp.Wmax"):
                if candidate in cfg.parameters:
                    wmax = cfg.parameters[candidate]
                    break
            self.producer_recorder(
                f"stdp:{conn_key}",
                scale=float(wmax.mean) if wmax is not None else None)
        return conn_key

    # ------------------------------------------------------------------
    # synapse views and plasticity
    # ------------------------------------------------------------------
    def synapses(self, conn_key: str) -> SynapseGroup:
        self._check_alive()
        try:
            return self.connection_groups[conn_key]
        except KeyError as exc:
            raise KeyError(
                f"connection {conn_key!r} not found in registry") from exc

    def synapse_count(self, conn_key: str) -> int:
        n = self._syn_counts.get(conn_key)
        if n is None:
            n = len(self.synapses(conn_key))
            self._syn_counts[conn_key] = n
        return int(n)

    def finalize_connections(self) -> None:
        for key, group in self.connection_groups.items():
            self._syn_counts[key] = len(group)
            if self.connection_cfgs[key].weight_recorder is not None:
                self._plastic_n_syn[key] = len(group)
        self.recapture_defaults()

    def read_plastic_events(
        self, clear: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for key, cfg in self.connection_cfgs.items():
            wr = cfg.weight_recorder
            if wr is None:
                continue
            try:
                ev = wr.get("events")
                weights = np.asarray(ev.get("weights", []), dtype=np.float64)
                times = np.asarray(ev.get("times", []), dtype=np.float64)
                senders = np.asarray(ev.get("senders", []), dtype=np.int64)
                targets = np.asarray(ev.get("targets", []), dtype=np.int64)
            except Exception:
                weights = np.empty(0, dtype=np.float64)
                times = np.empty(0, dtype=np.float64)
                senders = np.empty(0, dtype=np.int64)
                targets = np.empty(0, dtype=np.int64)
            if clear:
                try:
                    wr.set({"n_events": 0})
                except Exception:
                    pass
            out[key] = {"times": times, "weights": weights,
                        "senders": senders, "targets": targets}
        return out

    def sample_plastic_weight_deltas(
        self, t_before: float, t_now: float) -> Dict[str, float]:
        key_now = float(t_now)
        if self._plastic_cache_key == key_now:
            return dict(self._plastic_cache)
        events = self.read_plastic_events(clear=True)
        out: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for key, ev in events.items():
            weights = ev["weights"]
            counts[key] = int(weights.size)
            cfg = self.connection_cfgs[key]
            seed = float(cfg.extra_syn_spec.get("weight", 0.0))
            n_syn = (int(self._plastic_n_syn.get(key) or 0)
                     or max(1, int(weights.size)))
            last_w = self._plastic_last_w.setdefault(key, {})
            delta, _ = plastic_delta_from_events(ev, last_w, seed, n_syn)
            out[key] = delta
        self._plastic_counts = counts
        self._plastic_cache_key = key_now
        self._plastic_cache = out
        for key, delta in out.items():
            self.producer_recorder(f"stdp:{key}").record_delta(key_now, delta)
        return dict(out)

    def last_plastic_counts(self) -> Dict[str, int]:
        return dict(self._plastic_counts)

    def sample_plastic_weight_stats(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for key, cfg in self.connection_cfgs.items():
            if cfg.weight_recorder is None:
                continue
            try:
                w = np.asarray(self.synapses(key).get("weight"),
                               dtype=np.float64)
            except Exception:
                continue
            if w.size:
                out[key] = {"mean": float(w.mean()), "min": float(w.min()),
                            "max": float(w.max())}
        return out

    def final_plastic_weights(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for key, cfg in self.connection_cfgs.items():
            if cfg.weight_recorder is None:
                continue
            try:
                w = np.asarray(self.synapses(key).get("weight"),
                               dtype=np.float64)
            except Exception:
                continue
            if w.size:
                out[key] = w
        return out

    # ------------------------------------------------------------------
    # parameters, sampling, apply
    # ------------------------------------------------------------------
    def get_all_optimizable_params(self) -> Dict[str, PersonaParam]:
        all_params: Dict[str, PersonaParam] = {}
        for pop_name, pop in self.population_cfgs.items():
            for p_name, param_obj in pop.params.items():
                all_params[f"pop_{pop_name}_{p_name}"] = param_obj
        for conn_key, conn in self.connection_cfgs.items():
            for p_name, param_obj in conn.parameters.items():
                site_name = self._canonical_param_name(conn, p_name)
                all_params[f"conn_{conn_key}_{site_name}"] = param_obj
        return all_params

    def sample_all_actions(self) -> Dict[str, float]:
        return {k: param.sample_exploration()
                for k, param in self.get_all_optimizable_params().items()}

    def apply_explored_parameters(
        self, sampled_values: Dict[str, float]) -> None:
        self._check_alive()
        t_now = self._now_ms()
        self.last_apply_errors = []

        for pop_name, pop in self.population_cfgs.items():
            if pop.node_collection is None:
                continue
            updates: Dict[str, Any] = {}
            for p_name, param_obj in pop.params.items():
                full_key = f"pop_{pop_name}_{p_name}"
                if full_key in sampled_values:
                    updates[p_name] = param_obj.sample_population(
                        pop.n, center=sampled_values[full_key])
            if not updates:
                continue
            try:
                self._set_nodes(pop.node_collection, updates)
            except Exception as exc:
                self.last_apply_errors.append((f"pop_{pop_name}", str(exc)))
                continue
            for p_name, param_obj in pop.params.items():
                full_key = f"pop_{pop_name}_{p_name}"
                if full_key in sampled_values:
                    self.site_recorder(full_key).record_sample(
                        t_now, float(sampled_values[full_key]),
                        param_obj.sigma)

        for conn_key, conn in self.connection_cfgs.items():
            updates: Dict[str, Any] = {}
            zero_syn: List[str] = []
            for p_name, param_obj in conn.parameters.items():
                site_name = self._canonical_param_name(conn, p_name)
                full_key = f"conn_{conn_key}_{site_name}"
                if full_key not in sampled_values:
                    continue
                val = float(sampled_values[full_key])
                if conn.weight_recorder is not None:
                    updates[site_name] = val
                else:
                    n_syn = self.synapse_count(conn_key)
                    if n_syn <= 0:
                        zero_syn.append(site_name)
                        continue
                    updates[site_name] = param_obj.sample_population(
                        n_syn, center=val).tolist()
            if updates:
                try:
                    self.synapses(conn_key).set(updates)
                except Exception as exc:
                    self.last_apply_errors.append(
                        (f"conn_{conn_key}", str(exc)))
                    continue
            for p_name, param_obj in conn.parameters.items():
                site_name = self._canonical_param_name(conn, p_name)
                full_key = f"conn_{conn_key}_{site_name}"
                if full_key not in sampled_values:
                    continue
                if site_name in zero_syn:
                    self.last_apply_errors.append(
                        (f"conn_{conn_key}",
                         f"{site_name}: zero-synapse projection, "
                         "nothing realized"))
                self.site_recorder(full_key).record_sample(
                    t_now, float(sampled_values[full_key]), param_obj.sigma)

    # ------------------------------------------------------------------
    # introspection and extension points
    # ------------------------------------------------------------------
    def get(self, name: str) -> Any:
        self._check_alive()
        if name not in self.population_nodes:
            raise KeyError(f"Population '{name}' not found in registry")
        return self.population_nodes[name]

    def get_population_cfg(self, name: str) -> Population:
        return self.population_cfgs[name]

    def register_cell_type(self, name: str, factory: Any) -> None:
        if not hasattr(factory, "create"):
            raise TypeError("cell factory must provide create(registry, cfg)")
        self._cell_types[str(name)] = factory

    def register_synapse_type(self, name: str, factory: Any) -> None:
        if not hasattr(factory, "create"):
            raise TypeError("synapse factory must provide create(registry, ctx)")
        self._synapse_types[str(name)] = factory

    def register_recorder(self, name: str, factory: Any) -> None:
        self._recorders[str(name)] = factory


NeuronRegistry = NeuronAdapter
