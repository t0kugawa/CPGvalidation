import math
import nest
from typing import Any, Dict, List
import numpy as np

from persona.adapter import (
    Adapter,
    attrs_tuple,
    clamp_weight_sign,
    enforce_synaptic_sign,
    key_patterns,
    plastic_delta_from_events,
    resolve_site_keys,
)
from persona.parameter import PersonaParam
from persona.population import Population
from persona.connection import Connection
from persona.noise_recorder import NoiseRecorder
from persona.spring import SpringSpec


__all__ = ["NestAdapter", "clamp_weight_sign", "plastic_delta_from_events"]


class NestAdapter(Adapter):
    """
    Central repository for populations, connections, parameters, and noise recorders.
    Provides uniform interfaces for building NEST networks, sampling exploratory actions,
    and dynamically applying parameter changes.
    """
    def __init__(self):
        self.population_cfgs: Dict[str, Population] = {}
        self.connection_cfgs: Dict[str, Connection] = {}
        self.population_nodes: Dict[str, nest.NodeCollection] = {}
        self.noise_recorders: Dict[str, NoiseRecorder] = {}
        self._synapses: Dict[str, Any] = {}
        self._plastic_counts: Dict[str, int] = {}
        self._plastic_cache_key: float | None = None
        self._plastic_cache: Dict[str, float] = {}
        self._plastic_last_w: Dict[str, Dict[tuple, float]] = {}
        self._plastic_n_syn: Dict[str, int] = {}
        self._relay_noise: Dict[str, str] = {}
        self._relay_last: Dict[str, float] = {}
        self._syn_counts: Dict[str, int] = {}
        # Creation-time values per canonical site key, and the attributes the
        # caller pinned to them (PersonaParam exposes mean/sigma as learned).
        self._defaults: Dict[str, Dict[str, float]] = {}
        self._frozen: Dict[str, set[str]] = {}
        self._springs: Dict[str, SpringSpec] = {}
        # Failures of the realization path (create_population's set,
        # apply_explored_parameters); cleared on every apply, never swallowed.
        self.last_apply_errors: List[tuple[str, str]] = []

    def register_noise_recorder(self, recorder: NoiseRecorder):
        self.noise_recorders[recorder.name] = recorder

    def get_noise_recorder(self, name: str) -> NoiseRecorder | None:
        return self.noise_recorders.get(name)

    def site_recorder(self, full_key: str) -> NoiseRecorder:
        """Get-or-create the recorder of one personalizable parameter site."""
        name = f"site:{full_key}"
        rec = self.noise_recorders.get(name)
        if rec is None:
            rec = NoiseRecorder(name)
            self.register_noise_recorder(rec)
        return rec

    def producer_recorder(self, name: str,
                          scale: float | None = None) -> NoiseRecorder:
        """Get-or-create the recorder of one noise producer (e.g. 'stdp:cut_rge_L',
        'poisson:cut_L'); the registered name is prefixed with 'producer:'.

        `scale` is the producer's current value at registration, used to score
        its delta (`δ/R²`). It is fixed by the caller at registration time; a
        later call without `scale` leaves the existing value untouched.
        """
        full = name if name.startswith("producer:") else f"producer:{name}"
        rec = self.noise_recorders.get(full)
        if rec is None:
            rec = NoiseRecorder(full, scale=scale)
            self.register_noise_recorder(rec)
        elif scale is not None:
            rec.scale = float(scale) if float(scale) > 0.0 else None
        return rec

    def producer_recorders(self) -> Dict[str, NoiseRecorder]:
        """All producer recorders (STDP projections and sensory Poisson groups)."""
        return {n: r for n, r in self.noise_recorders.items()
                if n.startswith("producer:")}

    # --- creation defaults and freeze ---------------------------------------

    def _record_defaults(self, site_key: str, param: PersonaParam) -> None:
        """Snapshot the creation values freeze anchors to."""
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
        """Resolve exact canonical keys or fnmatch globs to sorted site keys."""
        return resolve_site_keys(self.get_all_optimizable_params(), keys)

    def recapture_defaults(self, keys=None) -> None:
        """Re-snapshot creation defaults after a post-creation mutation."""
        params = self.get_all_optimizable_params()
        resolved = tuple(params) if keys is None else self._resolve_site_keys(keys)
        for key in resolved:
            self._record_defaults(key, params[key])

    def freeze(self, keys, attrs=("mean",)) -> tuple[str, ...]:
        """Pin the named learned attributes of the matching sites to creation."""
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
        """Remove pins; values continue from where they are now."""
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

    # --- homeostatic springs ------------------------------------------------

    def create_spring(self, name: str, keys, *, power: float | None = None,
                      gain: float = 1.0,
                      max_power: float = 0.5) -> SpringSpec:
        """Create one spring over the matching sites' mean coordinates."""
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

    def spring_state(self, means: Dict[str, float], sigmas: Dict[str, float],
                     lr_t: float) -> Dict[str, Dict[str, float]]:
        """Per-site force and power for this step; frozen means are skipped."""
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

    # --- kernel lifecycle ---------------------------------------------------

    @classmethod
    def reset_kernel(
        cls,
        *,
        resolution_ms: float | None = None,
        v_init: float | None = None,
        cvode: bool = False,
        maxstep_ms: float | None = None,
    ) -> None:
        """Reset the NEST kernel; delegates to ``nest.ResetKernel``."""
        nest.ResetKernel()
        status: Dict[str, Any] = {}
        if resolution_ms is not None:
            status["resolution"] = float(resolution_ms)
        if status:
            nest.SetKernelStatus(status)

    def set_resolution(self, resolution_ms: float) -> None:
        nest.SetKernelStatus({"resolution": float(resolution_ms)})

    def initialize(self, v_init: float | None = None) -> None:
        # NEST initializes its own state when the kernel is reset / first used.
        return None

    def simulate(self, dt_ms: float) -> float:
        nest.Simulate(float(dt_ms))
        return self.time_ms

    @property
    def time_ms(self) -> float:
        try:
            return float(nest.GetKernelStatus("biological_time"))
        except Exception:
            return 0.0

    @property
    def dt_ms(self) -> float:
        try:
            return float(nest.GetKernelStatus("resolution"))
        except Exception:
            return 0.0

    def _now_ms(self) -> float:
        return self.time_ms

    def register_nodes(self, name: str, nodes: Any) -> Any:
        """Registers raw NEST nodes (readout parrots, relay parrots) by name."""
        self.population_nodes[name] = nodes
        return nodes

    def create_parrot_nodes(self, name: str, n: int) -> Any:
        """Creates and registers `n` parrot readout neurons (no intrinsic drive)."""
        return self.register_nodes(name, nest.Create("parrot_neuron", n))

    def create_spike_recorder(self, target: Any) -> Any:
        """Creates a memory spike recorder wired to `target`."""
        rec = nest.Create("spike_recorder", params={"record_to": "memory"})
        nest.Connect(target, rec)
        return rec

    def create_weight_recorder(self) -> Any:
        """Creates a weight recorder for one plastic projection's STDP model."""
        return nest.Create("weight_recorder")

    def _set_nodes(self, nodes: Any, update_dict: Dict[str, Any]) -> None:
        """
        Realizes per-subject values on a node collection: one vectorized `set`
        first (neurons accept arrays), then a per-node loop — device-only
        parameters such as a Poisson `rate` may reject arrays (spec §5.2).
        Raises with both messages when neither path works, so the caller can
        surface the failure instead of silently keeping the old value.
        """
        try:
            nodes.set(update_dict)
            return
        except Exception as arr_exc:
            arr_msg = str(arr_exc)
        try:
            for i, node in enumerate(nodes):
                per = {}
                for key, val in update_dict.items():
                    arr = np.asarray(val, dtype=np.float64).ravel()
                    per[key] = float(arr[0] if arr.size == 1 else arr[i])
                node.set(per)
        except Exception as loop_exc:
            raise RuntimeError(
                f"array set failed ({arr_msg}); "
                f"per-node realization failed ({loop_exc})") from arr_exc

    def create_population(self, cfg: Population, explored_means: Dict[str, float] | None = None) -> nest.NodeCollection:
        """
        Instantiates a NEST neuron population according to the Population specification.
        """
        nodes = nest.Create(cfg.model, cfg.n)
        cfg.node_collection = nodes
        self.population_cfgs[cfg.name] = cfg
        self.population_nodes[cfg.name] = nodes

        # One site recorder per optimizable parameter; the seed value it is
        # about to realize is the first recorded (sample, sigma).
        for p_name, param_obj in cfg.params.items():
            key = f"pop_{cfg.name}_{p_name}"
            self._record_defaults(key, param_obj)
            self.site_recorder(key).record_sample(
                self._now_ms(), param_obj.mean, param_obj.sigma)

        sampled = cfg.sample_matrix(explored_means=explored_means)
        if sampled:
            try:
                self._set_nodes(nodes, sampled)
            except Exception as exc:
                # surfaced, never swallowed (spec §5.5: no silent failures)
                self.last_apply_errors.append((f"pop_{cfg.name}", str(exc)))

        return nodes

    def create_connection(self, cfg: Connection, explored_params: Dict[str, float] | None = None) -> Any:
        """
        Connects two registered populations in NEST according to the Connection specification.
        """
        src_nodes = self.get(cfg.source)
        tgt_nodes = self.get(cfg.target)

        effective_p = float(np.clip(cfg.get_effective_p(), 0.001, 1.0))

        if cfg.rule == "pairwise_bernoulli":
            conn_spec = {"rule": "pairwise_bernoulli", "p": effective_p}
        elif cfg.rule == "all_to_all":
            conn_spec = {"rule": "all_to_all"}
        elif cfg.rule == "one_to_one":
            conn_spec = {"rule": "one_to_one"}
        else:
            conn_spec = {"rule": cfg.rule}

        # Type/sign identity: a static weight may not change from excitatory to
        # inhibitory (or back).  An explicit inhibitory type normalizes even a
        # positive magnitude to the signed-negative representation.  STDP
        # weights are seeded, not optimizable sites.
        effective_params = (dict(explored_params)
                            if explored_params is not None else None)
        weight = cfg.parameters.get("weight")
        if (weight is not None and cfg.weight_recorder is None
                and "stdp" not in str(cfg.model).lower()):
            model = str(cfg.model).lower()
            kind = ("inh" if "inh" in model
                    else "exc" if "exc" in model
                    else "signed")
            enforce_synaptic_sign(weight, kind)
            if effective_params is not None and "weight" in effective_params:
                value = float(effective_params["weight"])
                if kind == "inh" and value > 0.0:
                    effective_params["weight"] = -abs(value)
                elif kind == "exc" and value < 0.0:
                    effective_params["weight"] = abs(value)

        syn_spec = cfg.sample_syn_spec(explored_params=effective_params)
        nest.Connect(src_nodes, tgt_nodes, conn_spec=conn_spec, syn_spec=syn_spec)

        conn_key = cfg.key if cfg.key is not None else f"{cfg.source}_{cfg.target}"
        self.connection_cfgs[conn_key] = cfg
        for p_name, param_obj in cfg.parameters.items():
            key = f"conn_{conn_key}_{p_name}"
            self._record_defaults(key, param_obj)
            self.site_recorder(key).record_sample(
                self._now_ms(), param_obj.mean, param_obj.sigma)
        if cfg.weight_recorder is not None:
            # Noise scale: the projection's current weight ceiling at
            # registration (not the live Wmax site, which PERSONA may move).
            wmax = cfg.parameters.get("Wmax")
            self.producer_recorder(
                f"stdp:{conn_key}",
                scale=float(wmax.mean) if wmax is not None else None)
        return conn_key

    def create_relay(self, pg_key: str, in_key: str, n: int, *,
                     rate: float = 0.0,
                     rate_param: "PersonaParam | None" = None,
                     pop_name: str | None = None,
                     noise_name: str | None = None,
                     noise_scale: float | None = None) -> tuple[Any, Any]:
        """
        Poisson generator -> parrot relay, registered so no module creates or
        connects these nodes directly:

        - `pg_key`/`in_key` register the generator and the parrot in
          `population_nodes`;
        - `rate_param` makes the generator an optimizable `Population` under
          `pop_name` (defaults to `pg_key`);
        - `noise_name` attaches a producer recorder that `set_relay_rate` feeds
          with the per-step rate delta;
        - `noise_scale` is the producer's current value at registration (the
          nominal `rate` by default), used to score that delta (`δ/R²`). A zero
          rest rate (CUT) must pass an explicit value or the producer has no
          influence.
        """
        pg = nest.Create("poisson_generator", n, params={"rate": float(max(0.0, rate))})
        parrot = nest.Create("parrot_neuron", n)
        nest.Connect(pg, parrot, conn_spec={"rule": "one_to_one"})
        self.register_nodes(pg_key, pg)
        self.register_nodes(in_key, parrot)

        if rate_param is not None:
            name = pop_name or pg_key
            pop = Population(model="poisson_generator", n=n,
                             params={"rate": rate_param}, name=name)
            pop.node_collection = pg
            self.population_cfgs[name] = pop
            self._record_defaults(f"pop_{name}_rate", rate_param)
            self.site_recorder(f"pop_{name}_rate").record_sample(
                self._now_ms(), rate_param.mean, rate_param.sigma)
            # realize the seed dispersion per device, exactly like
            # create_population does for its neurons (spec §5.2):
            # recorded = applied from the first step
            try:
                self._set_nodes(
                    pg, {"rate": rate_param.sample_population(
                        n, center=rate_param.mean)})
            except Exception as exc:
                self.last_apply_errors.append((f"pop_{name}", str(exc)))

        if noise_name is not None:
            rec_name = f"producer:poisson:{noise_name}"
            value = (float(noise_scale) if noise_scale is not None
                     else float(max(0.0, rate)))
            self.producer_recorder(f"poisson:{noise_name}", scale=value)
            self._relay_noise[pg_key] = rec_name
            self._relay_last[pg_key] = float(max(0.0, rate))

        return pg, parrot

    def set_relay_rate(self, pg_key: str, rate: float):
        """Sets a relay generator's rate; a wired producer recorder receives the
        per-step delta (read-and-clear upstream)."""
        r = float(max(0.0, rate))
        self.get(pg_key).set({"rate": r})
        rec_name = self._relay_noise.get(pg_key)
        if rec_name is not None:
            prev = self._relay_last.get(pg_key, r)
            self._relay_last[pg_key] = r
            self.noise_recorders[rec_name].record_delta(self._now_ms(), r - prev)

    def synapses(self, conn_key: str) -> Any:
        """
        SynapseCollection for one projection. NEST recycles its ConnectionObject
        handles whenever new connections are created, so a collection captured
        before the network was complete can silently resolve to unrelated
        synapses; `finalize_connections` re-captures them once construction is
        done, and this falls back to a fresh query otherwise.
        """
        col = self._synapses.get(conn_key)
        if col is None:
            cfg = self.connection_cfgs[conn_key]
            col = nest.GetConnections(source=self.get(cfg.source),
                                      target=self.get(cfg.target),
                                      synapse_model=cfg.model)
            self._synapses[conn_key] = col
        return col

    def synapse_count(self, conn_key: str) -> int:
        """Number of synapses of one projection (cached by `finalize_connections`)."""
        n = self._syn_counts.get(conn_key)
        if n is None:
            try:
                n = int(len(self.synapses(conn_key)))
            except Exception:
                n = 0
            self._syn_counts[conn_key] = n
        return int(n)

    def finalize_connections(self):
        """
        Captures one SynapseCollection per projection; call once the network is
        fully built (after the last nest.Connect). Also caches every projection's
        synapse count — the realization length of static weight arrays and the
        N_syn of the STDP delta normalization.
        """
        self._synapses = {
            key: nest.GetConnections(source=self.get(cfg.source),
                                     target=self.get(cfg.target),
                                     synapse_model=cfg.model)
            for key, cfg in self.connection_cfgs.items()
        }
        self._syn_counts = {}
        self._plastic_n_syn = {}
        for key, cfg in self.connection_cfgs.items():
            try:
                self._syn_counts[key] = int(len(self._synapses[key]))
            except Exception:
                self._syn_counts[key] = 0
            if cfg.weight_recorder is not None:
                self._plastic_n_syn[key] = self._syn_counts[key]
        # Builders may mutate a PersonaParam after create_population (the
        # flexor pool is silenced this way); the anchor is the effective value
        # at the end of construction, not the provisional one at registration.
        self.recapture_defaults()

    def read_plastic_events(self, clear: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Drains each plastic projection's NEST weight_recorder and returns the
        events accumulated since the previous call
        ({key: {"times", "weights", "senders", "targets"}}).
        With `clear` the recorder is emptied afterwards so a long run neither
        grows memory without bound nor re-scans the whole history every step.
        """
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

    def sample_plastic_weight_deltas(self, t_before: float, t_now: float) -> Dict[str, float]:
        """
        Per-step STDP signal of each plastic projection: the per-synapse mean
        weight increment Σ(w − last_w)/N_syn over the events its weight recorder
        accumulated since the previous step. The recorder is drained, so the
        cost per step is proportional to the new events only; repeated calls
        within one step (one per limb) reuse the drained batch, and each
        projection's δ is recorded exactly once into its `stdp:<key>` producer
        recorder (read-and-clear upstream).
        """
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
            n_syn = int(self._plastic_n_syn.get(key) or 0) or max(1, int(weights.size))
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
        """Number of STDP weight updates each projection recorded last step."""
        return dict(self._plastic_counts)

    def sample_plastic_weight_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Current weight distribution (mean/min/max) of every plastic projection,
        read from the live synapse collections. Used to log the STDP weight
        trajectory itself (the recorders only report the updates).
        """
        out: Dict[str, Dict[str, float]] = {}
        for key, cfg in self.connection_cfgs.items():
            if cfg.weight_recorder is None:
                continue
            try:
                w = np.asarray(self.synapses(key).get("weight"), dtype=np.float64)
            except Exception:
                continue
            if w.size:
                out[key] = {"mean": float(w.mean()), "min": float(w.min()),
                            "max": float(w.max())}
        return out

    def final_plastic_weights(self) -> Dict[str, np.ndarray]:
        """Snapshot of every plastic projection's weight vector (end of run)."""
        out: Dict[str, np.ndarray] = {}
        for key, cfg in self.connection_cfgs.items():
            if cfg.weight_recorder is None:
                continue
            try:
                w = np.asarray(self.synapses(key).get("weight"), dtype=np.float64)
            except Exception:
                continue
            if w.size:
                out[key] = w
        return out

    def get(self, name: str) -> nest.NodeCollection:
        if name not in self.population_nodes:
            raise KeyError(f"Population '{name}' not found in registry")
        return self.population_nodes[name]

    def get_population_cfg(self, name: str) -> Population:
        return self.population_cfgs[name]

    def get_all_optimizable_params(self) -> Dict[str, PersonaParam]:
        """
        Gathers all PersonaParam instances across populations and connections
        into a flat dictionary with canonical keys.
        """
        all_params: Dict[str, PersonaParam] = {}

        # Population parameters
        for pop_name, pop in self.population_cfgs.items():
            for p_name, param_obj in pop.params.items():
                key = f"pop_{pop_name}_{p_name}"
                all_params[key] = param_obj

        # Connection parameters (connection probability `p` is excluded: NEST
        # cannot re-apply it post-hoc without rebuilding synapses).
        for conn_key, conn in self.connection_cfgs.items():
            for p_name, param_obj in conn.parameters.items():
                key = f"conn_{conn_key}_{p_name}"
                all_params[key] = param_obj

        return all_params

    def sample_all_actions(self) -> Dict[str, float]:
        """
        Draws an exploratory action (parameter mean) for every registered PersonaParam
        using its single standard deviation (sigma).
        """
        params = self.get_all_optimizable_params()
        return {k: param.sample_exploration() for k, param in params.items()}

    def apply_explored_parameters(self, sampled_values: Dict[str, float]):
        """
        Realizes the sampled action on the live network:

        - populations: one dispersion draw per subject (array `set`, per-node
          loop fallback for device-only parameters);
        - static synapses: one array realization per projection (`weight` of
          every synapse draws around the action);
        - STDP connections: scalar hyper-parameters (one synapse model per
          projection).

        `recorded = applied`: each site recorder stores exactly the (x̂, σ) the
        network realized. Failures land in `last_apply_errors` (cleared per
        call) — never swallowed.

        TODO(perf): this per-step re-sampling + `set()` of every neuron and every
        synapse dominates the step cost at full population sizes (~1.5 s/step);
        the exploration is required for personalization, so it is left in place.
        """
        t_now = self._now_ms()
        self.last_apply_errors = []

        # Apply to populations
        for pop_name, pop in self.population_cfgs.items():
            if pop.node_collection is None:
                continue
            update_dict = {}
            for p_name, param_obj in pop.params.items():
                full_key = f"pop_{pop_name}_{p_name}"
                if full_key in sampled_values:
                    val = sampled_values[full_key]
                    update_dict[p_name] = param_obj.sample_population(pop.n, center=val)
            if not update_dict:
                continue
            try:
                self._set_nodes(pop.node_collection, update_dict)
            except Exception as exc:
                self.last_apply_errors.append((f"pop_{pop_name}", str(exc)))
                continue
            for p_name, param_obj in pop.params.items():
                full_key = f"pop_{pop_name}_{p_name}"
                if full_key in sampled_values:
                    # recorded = applied: the site recorder holds exactly
                    # the sample the network just realized
                    self.site_recorder(full_key).record_sample(
                        t_now, float(sampled_values[full_key]), param_obj.sigma)

        # Apply to connections (synapses)
        for conn_key, conn in self.connection_cfgs.items():
            updates: Dict[str, Any] = {}
            zero_syn: list[str] = []
            for p_name, param_obj in conn.parameters.items():
                full_key = f"conn_{conn_key}_{p_name}"
                if full_key not in sampled_values:
                    continue
                val = float(sampled_values[full_key])
                if conn.weight_recorder is not None:
                    # STDP projection: the sampled hyper-parameters are scalars
                    updates[p_name] = val
                else:
                    # static synapses: one array realization for the projection
                    n_syn = self.synapse_count(conn_key)
                    if n_syn <= 0:
                        # surfaced, never swallowed (spec §5.5): the sample is
                        # still recorded (applied to an empty projection), and
                        # the emptiness is reported
                        zero_syn.append(p_name)
                        continue
                    updates[p_name] = param_obj.sample_population(
                        n_syn, center=val).tolist()
            if updates:
                try:
                    self.synapses(conn_key).set(updates)
                except Exception as exc:
                    self.last_apply_errors.append((f"conn_{conn_key}", str(exc)))
                    continue
            for p_name, param_obj in conn.parameters.items():
                full_key = f"conn_{conn_key}_{p_name}"
                if full_key not in sampled_values:
                    continue
                if p_name in zero_syn:
                    self.last_apply_errors.append(
                        (f"conn_{conn_key}",
                         f"{p_name}: zero-synapse projection, nothing realized"))
                self.site_recorder(full_key).record_sample(
                    t_now, float(sampled_values[full_key]), param_obj.sigma)
