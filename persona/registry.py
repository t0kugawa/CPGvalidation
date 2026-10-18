import nest
from typing import Dict, Any
import numpy as np

from persona.parameter import PersonaParam
from persona.population import Population
from persona.connection import Connection
from persona.noise_recorder import NoiseRecorder


class Registry:
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
        self._plastic_prev_mean: Dict[str, float] = {}
        self._plastic_counts: Dict[str, int] = {}
        self._plastic_cache_key: float | None = None
        self._plastic_cache: Dict[str, float] = {}

    def register_noise_recorder(self, recorder: NoiseRecorder):
        self.noise_recorders[recorder.name] = recorder

    def get_noise_recorder(self, name: str) -> NoiseRecorder | None:
        return self.noise_recorders.get(name)

    def create_population(self, cfg: Population, explored_means: Dict[str, float] | None = None) -> nest.NodeCollection:
        """
        Instantiates a NEST neuron population according to the Population specification.
        """
        nodes = nest.Create(cfg.model, cfg.n)
        cfg.node_collection = nodes
        self.population_cfgs[cfg.name] = cfg
        self.population_nodes[cfg.name] = nodes

        sampled = cfg.sample_matrix(explored_means=explored_means)
        for key, values in sampled.items():
            try:
                nodes.set({key: values})
            except Exception:
                # Some properties in NEST are scalar or have specific setters
                try:
                    nodes.set({key: float(np.mean(values))})
                except Exception:
                    pass

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

        syn_spec = cfg.sample_syn_spec(explored_params=explored_params)
        nest.Connect(src_nodes, tgt_nodes, conn_spec=conn_spec, syn_spec=syn_spec)

        conn_key = cfg.key if cfg.key is not None else f"{cfg.source}_{cfg.target}"
        self.connection_cfgs[conn_key] = cfg
        return conn_key

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

    def finalize_connections(self):
        """
        Captures one SynapseCollection per projection; call once the network is
        fully built (after the last nest.Connect).
        """
        self._synapses = {
            key: nest.GetConnections(source=self.get(cfg.source),
                                     target=self.get(cfg.target),
                                     synapse_model=cfg.model)
            for key, cfg in self.connection_cfgs.items()
        }

    def read_plastic_events(self, clear: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Drains each plastic projection's NEST weight_recorder and returns the
        events accumulated since the previous call ({key: {"times", "weights"}}).
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
            except Exception:
                weights = np.empty(0, dtype=np.float64)
                times = np.empty(0, dtype=np.float64)
            if clear:
                try:
                    wr.set({"n_events": 0})
                except Exception:
                    pass
            out[key] = {"times": times, "weights": weights}
        return out

    def sample_plastic_weight_deltas(self, t_before: float, t_now: float) -> Dict[str, float]:
        """
        Mean per-synapse weight change each plastic projection's weight_recorder
        observed within the last step, keyed by projection key. The recorder is
        drained (and cleared) so the cost per step is proportional to the new
        events only; repeated calls within one step (one per limb) reuse the
        drained result instead of finding the recorders empty.
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
            if weights.size == 0:
                out[key] = 0.0
                continue
            current = float(np.mean(weights))
            previous = self._plastic_prev_mean.get(key)
            out[key] = 0.0 if previous is None else current - previous
            self._plastic_prev_mean[key] = current
        self._plastic_counts = counts
        self._plastic_cache_key = key_now
        self._plastic_cache = out
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
        Applies exploratory values to active NEST populations and synapses where supported.
        TODO(perf): this per-step re-sampling + `set()` of every neuron and every
        synapse dominates the step cost at full population sizes (~1.5 s/step);
        the exploration is required for personalization, so it is left in place.
        """
        # Apply to populations
        for pop_name, pop in self.population_cfgs.items():
            if pop.node_collection is None:
                continue
            update_dict = {}
            for p_name, param_obj in pop.params.items():
                full_key = f"pop_{pop_name}_{p_name}"
                if full_key in sampled_values:
                    val = sampled_values[full_key]
                    sampled_arr = param_obj.sample_population(pop.n, center=val)
                    update_dict[p_name] = sampled_arr.tolist()
            if update_dict:
                try:
                    pop.node_collection.set(update_dict)
                except Exception:
                    pass

        # Apply to connections (synapses)
        for conn_key, conn in self.connection_cfgs.items():
            updates = {}
            for p_name in conn.parameters.keys():
                full_key = f"conn_{conn_key}_{p_name}"
                if full_key in sampled_values:
                    updates[p_name] = float(sampled_values[full_key])
            if not updates:
                continue
            try:
                self.synapses(conn_key).set(updates)
            except Exception:
                pass
