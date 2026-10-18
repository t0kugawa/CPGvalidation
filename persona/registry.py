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
        self.synapse_collections: Dict[str, Any] = {}
        self.noise_recorders: Dict[str, NoiseRecorder] = {}

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
        syn_col = nest.Connect(src_nodes, tgt_nodes, conn_spec=conn_spec, syn_spec=syn_spec)

        cfg.synapse_collection = syn_col
        conn_key = cfg.key if cfg.key is not None else f"{cfg.source}_{cfg.target}"
        self.connection_cfgs[conn_key] = cfg
        self.synapse_collections[conn_key] = syn_col
        return syn_col

    def sample_plastic_weight_deltas(self, t_before: float, t_now: float) -> Dict[str, float]:
        """
        Mean per-synapse weight change observed by each plastic projection's
        NEST weight_recorder within the last step, keyed by projection key.
        """
        out: Dict[str, float] = {}
        for key, cfg in self.connection_cfgs.items():
            wr = cfg.weight_recorder
            if wr is None:
                continue
            try:
                ev = wr.get("events")
                weights = np.asarray(ev.get("weights", []), dtype=np.float64)
                times = np.asarray(ev.get("times", []), dtype=np.float64)
            except Exception:
                out[key] = 0.0
                continue
            if weights.size == 0:
                out[key] = 0.0
                continue
            mask = times >= float(t_before)
            window = weights[mask] if mask.any() else weights
            out[key] = float(np.mean(window) - np.mean(weights))
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
        using its controlling standard deviation (sigma_ctrl).
        """
        params = self.get_all_optimizable_params()
        return {k: param.sample_exploration() for k, param in params.items()}

    def apply_explored_parameters(self, sampled_values: Dict[str, float]):
        """
        Applies exploratory values to active NEST populations and synapses where supported.
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
            syn_col = self.synapse_collections.get(conn_key)
            if syn_col is None:
                continue
            for p_name in conn.parameters.keys():
                full_key = f"conn_{conn_key}_{p_name}"
                if full_key in sampled_values:
                    val = float(sampled_values[full_key])
                    try:
                        syn_col.set({p_name: val})
                    except Exception:
                        pass
