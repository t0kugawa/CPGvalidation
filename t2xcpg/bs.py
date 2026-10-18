import nest

from persona.parameter import PersonaParam
from persona.population import Population
from persona.registry import Registry


class BrainStem:
    """
    Reticulospinal brainstem projection providing tonic drives to the spinal CPG:
    - BS to E: projects bilaterally to RGE.
    - BS to F: projects bilaterally to RGF.
    Integrated through the PERSONA registry so drive rates are optimizable.
    """
    def __init__(self,
                 registry: Registry,
                 n_bs: int = 50,
                 bs_e_hz: float = 60.0,
                 bs_f_hz: float = 60.0,
                 prefix: str = "BS"):
        self.registry = registry
        self.n_bs = n_bs
        self.prefix = prefix

        # Optimizable parameters for BS drive rates
        self.p_rate_e = PersonaParam(mean=bs_e_hz, sigma_param=2.0, sigma_ctrl=5.0, min=0.0, max=250.0)
        self.p_rate_f = PersonaParam(mean=bs_f_hz, sigma_param=2.0, sigma_ctrl=5.0, min=0.0, max=250.0)

        # BS to E generator and parrot relay
        self.bs_pg_e = nest.Create("poisson_generator", n_bs, params={"rate": float(bs_e_hz)})
        self.bs_in_e = nest.Create("parrot_neuron", n_bs)
        nest.Connect(self.bs_pg_e, self.bs_in_e, conn_spec={"rule": "one_to_one"})

        # BS to F generator and parrot relay
        self.bs_pg_f = nest.Create("poisson_generator", n_bs, params={"rate": float(bs_f_hz)})
        self.bs_in_f = nest.Create("parrot_neuron", n_bs)
        nest.Connect(self.bs_pg_f, self.bs_in_f, conn_spec={"rule": "one_to_one"})

        # Register in Registry for unified tracking
        self.registry.population_nodes[f"{prefix}_in_e"] = self.bs_in_e
        self.registry.population_nodes[f"{prefix}_in_f"] = self.bs_in_f

        pop_e = Population(model="poisson_generator", n=n_bs, params={"rate": self.p_rate_e}, name=f"{prefix}_E")
        pop_e.node_collection = self.bs_pg_e
        self.registry.population_cfgs[f"{prefix}_E"] = pop_e

        pop_f = Population(model="poisson_generator", n=n_bs, params={"rate": self.p_rate_f}, name=f"{prefix}_F")
        pop_f.node_collection = self.bs_pg_f
        self.registry.population_cfgs[f"{prefix}_F"] = pop_f

    def set_rates(self, rate_e: float, rate_f: float):
        self.bs_pg_e.set({"rate": float(max(0.0, rate_e))})
        self.bs_pg_f.set({"rate": float(max(0.0, rate_f))})


class BaseDrive:
    """
    Tonic reticulospinal base drive from `cpg_2legs_fast` (`base_pg -> base_in`
    projecting bilaterally onto RGE and RGF). The Poisson rate is a PERSONA
    parameter; the projection weights are ordinary static synapses.
    """
    def __init__(self, registry: Registry, n_bs: int = 50,
                 base_hz: float = 2.0, prefix: str = "BASE"):
        self.registry = registry
        self.n_bs = n_bs
        self.prefix = prefix

        self.p_rate = PersonaParam(mean=base_hz, sigma_param=2.0, sigma_ctrl=5.0,
                                   min=0.0, max=250.0)

        self.base_pg = nest.Create("poisson_generator", n_bs,
                                   params={"rate": float(base_hz)})
        self.base_in = nest.Create("parrot_neuron", n_bs)
        nest.Connect(self.base_pg, self.base_in, conn_spec={"rule": "one_to_one"})

        self.registry.population_nodes[f"{prefix}_in"] = self.base_in

        pop = Population(model="poisson_generator", n=n_bs,
                         params={"rate": self.p_rate}, name=f"{prefix}_pg")
        pop.node_collection = self.base_pg
        self.registry.population_cfgs[f"{prefix}_pg"] = pop

    def set_rate(self, rate: float):
        self.base_pg.set({"rate": float(max(0.0, rate))})
