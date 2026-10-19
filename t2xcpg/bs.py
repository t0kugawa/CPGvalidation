from persona.parameter import PersonaParam
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
        self.p_rate_e = PersonaParam(mean=bs_e_hz, sigma=5.0, min=0.0, max=250.0)
        self.p_rate_f = PersonaParam(mean=bs_f_hz, sigma=5.0, min=0.0, max=250.0)

        # BS to E/F generator + parrot relays, registered through the registry
        self.bs_pg_e, self.bs_in_e = registry.create_relay(
            "bs_pg_e", f"{prefix}_in_e", n_bs,
            rate=bs_e_hz, rate_param=self.p_rate_e, pop_name=f"{prefix}_E")
        self.bs_pg_f, self.bs_in_f = registry.create_relay(
            "bs_pg_f", f"{prefix}_in_f", n_bs,
            rate=bs_f_hz, rate_param=self.p_rate_f, pop_name=f"{prefix}_F")

    def set_rates(self, rate_e: float, rate_f: float):
        self.registry.set_relay_rate("bs_pg_e", rate_e)
        self.registry.set_relay_rate("bs_pg_f", rate_f)


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

        self.p_rate = PersonaParam(mean=base_hz, sigma=5.0,
                                   min=0.0, max=250.0)

        self.base_pg, self.base_in = registry.create_relay(
            "base_pg", f"{prefix}_in", n_bs,
            rate=base_hz, rate_param=self.p_rate, pop_name=f"{prefix}_pg")

    def set_rate(self, rate: float):
        self.registry.set_relay_rate("base_pg", rate)
