from persona.parameter import PersonaParam
from persona.registry import Registry
from t2xcpg.params import PARAMS


class BrainStem:
    """
    Reticulospinal brainstem projection providing tonic drives to the spinal CPG:
    - BS to E: projects bilaterally to RGE.
    - BS to F: projects bilaterally to RGF.
    Integrated through the PERSONA registry so drive rates are optimizable.
    """
    def __init__(self,
                 registry: Registry,
                 n_bs: int = PARAMS["n_bs_default"],
                 bs_e_hz: float = PARAMS["bs_e_hz"],
                 bs_f_hz: float = PARAMS["bs_f_hz"],
                 prefix: str = "BS",
                 bs_base_hz: float | None = None,
                 speed_ref_mps: float = PARAMS["bs_speed_ref_mps"]):
        self.registry = registry
        self.n_bs = n_bs
        self.prefix = prefix
        self.bs_base_hz = (PARAMS["bs_base_fraction"] * float(bs_e_hz)
                           if bs_base_hz is None else float(bs_base_hz))
        self.speed_ref_mps = float(speed_ref_mps)

        # Optimizable parameters for BS drive rates
        self.p_rate_e = PersonaParam(mean=bs_e_hz,
                                     sigma=PARAMS["bs_rate_sigma"],
                                     min=PARAMS["bs_rate_min"],
                                     max=PARAMS["bs_rate_max"])
        self.p_rate_f = PersonaParam(mean=bs_f_hz,
                                     sigma=PARAMS["bs_rate_sigma"],
                                     min=PARAMS["bs_rate_min"],
                                     max=PARAMS["bs_rate_max"])

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

    def set_speed(self, speed_mps: float, ref_e: float | None = None,
                  ref_f: float | None = None):
        """Scale the BS drive with treadmill speed around a nonzero base.

        The reference rate at ``speed_ref_mps`` is the personalized BS drive;
        the base keeps the BS input present even before the belt starts.
        """
        speed = max(0.0, float(speed_mps))
        ratio = speed / max(self.speed_ref_mps, 1e-9)
        re = self.p_rate_e.mean if ref_e is None else float(ref_e)
        rf = self.p_rate_f.mean if ref_f is None else float(ref_f)
        rate_e = self.bs_base_hz + (re - self.bs_base_hz) * ratio
        rate_f = self.bs_base_hz + (rf - self.bs_base_hz) * ratio
        self.set_rates(rate_e, rate_f)


class BaseDrive:
    """
    Tonic reticulospinal base drive from `cpg_2legs_fast` (`base_pg -> base_in`
    projecting bilaterally onto RGE and RGF). The Poisson rate is a PERSONA
    parameter; the projection weights are ordinary static synapses.
    """
    def __init__(self, registry: Registry,
                 n_bs: int = PARAMS["n_bs_default"],
                 base_hz: float = PARAMS["base_hz"], prefix: str = "BASE"):
        self.registry = registry
        self.n_bs = n_bs
        self.prefix = prefix

        self.p_rate = PersonaParam(mean=base_hz,
                                   sigma=PARAMS["base_rate_sigma"],
                                   min=PARAMS["base_rate_min"],
                                   max=PARAMS["base_rate_max"])

        self.base_pg, self.base_in = registry.create_relay(
            "base_pg", f"{prefix}_in", n_bs,
            rate=base_hz, rate_param=self.p_rate, pop_name=f"{prefix}_pg")

    def set_rate(self, rate: float):
        self.registry.set_relay_rate("base_pg", rate)


