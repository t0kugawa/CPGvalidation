from typing import Dict, Any
import nest

from persona.registry import Registry
from persona.connection import Connection
from persona.parameter import PersonaParam
from model.adapter import CPGAdapter, CPGInput, CPGOutput
from t2xcpg.bs import BrainStem, BaseDrive
from t2xcpg.leg import Leg


class L2CPG(CPGAdapter):
    """
    Two-limb spinal CPG network (Left and Right hindlimbs) with full commissural
    pathways matching Diagram 2, driven by Brainstem projections and modulated by
    sensory afferent feedback (CUT, Ia, Ib).
    Implements CPGAdapter to seamlessly interface with the MuJoCo biomechanical model.
    """
    def __init__(self,
                 registry: Registry,
                 bs: BrainStem,
                 legs: Dict[str, Leg],
                 dt_ms: float = 10.0,
                 p_comm: float = 0.40,
                 leading_leg: str = "L",
                 leading_gain: float = 1.5,
                 base: "BaseDrive | None" = None):
        self.registry = registry
        self.bs = bs
        self.legs = legs
        self.dt_ms = dt_ms
        self.current_time_ms = 0.0
        self.leading_leg = leading_leg
        self.leading_gain = float(leading_gain)
        self.base = base

        # Build Brainstem drives and Commissural Cross-Connections
        self._build_bs_connections()
        self._build_base_connections()
        self._build_commissural_connections(p_comm=p_comm)
        self.registry.finalize_connections()

    def _param(self, name: str, mean: float, sigma: float = 0.1,
               min_val: float = -100.0, max_val: float = 100.0) -> PersonaParam:
        return PersonaParam(
            mean=mean,
            sigma=sigma,
            min=min_val,
            max=max_val
        )

    def _connect(self, src: str, tgt: str, weight: float, p: float = 0.40, delay: float = 3.0) -> Any:
        w_param = self._param("weight", weight, abs(weight) * 0.1,
                              min_val=-100.0 if weight < 0 else 0.0,
                              max_val=0.0 if weight < 0 else 100.0)
        cfg = Connection(
            source=src,
            target=tgt,
            model="static_synapse",
            parameters={"weight": w_param},
            rule="pairwise_bernoulli",
            p=float(p),
            extra_syn_spec={"delay": float(delay)}
        )
        return self.registry.create_connection(cfg)

    def _build_bs_connections(self):
        """
        Connects Brainstem to Extensor (RGE) and Flexor (RGF) generators of all
        legs. The brainstem drive frequency is equal for both legs; the leading
        leg receives a larger synaptic weight so the first step emerges from it.

        The reference's BS->RG is an STDP projection that adapts down. A static
        18 pins RG at the 0.2 ms step cap; a seed of 4.0 puts it in the
        reference's measured band (85-229 Hz). PERSONA refines from here.
        """
        for leg_key, leg in self.legs.items():
            gain = self.leading_gain if leg_key == self.leading_leg else 1.0
            self._connect("BS_in_e", f"rg_e_{leg_key}", weight=4.0 * gain, p=0.45)
            self._connect("BS_in_f", f"rg_f_{leg_key}", weight=4.0 * gain, p=0.45)

    def _build_base_connections(self):
        """
        Tonic base drive (`cpg_2legs_fast` `BASE_DRIVE_*`): the shared base_in
        population projects bilaterally onto RGE and RGF with weight 1.0 and
        p 0.10, delay 1.0 ms.
        """
        if self.base is None:
            return
        for leg_key in self.legs:
            self._connect("BASE_in", f"rg_e_{leg_key}", weight=1.0, p=0.10, delay=1.0)
            self._connect("BASE_in", f"rg_f_{leg_key}", weight=1.0, p=0.10, delay=1.0)

    def _build_commissural_connections(self, p_comm: float = 0.40):
        """
        Commissural V-cell relay (NEURON `memcpg/leg.py` +
        `utils_cpg.py::add_external_connections`). Replaces the direct RG<->RG
        commissural inhibition that `cpg_2legs_fast` abstracts as
        "V0v/V0d/V2a/In1 classes":

          RG_F -> V2a -> V0v -> (contralateral) In1 -| RG_F
          RG_F -> V0d -| (contralateral) RG_F
          RG_F -> V3F -> (contralateral) RG_F

        V3F cross weight: memcpg's WALK mode leaves it at 0.0 and RUN uses 0.5;
        the scheme draws the projection, so the enabled reference value seeds it
        and PERSONA can tune it down to zero.
        """
        if "L" not in self.legs or "R" not in self.legs:
            return
        d = 3.0  # memcpg commissural delay
        self._connect("v0v_L", "in1_R", weight=1.3, p=p_comm, delay=d)
        self._connect("v0v_R", "in1_L", weight=1.3, p=p_comm, delay=d)
        self._connect("v0d_L", "rg_f_R", weight=-1.3, p=p_comm, delay=d)
        self._connect("v0d_R", "rg_f_L", weight=-1.3, p=p_comm, delay=d)
        self._connect("v3f_L", "rg_f_R", weight=0.5, p=p_comm, delay=d)
        self._connect("v3f_R", "rg_f_L", weight=0.5, p=p_comm, delay=d)

    def update(self, cpg_in: CPGInput):
        """
        Routes live sensory inputs from MuJoCo (CUT, Ia, Ib) to the CPG generators.
        """
        if "L" in self.legs:
            self.legs["L"].update_sensory_feedback(
                cut_hz=cpg_in.cut_L,
                ia_e_hz=cpg_in.ia_rate_E_L,
                ia_f_hz=cpg_in.ia_rate_F_L,
                ib_e_hz=cpg_in.ib_rate_E_L,
                ib_f_hz=cpg_in.ib_rate_F_L,
                current_time_ms=self.current_time_ms,
                cut_sites=cpg_in.cut_sites_L,
                cut_active=cpg_in.cut_active_L,
            )
        if "R" in self.legs:
            self.legs["R"].update_sensory_feedback(
                cut_hz=cpg_in.cut_R,
                ia_e_hz=cpg_in.ia_rate_E_R,
                ia_f_hz=cpg_in.ia_rate_F_R,
                ib_e_hz=cpg_in.ib_rate_E_R,
                ib_f_hz=cpg_in.ib_rate_F_R,
                current_time_ms=self.current_time_ms,
                cut_sites=cpg_in.cut_sites_R,
                cut_active=cpg_in.cut_active_R,
            )

    def simulate(self) -> None:
        """Advance the shared NEST kernel by one control step."""
        nest.Simulate(self.dt_ms)

    def readout(self) -> CPGOutput:
        """Drain this replica's recorders after the shared ``nest.Simulate``."""
        t_before = self.current_time_ms
        self.current_time_ms += self.dt_ms

        d_e_L, d_f_L = 0.0, 0.0
        d_e_R, d_f_R = 0.0, 0.0
        plastic: Dict[str, Dict[str, float]] = {"delta": {}, "n": {}}
        rg: Dict[str, float] = {}

        if "L" in self.legs:
            plastic["delta"].update(self.legs["L"].update_plastic_weights(t_before, self.current_time_ms))
            plastic["n"].update(self.legs["L"].plastic_counts)
            d_e_L, d_f_L = self.legs["L"].get_muscle_drives(self.dt_ms, self.current_time_ms)
            rg_e, rg_f = self.legs["L"].get_rg_rates(self.dt_ms, self.current_time_ms)
            rg.update({"rg_e_L": rg_e, "rg_f_L": rg_f})
        if "R" in self.legs:
            plastic["delta"].update(self.legs["R"].update_plastic_weights(t_before, self.current_time_ms))
            plastic["n"].update(self.legs["R"].plastic_counts)
            d_e_R, d_f_R = self.legs["R"].get_muscle_drives(self.dt_ms, self.current_time_ms)
            rg_e, rg_f = self.legs["R"].get_rg_rates(self.dt_ms, self.current_time_ms)
            rg.update({"rg_e_R": rg_e, "rg_f_R": rg_f})

        return CPGOutput(
            mus_E_L=d_e_L,
            mus_F_L=d_f_L,
            mus_E_R=d_e_R,
            mus_F_R=d_f_R,
            t_ms=self.current_time_ms,
            extra={"plastic": plastic, "rg": rg},
        )

    def step(self) -> CPGOutput:
        """
        Simulates NEST for dt_ms and reads out motor activations.
        """
        self.simulate()
        return self.readout()

    def build_cpg(self):
        pass

    def get_logs(self) -> Dict[str, Any]:
        return {
            "time_ms": self.current_time_ms,
            "legs": list(self.legs.keys())
        }


class MultiCPG:
    """N independent ``L2CPG`` replicas sharing one NEST kernel.

    One ``nest.Simulate(dt)`` advances every replica; each replica's recorders
    are drained afterwards. This is the only place that may call
    ``simulate()`` in a multi-replica run.
    """

    def __init__(self, cpgs):
        self.cpgs = list(cpgs)
        if not self.cpgs:
            raise ValueError("MultiCPG needs at least one replica")
        self.dt_ms = float(self.cpgs[0].dt_ms)

    @property
    def current_time_ms(self) -> float:
        return float(self.cpgs[0].current_time_ms)

    def update(self, cpg_inputs) -> None:
        inputs = list(cpg_inputs)
        if len(inputs) != len(self.cpgs):
            raise ValueError("one CPGInput per replica is required")
        for cpg, cpg_in in zip(self.cpgs, inputs):
            cpg.update(cpg_in)

    def step(self):
        self.cpgs[0].simulate()
        return [cpg.readout() for cpg in self.cpgs]

    def get_logs(self) -> Dict[str, Any]:
        return {
            "time_ms": self.current_time_ms,
            "replicas": len(self.cpgs),
        }

    def build_cpg(self):
        pass
