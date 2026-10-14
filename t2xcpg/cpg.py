from typing import Dict, Any
import nest
import numpy as np

from persona.registry import Registry
from persona.connection import Connection
from persona.parameter import PersonaParam
from model.adapter import CPGAdapter, CPGInput, CPGOutput
from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg
from t2xcpg.utils import get_jitter_delay


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
                 p_comm: float = 0.40):
        self.registry = registry
        self.bs = bs
        self.legs = legs
        self.dt_ms = dt_ms
        self.current_time_ms = 0.0

        # Build Brainstem drives and Commissural Cross-Connections
        self._build_bs_connections()
        self._build_commissural_connections(p_comm=p_comm)

    def _param(self, name: str, mean: float, sigma_param: float = 0.05,
               sigma_ctrl: float = 0.1, min_val: float = -100.0, max_val: float = 100.0) -> PersonaParam:
        return PersonaParam(
            mean=mean,
            sigma_param=sigma_param,
            sigma_ctrl=sigma_ctrl,
            min=min_val,
            max=max_val
        )

    def _connect(self, src: str, tgt: str, weight: float, p: float = 0.40, delay: float = 3.0) -> Any:
        w_param = self._param("weight", weight, sigma_param=abs(weight) * 0.05,
                              sigma_ctrl=abs(weight) * 0.1,
                              min_val=-100.0 if weight < 0 else 0.0,
                              max_val=0.0 if weight < 0 else 100.0)
        p_param = self._param("p", p, sigma_param=0.02, sigma_ctrl=0.05, min_val=0.01, max_val=1.0)

        cfg = Connection(
            source=src,
            target=tgt,
            model="static_synapse",
            parameters={"weight": w_param},
            rule="pairwise_bernoulli",
            p=p_param,
            extra_syn_spec={"delay": float(delay)}
        )
        return self.registry.create_connection(cfg)

    def _build_bs_connections(self):
        """
        Connects Brainstem to Extensor (RGE) and Flexor (RGF) generators of all legs.
        """
        for leg_key, leg in self.legs.items():
            self._connect("BS_in_e", f"rg_e_{leg_key}", weight=18.0, p=0.45)
            self._connect("BS_in_f", f"rg_f_{leg_key}", weight=18.0, p=0.45)

    def _build_commissural_connections(self, p_comm: float = 0.40):
        """
        Implements bilateral commissural coordination between left and right limb:
        1. V0v (L/R) -> In1 (R/L) -> RGF (R/L) [contra-lateral inhibition of flexor]
        2. V0d (L/R) -> RGF (R/L) [direct contra-lateral inhibition]
        3. V3-F (L/R) -> RGF (R/L) [contra-lateral excitation of flexor]
        """
        if "L" in self.legs and "R" in self.legs:
            # L -> R
            self._connect("v0v_L", "in1_R", weight=18.0, p=p_comm, delay=3.0)
            self._connect("v0d_L", "rg_f_R", weight=-22.0, p=p_comm, delay=3.0)
            self._connect("v3_f_L", "rg_f_R", weight=16.0, p=p_comm, delay=3.0)

            # R -> L
            self._connect("v0v_R", "in1_L", weight=18.0, p=p_comm, delay=3.0)
            self._connect("v0d_R", "rg_f_L", weight=-22.0, p=p_comm, delay=3.0)
            self._connect("v3_f_R", "rg_f_L", weight=16.0, p=p_comm, delay=3.0)

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
                current_time_ms=self.current_time_ms
            )
        if "R" in self.legs:
            self.legs["R"].update_sensory_feedback(
                cut_hz=cpg_in.cut_R,
                ia_e_hz=cpg_in.ia_rate_E_R,
                ia_f_hz=cpg_in.ia_rate_F_R,
                ib_e_hz=cpg_in.ib_rate_E_R,
                ib_f_hz=cpg_in.ib_rate_F_R,
                current_time_ms=self.current_time_ms
            )

    def step(self) -> CPGOutput:
        """
        Simulates NEST for dt_ms and reads out motor activations.
        """
        nest.Simulate(self.dt_ms)
        self.current_time_ms += self.dt_ms

        d_e_L, d_f_L = 0.0, 0.0
        d_e_R, d_f_R = 0.0, 0.0

        if "L" in self.legs:
            d_e_L, d_f_L = self.legs["L"].get_muscle_drives(self.dt_ms)
        if "R" in self.legs:
            d_e_R, d_f_R = self.legs["R"].get_muscle_drives(self.dt_ms)

        out = CPGOutput(
            mus_E_L=d_e_L,
            mus_F_L=d_f_L,
            mus_E_R=d_e_R,
            mus_F_R=d_f_R,
            t_ms=self.current_time_ms
        )
        return out

    def build_cpg(self):
        pass

    def get_logs(self) -> Dict[str, Any]:
        return {
            "time_ms": self.current_time_ms,
            "legs": list(self.legs.keys())
        }
