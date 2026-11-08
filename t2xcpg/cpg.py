from typing import Dict, Any
import os
import nest

from persona.registry import Registry
from persona.connection import Connection
from persona.parameter import PersonaParam
from model.adapter import CPGAdapter, CPGInput, CPGOutput
from t2xcpg.bs import BrainStem, BaseDrive
from t2xcpg.leg import Leg
from t2xcpg.params import PARAMS

# Drive-balance spec (2026-10-20): the chosen BS->RG base weight before the
# leading-leg gain. Both legs use the same base; only leading_gain differs.
W_BS2RG = PARAMS["w_con_bs_rg"]
SCALE_BS = float(os.environ.get("CPG_SCALE_BS", str(PARAMS["scale_bs"])))
SCALE_BASE = float(os.environ.get("CPG_SCALE_BASE", str(PARAMS["scale_base"])))


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
                 dt_ms: float = PARAMS["dt_ms"],
                 p_comm: float = PARAMS["p_comm"],
                 leading_leg: str = PARAMS["leading_leg"],
                 leading_gain: float = PARAMS["leading_gain"],
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

    def set_bs_speed(self, speed_mps: float, ref_e: float | None = None,
                     ref_f: float | None = None):
        """Speed-scale the preserved BS drive without removing it."""
        self.bs.set_speed(speed_mps, ref_e=ref_e, ref_f=ref_f)

    def _param(self, name: str, mean: float,
               sigma: float = PARAMS["l2_syn_weight_sigma_fraction"],
               min_val: float = PARAMS["l2_syn_weight_min"],
               max_val: float = PARAMS["l2_syn_weight_max"]) -> PersonaParam:
        return PersonaParam(
            mean=mean,
            sigma=sigma,
            min=min_val,
            max=max_val
        )

    def _connect(self, src: str, tgt: str, weight: float,
                 p: float = PARAMS["l2_p_default"],
                 delay: float = PARAMS["l2_delay_default"]) -> Any:
        w_param = self._param("weight", weight,
                              abs(weight) * PARAMS["l2_syn_weight_sigma_fraction"],
                              min_val=PARAMS["l2_syn_weight_min"] if weight < 0 else 0.0,
                              max_val=0.0 if weight < 0 else PARAMS["l2_syn_weight_max"])
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
            gain = (self.leading_gain if leg_key == self.leading_leg
                    else PARAMS["leading_gain_other"])
            self._connect("BS_in_e", f"rg_e_{leg_key}",
                          weight=W_BS2RG * gain * SCALE_BS,
                          p=PARAMS["p_con_bs_rg"])
            self._connect("BS_in_f", f"rg_f_{leg_key}",
                          weight=W_BS2RG * gain * SCALE_BS,
                          p=PARAMS["p_con_bs_rg"])

    def _build_base_connections(self):
        """
        Tonic base drive (`cpg_2legs_fast` `BASE_DRIVE_*`): the shared base_in
        population projects bilaterally onto RGE and RGF with weight 1.0 and
        p 0.10, delay 1.0 ms.
        """
        if self.base is None:
            return
        for leg_key in self.legs:
            self._connect("BASE_in", f"rg_e_{leg_key}",
                          weight=PARAMS["w_con_base_rg"] * SCALE_BASE,
                          p=PARAMS["p_con_base_rg"],
                          delay=PARAMS["delay_base_rg"])
            self._connect("BASE_in", f"rg_f_{leg_key}",
                          weight=PARAMS["w_con_base_rg"] * SCALE_BASE,
                          p=PARAMS["p_con_base_rg"],
                          delay=PARAMS["delay_base_rg"])

    def _build_commissural_connections(self, p_comm: float = PARAMS["p_comm"]):
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
        d = PARAMS["delay_comm"]  # memcpg commissural delay
        self._connect("v0v_L", "in1_R", weight=PARAMS["w_con_v0v_in1"], p=p_comm, delay=d)
        self._connect("v0v_R", "in1_L", weight=PARAMS["w_con_v0v_in1"], p=p_comm, delay=d)
        self._connect("v0d_L", "rg_f_R", weight=PARAMS["w_con_v0d_rg_f"], p=p_comm, delay=d)
        self._connect("v0d_R", "rg_f_L", weight=PARAMS["w_con_v0d_rg_f"], p=p_comm, delay=d)
        self._connect("v3f_L", "rg_f_R", weight=PARAMS["w_con_v3f_rg_f"], p=p_comm, delay=d)
        self._connect("v3f_R", "rg_f_L", weight=PARAMS["w_con_v3f_rg_f"], p=p_comm, delay=d)

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


