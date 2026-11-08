from model.adapter import CPGAdapter, CPGInput, CPGOutput
from model.leg import Leg
from model.params import PARAMS


class Model:
    """
    Closed-loop integration model coupling the biomechanical hindlimb
    with the spiking CPG network via the CPGAdapter interface.
    """
    def __init__(self,
                 cpg: CPGAdapter,
                 mujoco_cfg: str = "dual_rat_hindlimb.xml",
                 render_fps: int = PARAMS["render_fps"],
                 dt_ms: float = PARAMS["model_dt_ms"]):
        self.cpg = cpg
        self.dt_ms = dt_ms
        self.leg = Leg(mujoco_cfg=mujoco_cfg, render_fps=render_fps, dt_ms=dt_ms)

    def step(self,
             target_ankle_L: float | None = None,
             target_ankle_R: float | None = None) -> tuple[CPGInput, CPGOutput]:
        """
        Executes one closed-loop cycle:
        1. CPG generates motor neuron output for dt.
        2. Biomechanical model steps physics and calculates sensory afferents (CUT, Ia, Ib).
        3. CPG is updated with new sensory feedback rates.
        """
        cpg_out = self.cpg.step()
        cpg_in = self.leg.step(cpg_out, target_ankle_L=target_ankle_L,
                               target_ankle_R=target_ankle_R)
        self.cpg.update(cpg_in)
        return cpg_in, cpg_out

    def reset(self):
        self.leg.reset()
