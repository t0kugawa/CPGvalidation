from adapter import CPGAdapter
from leg import Leg


class Model:
    def __init__(self,
                 cpg: CPGAdapter,
                 mujoco_cfg: str = "dual_rat_hindlimb.xml",
                 render_fps: int = 40):
        self.legs = {"L": Leg(mujoco_cfg, render_fps=render_fps),
                     "R": Leg(mujoco_cfg, render_fps=render_fps)}

        self.cpg = cpg


