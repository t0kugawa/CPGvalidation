import os
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio

from adapter import CPGAdapter, CPGOutput, CPGInput
from cut import CUTAfferent

class Leg: # Generates angles
    def __init__(self, mujoco_cfg: str = "dual_rat_hindlimb.xml", render_fps: int = 40, dt_ms: float = 2.0):
        self.time = 0.0
        self.frame = 0
        self.xml_path = mujoco_cfg
        self.render_fps = int(render_fps)
        self.dt_ms = dt_ms

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # --- Joints: model limb ---
        self.j_hip_m = self._jid("hip_m")
        self.j_knee_m = self._jid("knee_m")
        self.j_ankle_m = self._jid("ankle_m")
        self.j_toe_m = self._jid("toe_m_jnt")

        # --- Joints: reference limb ---
        self.j_hip_ref = self._jid("hip_ref")
        self.j_knee_ref = self._jid("knee_ref")
        self.j_ankle_ref = self._jid("ankle_ref")
        self.j_toe_ref = self._jid("toe_ref_jnt")
        self.act_sol_m = self._aid("sol_m")
        self.act_ta_m = self._aid("ta_m")
        self.act_toe_m = self._aid("toe_m_ctrl")

        # reference limb: position-актуаторы
        self.act_hip_ref = self._aid("hip_ref_ctrl")
        self.act_knee_ref = self._aid("knee_ref_ctrl")
        self.act_ankle_ref = self._aid("ankle_ref_ctrl")
        self.act_toe_ref = self._aid("toe_ref_ctrl")

        # --- Sites ---
        self.site_contact_m = self._sid("paw_contact_m")

        # --- Рецепторы CUT (для быстрого доступа) ---
        self.cut_sites_m = [
            self._sid("rct_heel_0"), self._sid("rct_heel_1"), self._sid("rct_heel_2"),
            self._sid("rct_mid_0"), self._sid("rct_mid_1"),
            self._sid("rct_fore_0"), self._sid("rct_fore_1"),
            self._sid("rct_toe_0"),
        ]

        # --- CUT afferent ---
        self.cut = CUTAfferent(r_peak_hz=120.0)

        # --- Renderer ---
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)

    def _jid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise ValueError(f"Joint '{name}' not found")
        return i

    def _aid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"Actuator '{name}' not found")
        return i

    def _sid(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise ValueError(f"Site '{name}' not found")
        return i

    def contract_muscles(self, cpg_output: CPGOutput):
        sol_act = float(np.clip(cpg_output.mus_E_L, 0.0, 1.0))
        ta_act = float(np.clip(cpg_output.mus_F_L, 0.0, 1.0))

        self.data.ctrl[self.act_sol_m] = sol_act
        self.data.ctrl[self.act_ta_m] = ta_act

    def step(self, cpg_output: CPGOutput) -> CPGInput:
        self.contract_muscles(cpg_output)
        self.time += self.dt_ms

        self.contract_muscles(cpg_output)

        return # TODO: CUT/Ia/Ib Input



