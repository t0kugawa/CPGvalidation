import os
import numpy as np
import mujoco

from model.adapter import CPGOutput, CPGInput
from model.cut import CUTAfferent, IaAfferent, IbAfferent

SIDES = ("L", "R")


class Leg:
    """
    Bilateral rat hindlimb biomechanical simulation in MuJoCo.
    Simulates four limbs (controlled L/R actuated by SOL/TA muscles, target
    L/R driven by kinematic servos) and extracts biological sensory feedback
    for both controlled legs:
    - CUT afferent (plantar mechanoreceptors)
    - Ia afferent (muscle spindle length and stretch-velocity feedback)
    - Ib afferent (Golgi tendon organ force/tension feedback)
    Only the ankle is muscle-driven; every other controlled joint is servo-held
    to the target trajectory so the ankle is the free variable (MVP).
    """
    def __init__(self,
                 mujoco_cfg: str = "dual_rat_hindlimb.xml",
                 render_fps: int = 40,
                 dt_ms: float = 2.0):
        self.time_ms = 0.0
        self.frame = 0
        self.xml_path = mujoco_cfg
        self.render_fps = int(render_fps)
        self.dt_ms = dt_ms

        if not os.path.isabs(self.xml_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(base_dir, self.xml_path)
            if os.path.exists(candidate):
                self.xml_path = candidate

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.physics_dt = self.model.opt.timestep

        self.j_hip_m = {s: self._jid(f"hip_m_{s}") for s in SIDES}
        self.j_knee_m = {s: self._jid(f"knee_m_{s}") for s in SIDES}
        self.j_ankle_m = {s: self._jid(f"ankle_m_{s}") for s in SIDES}
        self.j_toe_m = {s: self._jid(f"toe_m_jnt_{s}") for s in SIDES}

        self.j_hip_ref = {s: self._jid(f"hip_ref_{s}") for s in SIDES}
        self.j_knee_ref = {s: self._jid(f"knee_ref_{s}") for s in SIDES}
        self.j_ankle_ref = {s: self._jid(f"ankle_ref_{s}") for s in SIDES}
        self.j_toe_ref = {s: self._jid(f"toe_ref_jnt_{s}") for s in SIDES}

        self.act_sol_m = {s: self._aid(f"sol_m_{s}") for s in SIDES}
        self.act_ta_m = {s: self._aid(f"ta_m_{s}") for s in SIDES}

        self.act_hip_m = {s: self._aid(f"hip_m_ctrl_{s}") for s in SIDES}
        self.act_knee_m = {s: self._aid(f"knee_m_ctrl_{s}") for s in SIDES}
        self.act_toe_m = {s: self._aid(f"toe_m_ctrl_{s}") for s in SIDES}

        self.act_hip_ref = {s: self._aid(f"hip_ref_ctrl_{s}") for s in SIDES}
        self.act_knee_ref = {s: self._aid(f"knee_ref_ctrl_{s}") for s in SIDES}
        self.act_ankle_ref = {s: self._aid(f"ankle_ref_ctrl_{s}") for s in SIDES}
        self.act_toe_ref = {s: self._aid(f"toe_ref_ctrl_{s}") for s in SIDES}

        self.ten_sol = {s: self._tid(f"sol_tendon_m_{s}") for s in SIDES}
        self.ten_ta = {s: self._tid(f"ta_tendon_m_{s}") for s in SIDES}

        # Contact bodies for the CUT model (foot + toe of each controlled leg).
        self.foot_bodies = {
            s: [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in (f"foot_m_{s}", f"toe_m_{s}")]
            for s in SIDES
        }

        self.cut_afferent = {s: CUTAfferent(dt_ms=dt_ms, r_peak_hz=120.0) for s in SIDES}
        self.ia_sol = {s: IaAfferent(l_0=0.040, k_l=600.0, k_v=120.0, r_base_hz=10.0, r_max_hz=250.0) for s in SIDES}
        self.ia_ta = {s: IaAfferent(l_0=0.028, k_l=600.0, k_v=120.0, r_base_hz=10.0, r_max_hz=250.0) for s in SIDES}
        self.ib_sol = {s: IbAfferent(k_force=15.0, k_act=40.0, r_base_hz=2.0, r_max_hz=200.0) for s in SIDES}
        self.ib_ta = {s: IbAfferent(k_force=15.0, k_act=40.0, r_base_hz=2.0, r_max_hz=200.0) for s in SIDES}

        self.reset()

    def _jid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise ValueError(f"Joint '{name}' not found")
        return i

    def _aid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise ValueError(f"Actuator '{name}' not found")
        return i

    def _tid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
        if i < 0:
            raise ValueError(f"Tendon '{name}' not found")
        return i

    def _sensid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if i < 0:
            raise ValueError(f"Sensor '{name}' not found")
        return i

    def _gid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise ValueError(f"Geom '{name}' not found")
        return i

    def set_treadmill_surfacevel(self, x_mps: float):
        """Sets the treadmill belt x-velocity (m/s) from the trial metadata."""
        gid = self._gid("treadmill")
        self.model.geom_surfacevel[gid, 0] = -float(x_mps)

    def _activation(self, actuator_id: int) -> float:
        adr = self.model.actuator_actadr[actuator_id]
        if adr >= 0 and adr < len(self.data.act):
            return float(self.data.act[adr])
        return 0.0

    def _contact_force(self, side: str) -> float:
        """Total normal contact force (N) on the foot/toe bodies of one leg."""
        total = 0.0
        bodies = set(self.foot_bodies[side])
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if b1 in bodies or b2 in bodies:
                f = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                total += abs(float(f[0]))
        return total

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.time_ms = 0.0
        for s in SIDES:
            self.cut_afferent[s].reset()
            q_hip_m = self.model.jnt_qposadr[self.j_hip_m[s]]
            q_knee_m = self.model.jnt_qposadr[self.j_knee_m[s]]
            q_ankle_m = self.model.jnt_qposadr[self.j_ankle_m[s]]
            q_hip_ref = self.model.jnt_qposadr[self.j_hip_ref[s]]
            q_knee_ref = self.model.jnt_qposadr[self.j_knee_ref[s]]
            q_ankle_ref = self.model.jnt_qposadr[self.j_ankle_ref[s]]

            self.data.qpos[q_hip_m] = -0.2
            self.data.qpos[q_knee_m] = 0.6
            self.data.qpos[q_ankle_m] = 0.3

            self.data.qpos[q_hip_ref] = -0.2
            self.data.qpos[q_knee_ref] = 0.6
            self.data.qpos[q_ankle_ref] = 0.3

        mujoco.mj_forward(self.model, self.data)

    def get_reference_kinematics(self, t_s: float) -> tuple[float, float, float]:
        """
        Bio-plausible periodic reference joint waveforms (hip, knee, ankle) for
        walking at ~1.5 Hz. Used when no external target trajectory is supplied.
        """
        freq = 1.5
        phase = 2.0 * np.pi * freq * t_s
        hip_ref = -0.25 + 0.35 * np.cos(phase)
        knee_ref = 0.65 + 0.30 * np.sin(phase)
        ankle_ref = 0.40 + 0.45 * np.sin(phase - 0.5)
        return hip_ref, knee_ref, ankle_ref

    def contract_muscles(self, cpg_output: CPGOutput):
        for s, e_key, f_key in (("L", "mus_E_L", "mus_F_L"), ("R", "mus_E_R", "mus_F_R")):
            self.data.ctrl[self.act_sol_m[s]] = float(np.clip(getattr(cpg_output, e_key), 0.0, 1.0))
            self.data.ctrl[self.act_ta_m[s]] = float(np.clip(getattr(cpg_output, f_key), 0.0, 1.0))
            self.data.ctrl[self.act_toe_m[s]] = 0.0

    def step(self, cpg_output: CPGOutput,
             target_ankle_L: float | None = None,
             target_ankle_R: float | None = None,
             target_hip_L: float | None = None,
             target_knee_L: float | None = None,
             target_hip_R: float | None = None,
             target_knee_R: float | None = None) -> CPGInput:
        """
        Applies muscle drives, servos non-ankle joints to the target trajectory,
        steps physics for dt_ms, and populates CPGInput for both controlled legs.
        Hip/knee targets default to the procedural reference when not supplied.
        """
        self.contract_muscles(cpg_output)

        t_s = self.time_ms / 1000.0
        hip_ref, knee_ref, ankle_ref = self.get_reference_kinematics(t_s)
        target_hip = {"L": target_hip_L, "R": target_hip_R}
        target_knee = {"L": target_knee_L, "R": target_knee_R}
        target_ankle = {"L": target_ankle_L, "R": target_ankle_R}

        for s in SIDES:
            hip_s = hip_ref if target_hip[s] is None else float(target_hip[s])
            knee_s = knee_ref if target_knee[s] is None else float(target_knee[s])
            self.data.ctrl[self.act_hip_m[s]] = hip_s
            self.data.ctrl[self.act_knee_m[s]] = knee_s

            self.data.ctrl[self.act_hip_ref[s]] = hip_s
            self.data.ctrl[self.act_knee_ref[s]] = knee_s
            self.data.ctrl[self.act_ankle_ref[s]] = (
                ankle_ref if target_ankle[s] is None else float(target_ankle[s])
            )
            self.data.ctrl[self.act_toe_ref[s]] = 0.0

        n_substeps = max(1, int(round((self.dt_ms / 1000.0) / self.physics_dt)))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.time_ms += self.dt_ms

        sensory = {}
        for s in SIDES:
            total_contact_force = self._contact_force(s)
            cut_hz = self.cut_afferent[s].get_cut_hz(total_contact_force)
            phase = "stance" if total_contact_force > 0.005 else "swing"

            sol_len = float(self.data.ten_length[self.ten_sol[s]])
            sol_vel = float(self.data.ten_velocity[self.ten_sol[s]])
            ta_len = float(self.data.ten_length[self.ten_ta[s]])
            ta_vel = float(self.data.ten_velocity[self.ten_ta[s]])
            ia_rate_e = self.ia_sol[s].get_ia_hz(sol_len, sol_vel)
            ia_rate_f = self.ia_ta[s].get_ia_hz(ta_len, ta_vel)

            sol_force = float(abs(self.data.actuator_force[self.act_sol_m[s]]))
            ta_force = float(abs(self.data.actuator_force[self.act_ta_m[s]]))
            sol_act = self._activation(self.act_sol_m[s])
            ta_act = self._activation(self.act_ta_m[s])
            ib_rate_e = self.ib_sol[s].get_ib_hz(sol_force, sol_act)
            ib_rate_f = self.ib_ta[s].get_ib_hz(ta_force, ta_act)

            q_ankle_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_ankle_m[s]]])
            q_ankle_ref = float(self.data.qpos[self.model.jnt_qposadr[self.j_ankle_ref[s]]])
            q_knee_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_knee_m[s]]])
            q_hip_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_hip_m[s]]])

            sensory[s] = dict(
                cut=cut_hz, phase=phase,
                sol_len=sol_len, sol_vel=sol_vel, ta_len=ta_len, ta_vel=ta_vel,
                ia_rate_e=ia_rate_e, ia_rate_f=ia_rate_f,
                sol_force=sol_force, ta_force=ta_force,
                sol_act=sol_act, ta_act=ta_act,
                ib_rate_e=ib_rate_e, ib_rate_f=ib_rate_f,
                q_ankle_m=q_ankle_m, q_ankle_ref=q_ankle_ref,
                q_knee_m=q_knee_m, q_hip_m=q_hip_m,
            )

        L, R = sensory["L"], sensory["R"]
        return CPGInput(
            cut_L=L["cut"], cut_R=R["cut"],
            ia_len_E_L=L["sol_len"], ia_vel_E_L=L["sol_vel"], ia_rate_E_L=L["ia_rate_e"],
            ia_len_F_L=L["ta_len"], ia_vel_F_L=L["ta_vel"], ia_rate_F_L=L["ia_rate_f"],
            ia_len_E_R=R["sol_len"], ia_vel_E_R=R["sol_vel"], ia_rate_E_R=R["ia_rate_e"],
            ia_len_F_R=R["ta_len"], ia_vel_F_R=R["ta_vel"], ia_rate_F_R=R["ia_rate_f"],
            ib_act_E_L=L["sol_act"], ib_force_E_L=L["sol_force"], ib_rate_E_L=L["ib_rate_e"],
            ib_act_F_L=L["ta_act"], ib_force_F_L=L["ta_force"], ib_rate_F_L=L["ib_rate_f"],
            ib_act_E_R=R["sol_act"], ib_force_E_R=R["sol_force"], ib_rate_E_R=R["ib_rate_e"],
            ib_act_F_R=R["ta_act"], ib_force_F_R=R["ta_force"], ib_rate_F_R=R["ib_rate_f"],
            phase_L=L["phase"], phase_R=R["phase"],
            ankle_angle_m_L=L["q_ankle_m"], ankle_angle_ref_L=L["q_ankle_ref"],
            knee_angle_m_L=L["q_knee_m"], hip_angle_m_L=L["q_hip_m"],
            ankle_angle_m_R=R["q_ankle_m"], ankle_angle_ref_R=R["q_ankle_ref"],
            knee_angle_m_R=R["q_knee_m"], hip_angle_m_R=R["q_hip_m"],
        )