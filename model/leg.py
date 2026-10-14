import os
import numpy as np
import mujoco

from model.adapter import CPGOutput, CPGInput
from model.cut import CUTAfferent, IaAfferent, IbAfferent


class Leg:
    """
    Rat hindlimb biomechanical simulation in MuJoCo.
    Simulates:
    - Model limb actuated by soleus (extensor) and tibialis anterior (flexor) muscles.
    - Reference limb driven by kinematic trajectory servos for comparative validation.
    Extracts biological sensory feedback:
    - CUT afferent (plantar mechanoreceptors)
    - Ia afferent (muscle spindle length and stretch-velocity feedback)
    - Ib afferent (Golgi tendon organ force/tension feedback)
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

        # Joint IDs - Model Limb
        self.j_hip_m = self._jid("hip_m")
        self.j_knee_m = self._jid("knee_m")
        self.j_ankle_m = self._jid("ankle_m")
        self.j_toe_m = self._jid("toe_m_jnt")

        # Joint IDs - Reference Limb
        self.j_hip_ref = self._jid("hip_ref")
        self.j_knee_ref = self._jid("knee_ref")
        self.j_ankle_ref = self._jid("ankle_ref")
        self.j_toe_ref = self._jid("toe_ref_jnt")

        # Actuator IDs - Model Limb Muscles
        self.act_sol_m = self._aid("sol_m")
        self.act_ta_m = self._aid("ta_m")
        self.act_toe_m = self._aid("toe_m_ctrl")

        # Actuator IDs - Reference Limb Servos
        self.act_hip_ref = self._aid("hip_ref_ctrl")
        self.act_knee_ref = self._aid("knee_ref_ctrl")
        self.act_ankle_ref = self._aid("ankle_ref_ctrl")
        self.act_toe_ref = self._aid("toe_ref_ctrl")

        # Tendon IDs
        self.ten_sol = self._tid("sol_tendon_m")
        self.ten_ta = self._tid("ta_tendon_m")

        # Plantar contact sensors (CUT)
        self.cut_sensor_names = [
            "cut_heel_0", "cut_heel_1", "cut_heel_2",
            "cut_mid_0", "cut_mid_1",
            "cut_fore_0", "cut_fore_1",
            "cut_toe_0"
        ]
        self.cut_sensor_ids = [self._sensid(name) for name in self.cut_sensor_names]

        # Sensory afferent models
        self.cut_afferent = CUTAfferent(dt_ms=dt_ms, r_peak_hz=120.0)
        self.ia_sol = IaAfferent(l_0=0.040, k_l=600.0, k_v=120.0, r_base_hz=10.0, r_max_hz=250.0)
        self.ia_ta = IaAfferent(l_0=0.028, k_l=600.0, k_v=120.0, r_base_hz=10.0, r_max_hz=250.0)
        self.ib_sol = IbAfferent(k_force=15.0, k_act=40.0, r_base_hz=2.0, r_max_hz=200.0)
        self.ib_ta = IbAfferent(k_force=15.0, k_act=40.0, r_base_hz=2.0, r_max_hz=200.0)

        # Baseline resting angles
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

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.time_ms = 0.0
        self.cut_afferent.reset()

        # Set default joint angles: comfortable stance posture
        q_hip_m = self.model.jnt_qposadr[self.j_hip_m]
        q_knee_m = self.model.jnt_qposadr[self.j_knee_m]
        q_ankle_m = self.model.jnt_qposadr[self.j_ankle_m]

        q_hip_ref = self.model.jnt_qposadr[self.j_hip_ref]
        q_knee_ref = self.model.jnt_qposadr[self.j_knee_ref]
        q_ankle_ref = self.model.jnt_qposadr[self.j_ankle_ref]

        self.data.qpos[q_hip_m] = -0.2
        self.data.qpos[q_knee_m] = 0.6
        self.data.qpos[q_ankle_m] = 0.3

        self.data.qpos[q_hip_ref] = -0.2
        self.data.qpos[q_knee_ref] = 0.6
        self.data.qpos[q_ankle_ref] = 0.3

        mujoco.mj_forward(self.model, self.data)

    def get_reference_kinematics(self, t_s: float) -> tuple[float, float, float]:
        """
        Calculates reference limb target angles (hip, knee, ankle) for walking at frequency ~1.5 Hz.
        Produces bio-plausible periodic flexion-extension waveforms.
        """
        freq = 1.5
        phase = 2.0 * np.pi * freq * t_s

        # Typical rat hindlimb joint excursions during locomotion:
        # Hip: swings back during stance (extension) and forwards during swing (flexion)
        hip_ref = -0.25 + 0.35 * np.cos(phase)
        # Knee: flexes during early swing and early stance
        knee_ref = 0.65 + 0.30 * np.sin(phase)
        # Ankle: plantarflexes during stance push-off (higher angle), dorsiflexes in swing (lower angle)
        ankle_ref = 0.40 + 0.45 * np.sin(phase - 0.5)

        return hip_ref, knee_ref, ankle_ref

    def contract_muscles(self, cpg_output: CPGOutput):
        sol_act = float(np.clip(cpg_output.mus_E_L, 0.0, 1.0))
        ta_act = float(np.clip(cpg_output.mus_F_L, 0.0, 1.0))

        self.data.ctrl[self.act_sol_m] = sol_act
        self.data.ctrl[self.act_ta_m] = ta_act
        self.data.ctrl[self.act_toe_m] = 0.0

    def step(self, cpg_output: CPGOutput, external_ref_ankle: float | None = None) -> CPGInput:
        """
        Applies muscle drives, drives reference limb, steps physics for dt_ms,
        and computes sensory afferents (CUT, Ia, Ib) for CPGInput.
        """
        # 1. Apply motor control to muscles
        self.contract_muscles(cpg_output)

        # 2. Drive reference limb kinematics
        t_s = self.time_ms / 1000.0
        hip_ref, knee_ref, ankle_ref = self.get_reference_kinematics(t_s)
        if external_ref_ankle is not None:
            ankle_ref = external_ref_ankle

        self.data.ctrl[self.act_hip_ref] = hip_ref
        self.data.ctrl[self.act_knee_ref] = knee_ref
        self.data.ctrl[self.act_ankle_ref] = ankle_ref
        self.data.ctrl[self.act_toe_ref] = 0.0

        # 3. Step physics simulation for dt_ms
        n_substeps = max(1, int(round((self.dt_ms / 1000.0) / self.physics_dt)))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.time_ms += self.dt_ms

        # 4. Extract Plantar Contact Forces & CUT rate
        total_contact_force = 0.0
        for sens_id in self.cut_sensor_ids:
            adr = self.model.sensor_adr[sens_id]
            total_contact_force += float(self.data.sensordata[adr])

        cut_hz = self.cut_afferent.get_cut_hz(total_contact_force)
        phase = "stance" if total_contact_force > 0.005 else "swing"

        # 5. Extract Tendon lengths and velocities for Ia afferents
        sol_len = float(self.data.ten_length[self.ten_sol])
        sol_vel = float(self.data.ten_velocity[self.ten_sol])
        ta_len = float(self.data.ten_length[self.ten_ta])
        ta_vel = float(self.data.ten_velocity[self.ten_ta])

        ia_rate_e = self.ia_sol.get_ia_hz(sol_len, sol_vel)
        ia_rate_f = self.ia_ta.get_ia_hz(ta_len, ta_vel)

        # 6. Extract Actuator forces and activations for Ib afferents
        sol_force = float(abs(self.data.actuator_force[self.act_sol_m]))
        ta_force = float(abs(self.data.actuator_force[self.act_ta_m]))
        sol_act = float(self.data.act[self.act_sol_m]) if len(self.data.act) > self.act_sol_m else float(cpg_output.mus_E_L)
        ta_act = float(self.data.act[self.act_ta_m]) if len(self.data.act) > self.act_ta_m else float(cpg_output.mus_F_L)

        ib_rate_e = self.ib_sol.get_ib_hz(sol_force, sol_act)
        ib_rate_f = self.ib_ta.get_ib_hz(ta_force, ta_act)

        # 7. Extract Joint angles
        q_ankle_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_ankle_m]])
        q_ankle_ref = float(self.data.qpos[self.model.jnt_qposadr[self.j_ankle_ref]])
        q_knee_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_knee_m]])
        q_hip_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_hip_m]])

        # Package into CPGInput
        cpg_input = CPGInput(
            cut_L=cut_hz,
            cut_R=cut_hz,
            ia_len_E_L=sol_len,
            ia_vel_E_L=sol_vel,
            ia_rate_E_L=ia_rate_e,
            ia_len_F_L=ta_len,
            ia_vel_F_L=ta_vel,
            ia_rate_F_L=ia_rate_f,
            ia_len_E_R=sol_len,
            ia_vel_E_R=sol_vel,
            ia_rate_E_R=ia_rate_e,
            ia_len_F_R=ta_len,
            ia_vel_F_R=ta_vel,
            ia_rate_F_R=ia_rate_f,
            ib_act_E_L=sol_act,
            ib_force_E_L=sol_force,
            ib_rate_E_L=ib_rate_e,
            ib_act_F_L=ta_act,
            ib_force_F_L=ta_force,
            ib_rate_F_L=ib_rate_f,
            ib_act_E_R=sol_act,
            ib_force_E_R=sol_force,
            ib_rate_E_R=ib_rate_e,
            ib_act_F_R=ta_act,
            ib_force_F_R=ta_force,
            ib_rate_F_R=ib_rate_f,
            phase_L=phase,
            phase_R="swing" if phase == "stance" else "stance",
            ankle_angle_m_L=q_ankle_m,
            ankle_angle_ref_L=q_ankle_ref,
            knee_angle_m_L=q_knee_m,
            hip_angle_m_L=q_hip_m
        )

        return cpg_input
