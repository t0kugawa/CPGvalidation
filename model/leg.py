import os
import numpy as np
import mujoco

from model.adapter import CPGOutput, CPGInput
from model.cut import CUTAfferent, IaAfferent, IbAfferent
from model.model_cfg import CONTACT_Z_EPS, TREADMILL_PENETRATION_M

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
        self._belt_override = None
        self.belt_z = None

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

        # Treadmill + target-limb foot geoms (used to level the plane).
        self.treadmill_gid = self._gid("treadmill")
        self.ref_foot_geoms = {
            s: [self._gid(f"foot_ref_geom_{s}"), self._gid(f"toe_ref_geom_{s}")]
            for s in SIDES
        }

        # Target-limb mocap bases: the hip (pelvis_ref_*) is placed each frame at
        # the dataset's GT position, so the target limb reproduces the trial.
        # TODO(option-B): a future version may instead place an independent mocap
        # body at EVERY marker (IC/GT/KN/EA/MH) to reproduce the recorded points
        # exactly, rather than the articulated rigid-bone chain (approach A).
        self.ref_pelvis = {
            s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"pelvis_ref_{s}")
            for s in SIDES
        }
        self.ref_mocap = {s: int(self.model.body_mocapid[self.ref_pelvis[s]]) for s in SIDES}
        self.ref_base_y = {s: float(self.model.body_pos[self.ref_pelvis[s], 1]) for s in SIDES}
        # Controlled-limb mocap bases: the model hip also tracks the dataset GT so
        # both limbs live in the dataset frame (non-ankle joints already follow the
        # target trajectory per spec D9), keeping their feet at comparable heights.
        self.model_pelvis = {
            s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"pelvis_m_{s}")
            for s in SIDES
        }
        self.model_mocap = {s: int(self.model.body_mocapid[self.model_pelvis[s]]) for s in SIDES}
        self.model_base_y = {s: float(self.model.body_pos[self.model_pelvis[s], 1]) for s in SIDES}
        self.model_foot_geoms = {
            s: [gid for gid in range(self.model.ngeom)
                if int(self.model.geom_bodyid[gid]) in set(self.foot_bodies[s])]
            for s in SIDES
        }

        self.cut_afferent = {s: CUTAfferent(dt_ms=dt_ms, r_peak_hz=120.0) for s in SIDES}
        # Reference lengths are placeholders: `calibrate_spindles` overwrites
        # them with the tendon length at a zero ankle angle, so the stretch term
        # stays alive whatever scale the limb bones are built at.
        self.ia_sol = {s: IaAfferent(l_0=0.0142, k_l=2000.0, k_v=120.0,
                                     k_force=6.0, r_base_hz=10.0, r_max_hz=250.0)
                       for s in SIDES}
        self.ia_ta = {s: IaAfferent(l_0=0.0072, k_l=2000.0, k_v=120.0,
                                    k_force=6.0, r_base_hz=10.0, r_max_hz=250.0)
                      for s in SIDES}
        self.ib_sol = {s: IbAfferent(k_force=15.0, k_act=40.0, r_base_hz=2.0, r_max_hz=200.0) for s in SIDES}
        self.ib_ta = {s: IbAfferent(k_force=15.0, k_act=40.0, r_base_hz=2.0, r_max_hz=200.0) for s in SIDES}

        self.calibrate_spindles()
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
        """Total normal contact force (N) on the MODEL foot/toe bodies of one leg."""
        total = 0.0
        bodies = set(self.foot_bodies[side])
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if b1 in bodies or b2 in bodies:
                total += self._contact_force_i(i)
        return total

    def _contact_force_i(self, i: int) -> float:
        f = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, i, f)
        return abs(float(f[0]))

    def _foot_lowest(self, geoms) -> tuple[float, float]:
        """
        Lowest surface point (x, z) in world coords (m) of the given foot/toe
        geoms. For capsules the true lowest point is the lower `fromto` endpoint
        minus the radius, not the geom centre minus the radius (which
        overestimates the height of a tilted capsule by up to its half length).
        """
        lo = np.inf
        lo_x = 0.0
        for gid in geoms:
            r = float(self.model.geom_size[gid, 0])
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]
                hl = float(self.model.geom_size[gid, 1])
                s = 1.0 if float(axis[2]) >= 0.0 else -1.0
                x = float(self.data.geom_xpos[gid, 0]) + s * hl * float(axis[0])
                z = float(self.data.geom_xpos[gid, 2]) - hl * abs(float(axis[2]))
            else:
                x = float(self.data.geom_xpos[gid, 0])
                z = float(self.data.geom_xpos[gid, 2])
            if z - r < lo:
                lo = z - r
                lo_x = x
        if not np.isfinite(lo):
            return 0.0, 0.0
        return float(lo_x), float(lo)

    def _foot_lowest_z(self, geoms) -> float:
        return self._foot_lowest(geoms)[1]

    def reset(self, pose=None, hip_pos=None):
        """
        Resets the simulation. If `pose` (dict with hip/knee/ankle per side) is
        given, the controlled limbs start from that pose so they match the
        target limbs; otherwise the default stance pose is used. `hip_pos`
        (dict {side: (x_m, z_m)}) places the target limbs' mocap bases at the
        dataset GT for the initial frame.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.time_ms = 0.0
        if pose is None:
            pose = {s: {"hip": -0.2, "knee": 0.6, "ankle": 0.3} for s in SIDES}
        for s in SIDES:
            self.cut_afferent[s].reset()
            p = pose[s]
            for jid, val in (
                (self.j_hip_m[s], p["hip"]),
                (self.j_knee_m[s], p["knee"]),
                (self.j_ankle_m[s], p["ankle"]),
                (self.j_hip_ref[s], p["hip"]),
                (self.j_knee_ref[s], p["knee"]),
                (self.j_ankle_ref[s], p["ankle"]),
            ):
                self.data.qpos[self.model.jnt_qposadr[jid]] = float(val)
        if hip_pos is not None:
            for s in SIDES:
                if s in hip_pos:
                    self.set_target_hip_position(s, float(hip_pos[s][0]), float(hip_pos[s][1]))

        mujoco.mj_forward(self.model, self.data)
        # `_level_treadmill` calls mj_setConst, which resets data (incl. qpos) to
        # the reference configuration, so preserve the pose across it.
        qpos_keep = self.data.qpos.copy()
        mocap_keep = self.data.mocap_pos.copy()
        self._level_treadmill()
        self.data.qpos[:] = qpos_keep
        self.data.mocap_pos[:] = mocap_keep
        # Re-apply after moving the plane so contacts start settled.
        mujoco.mj_forward(self.model, self.data)

    def set_target_hip_position(self, side: str, x_m: float, z_m: float):
        """
        Places both limbs' hip bases at (x, z) in world (each keeps its own
        lateral offset): the target limb's mocap base reproduces the dataset GT,
        and the controlled limb's base follows it so the two limbs share a frame.
        """
        for mocap_id, base_y in ((self.ref_mocap[side], self.ref_base_y[side]),
                                 (self.model_mocap[side], self.model_base_y[side])):
            if mocap_id < 0:
                continue
            self.data.mocap_pos[mocap_id, 0] = float(x_m)
            self.data.mocap_pos[mocap_id, 1] = base_y
            self.data.mocap_pos[mocap_id, 2] = float(z_m)

    def set_treadmill_height(self, z_m: float):
        """Places the belt surface at an explicit world height (m)."""
        self._belt_override = float(z_m)
        self._apply_belt(self._belt_override)

    def _apply_belt(self, z_m: float):
        self.belt_z = float(z_m)
        self.model.geom_pos[self.treadmill_gid, 2] = self.belt_z
        mujoco.mj_setConst(self.model, self.data)

    def _level_treadmill(self):
        """
        Places the treadmill plane so the limbs' feet rest on the belt: at the
        lowest surface point of the CONTROLLED limbs' feet plus a small
        penetration (`TREADMILL_PENETRATION_M`), so a resting contact (and thus a
        stance phase) is actually generated. A positive gap would suppress contact
        entirely. Returns the plane height used. If an explicit belt height was set
        with `set_treadmill_height`, that value is used instead.

        The controlled foot is the one that must touch: it is the only limb with
        collision geoms (the target limb is a non-colliding kinematic mirror) and
        the only one whose CUT contact feeds the CPG. The controlled foot's lowest
        surface also sits slightly above the target foot's over the cycle, so
        levelling on the target foot leaves the controlled foot in permanent air.
        """
        if self._belt_override is not None:
            self._apply_belt(self._belt_override)
            return self.belt_z
        lo = np.inf
        for s in SIDES:
            lo = min(lo, self._foot_lowest_z(self.model_foot_geoms[s]))
        if not np.isfinite(lo):
            return float(self.model.geom_pos[self.treadmill_gid, 2])
        self._apply_belt(lo + TREADMILL_PENETRATION_M)
        return self.belt_z

    def target_foot_lowest_percentile(self, hip_x, hip_z, hip, knee, ankle,
                                      percentile: float = 5.0) -> float:
        """
        Sweeps the target limb through the trial's hip/joint trajectory using
        forward kinematics only (no dynamics, no contact) and returns the given
        percentile of its foot's lowest surface height (m). This is the
        model-derived alternative to placing the belt at the dataset's ground.
        `hip_x`/`hip_z`/`hip`/`knee`/`ankle` map side -> per-frame array.
        """
        lows = []
        for s in SIDES:
            hip_adr = self.model.jnt_qposadr[self.j_hip_ref[s]]
            knee_adr = self.model.jnt_qposadr[self.j_knee_ref[s]]
            ankle_adr = self.model.jnt_qposadr[self.j_ankle_ref[s]]
            for i in range(len(np.asarray(ankle[s]))):
                self.data.qpos[hip_adr] = float(hip[s][i])
                self.data.qpos[knee_adr] = float(knee[s][i])
                self.data.qpos[ankle_adr] = float(ankle[s][i])
                self.set_target_hip_position(s, float(hip_x[s][i]), float(hip_z[s][i]))
                mujoco.mj_kinematics(self.model, self.data)
                lows.append(self._foot_lowest_z(self.ref_foot_geoms[s]))
        if not lows:
            return float(self.model.geom_pos[self.treadmill_gid, 2])
        return float(np.percentile(np.asarray(lows), percentile))

    def set_limb_lengths(self, prefix: str, side: str, thigh: float,
                         shank: float, foot: float):
        """
        Rescales one rig limb (`prefix` "m" controlled / "ref" target) to the
        given sagittal segment lengths (m) while preserving the rest directions,
        so the output limb matches the target's scale (approach A).
        """
        kids = ((f"shank_{prefix}_{side}", thigh), (f"foot_{prefix}_{side}", shank),
                (f"toe_{prefix}_{side}", foot))
        ratios = {}
        for name, target in kids:
            bid = self._bodyid(name)
            off = self.model.body_pos[bid].copy()
            cur = float(np.linalg.norm(off))
            ratios[name] = float(target) / cur if cur > 0.0 else 1.0
            self.model.body_pos[bid] = off * ratios[name]
        ratios[f"thigh_{prefix}_{side}"] = ratios[kids[0][0]]
        ratios[f"shank_{prefix}_{side}"] = ratios[kids[1][0]]
        ratios[f"foot_{prefix}_{side}"] = ratios[kids[2][0]]
        ratios[f"toe_{prefix}_{side}"] = ratios[kids[2][0]]
        for name, ratio in ratios.items():
            bid = self._bodyid(name)
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) == bid:
                    self.model.geom_pos[gid] *= ratio
                    self.model.geom_size[gid] *= ratio
            for sid in range(self.model.nsite):
                if int(self.model.site_bodyid[sid]) == bid:
                    self.model.site_pos[sid] *= ratio
        mujoco.mj_setConst(self.model, self.data)
        self.calibrate_spindles()

    def calibrate_spindles(self):
        """
        Sets each spindle's reference length to its tendon length at a zero ankle
        angle. The tendon sites move with the bones, so this must run whenever
        `set_limb_lengths` rescales a limb; otherwise the stretch term is
        measured against a length the tendon can never reach (and reads zero).
        """
        for s in SIDES:
            adr = int(self.model.jnt_qposadr[self.j_ankle_m[s]])
            keep = float(self.data.qpos[adr])
            self.data.qpos[adr] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.ia_sol[s].set_reference_length(
                float(self.data.ten_length[self.ten_sol[s]]))
            self.ia_ta[s].set_reference_length(
                float(self.data.ten_length[self.ten_ta[s]]))
            self.data.qpos[adr] = keep
        mujoco.mj_forward(self.model, self.data)

    def _bodyid(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"Body '{name}' not found")
        return i

    def get_reference_kinematics(self, t_s: float, phase_offset: float = 0.0) -> tuple[float, float, float]:
        """
        Bio-plausible periodic reference joint waveforms (hip, knee, ankle) for
        walking at ~1.5 Hz. `phase_offset` shifts the cycle for the right leg so
        the two reference limbs are not identical.
        """
        freq = 1.5
        phase = 2.0 * np.pi * freq * t_s + phase_offset
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
             target_knee_R: float | None = None,
             target_hipx_L: float | None = None,
             target_hipz_L: float | None = None,
             target_hipx_R: float | None = None,
             target_hipz_R: float | None = None) -> CPGInput:
        """
        Applies muscle drives, servos non-ankle joints to the target trajectory,
        steps physics for dt_ms, and populates CPGInput for both controlled legs.
        Hip/knee targets default to the procedural reference when not supplied.
        When `target_hipx_*`/`target_hipz_*` are supplied, the target limb's mocap
        base tracks the dataset GT position (approach A).
        """
        self.contract_muscles(cpg_output)

        t_s = self.time_ms / 1000.0
        # Distinct left/right reference trajectories (R is phase-shifted).
        ref = {s: self.get_reference_kinematics(t_s, 0.0 if s == "L" else np.pi)
               for s in SIDES}
        target_hip = {"L": target_hip_L, "R": target_hip_R}
        target_knee = {"L": target_knee_L, "R": target_knee_R}
        target_ankle = {"L": target_ankle_L, "R": target_ankle_R}
        target_hipx = {"L": target_hipx_L, "R": target_hipx_R}
        target_hipz = {"L": target_hipz_L, "R": target_hipz_R}

        for s in SIDES:
            hip_ref, knee_ref, ankle_ref = ref[s]
            hip_s = hip_ref if target_hip[s] is None else float(target_hip[s])
            knee_s = knee_ref if target_knee[s] is None else float(target_knee[s])
            ankle_s = ankle_ref if target_ankle[s] is None else float(target_ankle[s])

            # Target limb base tracks the dataset GT (approach A).
            if target_hipx[s] is not None and target_hipz[s] is not None:
                self.set_target_hip_position(s, target_hipx[s], target_hipz[s])

            # Controlled limb: non-ankle joints servoed to the target limb's pose.
            self.data.ctrl[self.act_hip_m[s]] = hip_s
            self.data.ctrl[self.act_knee_m[s]] = knee_s

            # Target limb: all three joints follow the reference trajectory.
            self.data.ctrl[self.act_hip_ref[s]] = hip_s
            self.data.ctrl[self.act_knee_ref[s]] = knee_s
            self.data.ctrl[self.act_ankle_ref[s]] = ankle_s
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

            foot_x_m, foot_z_m = self._foot_lowest(self.model_foot_geoms[s])
            foot_z_ref = self._foot_lowest_z(self.ref_foot_geoms[s])
            ref_contact = (self.belt_z is not None
                           and foot_z_ref <= self.belt_z + CONTACT_Z_EPS)
            phase_ref = "stance" if ref_contact else "swing"
            contact_ref = 1.0 if ref_contact else 0.0

            sol_len = float(self.data.ten_length[self.ten_sol[s]])
            sol_vel = float(self.data.ten_velocity[self.ten_sol[s]])
            ta_len = float(self.data.ten_length[self.ten_ta[s]])
            ta_vel = float(self.data.ten_velocity[self.ten_ta[s]])

            sol_force = float(abs(self.data.actuator_force[self.act_sol_m[s]]))
            ta_force = float(abs(self.data.actuator_force[self.act_ta_m[s]]))
            sol_act = self._activation(self.act_sol_m[s])
            ta_act = self._activation(self.act_ta_m[s])

            ia_rate_e = self.ia_sol[s].get_ia_hz(sol_len, sol_vel, sol_force)
            ia_rate_f = self.ia_ta[s].get_ia_hz(ta_len, ta_vel, ta_force)
            ib_rate_e = self.ib_sol[s].get_ib_hz(sol_force, sol_act)
            ib_rate_f = self.ib_ta[s].get_ib_hz(ta_force, ta_act)

            q_ankle_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_ankle_m[s]]])
            q_ankle_ref = float(self.data.qpos[self.model.jnt_qposadr[self.j_ankle_ref[s]]])
            q_knee_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_knee_m[s]]])
            q_hip_m = float(self.data.qpos[self.model.jnt_qposadr[self.j_hip_m[s]]])
            q_knee_ref = float(self.data.qpos[self.model.jnt_qposadr[self.j_knee_ref[s]]])
            q_hip_ref = float(self.data.qpos[self.model.jnt_qposadr[self.j_hip_ref[s]]])

            sensory[s] = dict(
                cut=cut_hz, phase=phase, phase_ref=phase_ref,
                contact=total_contact_force, contact_ref=contact_ref,
                foot_x_m=foot_x_m, foot_z_m=foot_z_m, foot_z_ref=foot_z_ref,
                sol_len=sol_len, sol_vel=sol_vel, ta_len=ta_len, ta_vel=ta_vel,
                ia_rate_e=ia_rate_e, ia_rate_f=ia_rate_f,
                sol_force=sol_force, ta_force=ta_force,
                sol_act=sol_act, ta_act=ta_act,
                ib_rate_e=ib_rate_e, ib_rate_f=ib_rate_f,
                q_ankle_m=q_ankle_m, q_ankle_ref=q_ankle_ref,
                q_knee_m=q_knee_m, q_hip_m=q_hip_m,
                q_knee_ref=q_knee_ref, q_hip_ref=q_hip_ref,
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
            phase_ref_L=L["phase_ref"], phase_ref_R=R["phase_ref"],
            contact_m_L=L["contact"], contact_m_R=R["contact"],
            contact_ref_L=L["contact_ref"], contact_ref_R=R["contact_ref"],
            foot_x_m_L=L["foot_x_m"], foot_x_m_R=R["foot_x_m"],
            foot_z_m_L=L["foot_z_m"], foot_z_m_R=R["foot_z_m"],
            foot_z_ref_L=L["foot_z_ref"], foot_z_ref_R=R["foot_z_ref"],
            ankle_angle_m_L=L["q_ankle_m"], ankle_angle_ref_L=L["q_ankle_ref"],
            knee_angle_m_L=L["q_knee_m"], hip_angle_m_L=L["q_hip_m"],
            ankle_angle_m_R=R["q_ankle_m"], ankle_angle_ref_R=R["q_ankle_ref"],
            knee_angle_m_R=R["q_knee_m"], hip_angle_m_R=R["q_hip_m"],
            knee_angle_ref_L=L["q_knee_ref"], hip_angle_ref_L=L["q_hip_ref"],
            knee_angle_ref_R=R["q_knee_ref"], hip_angle_ref_R=R["q_hip_ref"],
        )