from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from persona.registry import Registry
from persona.agent import Agent
from model.adapter import CPGInput


class MocapTargetDriver:
    """
    Serves the target-limb joint trajectory to the rig. `pose()` returns the
    Stage-1 initial pose; `target(t_ms)` returns the Stage-2 trajectory value
    (interpolated at the simulation time). Ankle is always tracked; hip/knee
    are optional (default to the rig's procedural reference when omitted).
    """
    def __init__(self, ankle_L, ankle_R, dt_ms: float = 10.0, pose=None,
                 hip_L=None, knee_L=None, hip_R=None, knee_R=None,
                 hip_x_L=None, hip_z_L=None, hip_x_R=None, hip_z_R=None,
                 loop: bool = True):
        self.ankle_L = np.asarray(ankle_L, dtype=np.float64)
        self.ankle_R = np.asarray(ankle_R, dtype=np.float64)
        self.dt_ms = float(dt_ms)
        self.loop = bool(loop)
        self.period_ms = float(len(self.ankle_L)) * self.dt_ms
        self.hip_L = None if hip_L is None else np.asarray(hip_L, dtype=np.float64)
        self.knee_L = None if knee_L is None else np.asarray(knee_L, dtype=np.float64)
        self.hip_R = None if hip_R is None else np.asarray(hip_R, dtype=np.float64)
        self.knee_R = None if knee_R is None else np.asarray(knee_R, dtype=np.float64)
        self.hip_x_L = None if hip_x_L is None else np.asarray(hip_x_L, dtype=np.float64)
        self.hip_z_L = None if hip_z_L is None else np.asarray(hip_z_L, dtype=np.float64)
        self.hip_x_R = None if hip_x_R is None else np.asarray(hip_x_R, dtype=np.float64)
        self.hip_z_R = None if hip_z_R is None else np.asarray(hip_z_R, dtype=np.float64)
        if pose is None:
            self._pose = (float(self.ankle_L[0]), float(self.ankle_R[0]))
        else:
            self._pose = (float(pose[0]), float(pose[1]))

    def _at(self, arr, t_ms):
        if arr is None:
            return None
        t = float(t_ms)
        if self.loop and self.period_ms > 0.0:
            t = t % self.period_ms
        i = int(round(t / self.dt_ms))
        i = min(max(i, 0), len(arr) - 1)
        return float(arr[i])

    def pose(self) -> tuple[float, float]:
        return self._pose

    def target(self, t_ms: float) -> dict:
        return {
            "target_ankle_L": self._at(self.ankle_L, t_ms),
            "target_ankle_R": self._at(self.ankle_R, t_ms),
            "target_hip_L": self._at(self.hip_L, t_ms),
            "target_knee_L": self._at(self.knee_L, t_ms),
            "target_hip_R": self._at(self.hip_R, t_ms),
            "target_knee_R": self._at(self.knee_R, t_ms),
            "target_hipx_L": self._at(self.hip_x_L, t_ms),
            "target_hipz_L": self._at(self.hip_z_L, t_ms),
            "target_hipx_R": self._at(self.hip_x_R, t_ms),
            "target_hipz_R": self._at(self.hip_z_R, t_ms),
        }


@dataclass
class RunLog:
    time_ms: List[float] = field(default_factory=list)
    stage: List[str] = field(default_factory=list)

    # Tracked output vs target (rad) and the PERSONA learning signals
    ankle_model_L: List[float] = field(default_factory=list)
    ankle_model_R: List[float] = field(default_factory=list)
    ankle_target_L: List[float] = field(default_factory=list)
    ankle_target_R: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    advantage: List[float] = field(default_factory=list)
    adv_L: List[float] = field(default_factory=list)
    adv_R: List[float] = field(default_factory=list)

    # Locomotion phase (model: contact-driven; ref: geometric)
    phase_L: List[str] = field(default_factory=list)
    phase_R: List[str] = field(default_factory=list)
    phase_ref_L: List[str] = field(default_factory=list)
    phase_ref_R: List[str] = field(default_factory=list)

    # Muscle drives
    mus_E_L: List[float] = field(default_factory=list)
    mus_F_L: List[float] = field(default_factory=list)
    mus_E_R: List[float] = field(default_factory=list)
    mus_F_R: List[float] = field(default_factory=list)

    # Rhythm-generator population activity [0, 1]
    rg_e_L: List[float] = field(default_factory=list)
    rg_f_L: List[float] = field(default_factory=list)
    rg_e_R: List[float] = field(default_factory=list)
    rg_f_R: List[float] = field(default_factory=list)

    # Cutaneous afferents (Hz)
    cut_L: List[float] = field(default_factory=list)
    cut_R: List[float] = field(default_factory=list)

    # Ia afferents (spindle: length m, stretch velocity m/s, rate Hz)
    ia_len_E_L: List[float] = field(default_factory=list)
    ia_vel_E_L: List[float] = field(default_factory=list)
    ia_rate_E_L: List[float] = field(default_factory=list)
    ia_len_F_L: List[float] = field(default_factory=list)
    ia_vel_F_L: List[float] = field(default_factory=list)
    ia_rate_F_L: List[float] = field(default_factory=list)
    ia_len_E_R: List[float] = field(default_factory=list)
    ia_vel_E_R: List[float] = field(default_factory=list)
    ia_rate_E_R: List[float] = field(default_factory=list)
    ia_len_F_R: List[float] = field(default_factory=list)
    ia_vel_F_R: List[float] = field(default_factory=list)
    ia_rate_F_R: List[float] = field(default_factory=list)

    # Ib afferents (Golgi tendon organ: activation, tendon force N, rate Hz)
    ib_act_E_L: List[float] = field(default_factory=list)
    ib_force_E_L: List[float] = field(default_factory=list)
    ib_rate_E_L: List[float] = field(default_factory=list)
    ib_act_F_L: List[float] = field(default_factory=list)
    ib_force_F_L: List[float] = field(default_factory=list)
    ib_rate_F_L: List[float] = field(default_factory=list)
    ib_act_E_R: List[float] = field(default_factory=list)
    ib_force_E_R: List[float] = field(default_factory=list)
    ib_rate_E_R: List[float] = field(default_factory=list)
    ib_act_F_R: List[float] = field(default_factory=list)
    ib_force_F_R: List[float] = field(default_factory=list)
    ib_rate_F_R: List[float] = field(default_factory=list)

    # Joint angles (rad): controlled model and kinematic target
    hip_model_L: List[float] = field(default_factory=list)
    hip_model_R: List[float] = field(default_factory=list)
    knee_model_L: List[float] = field(default_factory=list)
    knee_model_R: List[float] = field(default_factory=list)
    hip_ref_L: List[float] = field(default_factory=list)
    hip_ref_R: List[float] = field(default_factory=list)
    knee_ref_L: List[float] = field(default_factory=list)
    knee_ref_R: List[float] = field(default_factory=list)
    ankle_ref_L: List[float] = field(default_factory=list)
    ankle_ref_R: List[float] = field(default_factory=list)

    # Contact / foot diagnostics
    contact_m_L: List[float] = field(default_factory=list)
    contact_m_R: List[float] = field(default_factory=list)
    contact_ref_L: List[float] = field(default_factory=list)
    contact_ref_R: List[float] = field(default_factory=list)
    foot_x_m_L: List[float] = field(default_factory=list)
    foot_x_m_R: List[float] = field(default_factory=list)
    foot_z_m_L: List[float] = field(default_factory=list)
    foot_z_m_R: List[float] = field(default_factory=list)
    foot_z_ref_L: List[float] = field(default_factory=list)
    foot_z_ref_R: List[float] = field(default_factory=list)

    # STDP weight-recorder trace (populated by the loop, h5-only)
    stdp: Dict[str, Any] = field(default_factory=dict)

    _NUMERIC = ("time_ms", "ankle_model_L", "ankle_model_R", "ankle_target_L",
                "ankle_target_R", "loss", "advantage", "adv_L", "adv_R",
                "mus_E_L", "mus_F_L", "mus_E_R", "mus_F_R",
                "rg_e_L", "rg_f_L", "rg_e_R", "rg_f_R",
                "cut_L", "cut_R",
                "ia_len_E_L", "ia_vel_E_L", "ia_rate_E_L",
                "ia_len_F_L", "ia_vel_F_L", "ia_rate_F_L",
                "ia_len_E_R", "ia_vel_E_R", "ia_rate_E_R",
                "ia_len_F_R", "ia_vel_F_R", "ia_rate_F_R",
                "ib_act_E_L", "ib_force_E_L", "ib_rate_E_L",
                "ib_act_F_L", "ib_force_F_L", "ib_rate_F_L",
                "ib_act_E_R", "ib_force_E_R", "ib_rate_E_R",
                "ib_act_F_R", "ib_force_F_R", "ib_rate_F_R",
                "hip_model_L", "hip_model_R", "knee_model_L", "knee_model_R",
                "hip_ref_L", "hip_ref_R", "knee_ref_L", "knee_ref_R",
                "ankle_ref_L", "ankle_ref_R",
                "contact_m_L", "contact_m_R", "contact_ref_L", "contact_ref_R",
                "foot_x_m_L", "foot_x_m_R", "foot_z_m_L", "foot_z_m_R",
                "foot_z_ref_L", "foot_z_ref_R")
    _STR = ("phase_L", "phase_R", "phase_ref_L", "phase_ref_R", "stage")

    def finalize(self) -> "RunLog":
        for name in self._NUMERIC:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        return self

    def to_csv(self, path: str) -> None:
        import csv
        cols = list(self._NUMERIC) + list(self._STR)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for i in range(len(self.stage)):
                w.writerow([getattr(self, c)[i] for c in cols])


class Personalizer:
    """
    Two-stage online personalization:
      Stage 1 ("pose")  — hold the target at its initial pose until the model's
                          two ankle errors stay below `eps_pose` for
                          `pose_hold_steps` consecutive steps.
      Stage 2 ("track") — follow the mocap target, optimizing until the
                          windowed-mean loss drops below `loss_stop`.
    Every step is an ordinary PERSONA update (sample -> apply -> simulate -> step).
    """
    def __init__(self,
                 registry: Registry,
                 cpg: Any,
                 rig: Any,
                 agent: Agent,
                 target_driver: Any,
                 dt_ms: float = 10.0,
                 eps_pose: float = 0.08,
                 pose_hold_steps: int = 20,
                 loss_stop: float = 0.05,
                 loss_window: int = 30,
                 max_steps: int = 1000,
                 log_every: int = 20,
                 stdp_every: int = 100,
                 verbose: bool = True,
                 belt_mps: float = 0.2,
                 belt_start_step: int = 20,
                 belt_ramp_steps: int = 50):
        self.registry = registry
        self.cpg = cpg
        self.rig = rig
        self.agent = agent
        self.target_driver = target_driver
        self.dt_ms = float(dt_ms)
        self.eps_pose = float(eps_pose)
        self.pose_hold_steps = int(pose_hold_steps)
        self.loss_stop = float(loss_stop)
        self.loss_window = int(loss_window)
        self.max_steps = int(max_steps)
        self.log_every = int(log_every)
        self.stdp_every = int(stdp_every)
        self.verbose = bool(verbose)
        self.belt_mps = float(belt_mps)
        self.belt_start_step = int(belt_start_step)
        self.belt_ramp_steps = max(1, int(belt_ramp_steps))
        self._seen_apply_errors: set[tuple[str, str]] = set()

    def _log_new_apply_errors(self):
        """Prints every newly surfaced realization failure exactly once."""
        for key, msg in self.registry.last_apply_errors:
            entry = (str(key), str(msg))
            if entry not in self._seen_apply_errors:
                self._seen_apply_errors.add(entry)
                print(f"[apply-error] {entry[0]}: {entry[1]}", flush=True)

    def _log_progress(self, log: RunLog, stage: str):
        if not self.verbose:
            return
        i = len(log.stage) - 1
        print(
            f"[{log.time_ms[i]:7.0f} ms] {stage:5s} "
            f"| L model {log.ankle_model_L[i]:+.3f} tgt {log.ankle_target_L[i]:+.3f} "
            f"| R model {log.ankle_model_R[i]:+.3f} tgt {log.ankle_target_R[i]:+.3f} "
            f"| loss {log.loss[i]:.4f} adv {log.advantage[i]:+.4f} "
            f"| drive E/F {log.mus_E_L[i]:.2f}/{log.mus_F_L[i]:.2f} "
            f"| CUT {log.cut_L[i]:.0f} Ia {log.ia_rate_E_L[i]:.0f} Ib {log.ib_rate_E_L[i]:.0f}",
            flush=True)

    def _log_plastic(self, log: RunLog, extra: dict, t_ms: float):
        """
        Records the STDP weight-recorder trace: per step the mean weight change
        and the number of updates the recorders saw, plus (every `stdp_every`
        steps) the live weight distribution of each plastic projection.
        """
        st = log.stdp
        if "delta" not in st:
            st.update({"delta": {}, "n": {}, "sample_time_ms": [],
                       "weight_mean": {}, "weight_min": {}, "weight_max": {}})
        plastic = extra.get("plastic") or {}
        for key, val in (plastic.get("delta") or {}).items():
            st["delta"].setdefault(key, []).append(float(val))
        for key, val in (plastic.get("n") or {}).items():
            st["n"].setdefault(key, []).append(int(val))
        if self.stdp_every > 0 and len(log.stage) % self.stdp_every == 0:
            st["sample_time_ms"].append(float(t_ms))
            for key, s in self.registry.sample_plastic_weight_stats().items():
                st["weight_mean"].setdefault(key, []).append(s["mean"])
                st["weight_min"].setdefault(key, []).append(s["min"])
                st["weight_max"].setdefault(key, []).append(s["max"])

    def run(self, step_hook=None, wall_clock_s: float | None = None,
            pose_max_steps: int = 0, realize_every_k: int = 1) -> RunLog:
        """
        Runs the two-stage loop. `step_hook(step, log, cpg_in, cpg_out, stage)`
        is invoked after every step (e.g. to render a frame). When
        `wall_clock_s` is given the loop stops once that much real time has
        elapsed, so a run fits a server time budget regardless of step cost.
        `pose_max_steps` caps the settling ("pose") stage so the loop always
        proceeds to tracking even when contact-driven CPG activity keeps the
        ankle error above `eps_pose`.

        `realize_every_k` (spec §5.2): sample + realize + record the network
        every k-th step only; the strategy still updates every dt from the
        held (recorded) sample, so `k > 1` trades learning noise for wall-clock
        without changing the update rule.
        """
        import time
        t_start = time.monotonic()
        k_realize = max(1, int(realize_every_k))
        log = RunLog()
        cpg_in = CPGInput()
        stage = "pose"
        pose_ok = 0
        pose_steps = 0
        # construction-time failures surface before the first apply clears them
        self._log_new_apply_errors()

        for step in range(self.max_steps):
            if step >= self.belt_start_step and hasattr(self.rig, "set_treadmill_surfacevel"):
                frac = min(1.0, (step - self.belt_start_step + 1) / self.belt_ramp_steps)
                self.rig.set_treadmill_surfacevel(self.belt_mps * frac)
            if stage == "pose":
                # Hold the whole target (incl. the mocap hip base) at t=0.
                rig_kwargs = self.target_driver.target(0.0)
            else:
                rig_kwargs = self.target_driver.target(self.cpg.current_time_ms)
            tgt_L = rig_kwargs["target_ankle_L"]
            tgt_R = rig_kwargs["target_ankle_R"]

            if step % k_realize == 0:
                actions = self.agent.sample_actions(t_ms=self.cpg.current_time_ms)
                self.registry.apply_explored_parameters(actions)
                self._log_new_apply_errors()

            self.cpg.update(cpg_in)
            cpg_out = self.cpg.step()
            cpg_in = self.rig.step(cpg_out, **rig_kwargs)

            res = self.agent.step(
                output=np.array([cpg_in.ankle_angle_m_L, cpg_in.ankle_angle_m_R]),
                target=np.array([tgt_L, tgt_R]),
                phase=[cpg_in.phase_L, cpg_in.phase_R],
                t_ms=self.cpg.current_time_ms,
            )

            log.time_ms.append(self.cpg.current_time_ms)
            log.stage.append(stage)
            log.ankle_model_L.append(cpg_in.ankle_angle_m_L)
            log.ankle_model_R.append(cpg_in.ankle_angle_m_R)
            log.ankle_target_L.append(tgt_L)
            log.ankle_target_R.append(tgt_R)
            log.loss.append(float(res["loss"]))
            adv = np.asarray(res["advantage_vector"], dtype=np.float64).ravel()
            log.advantage.append(float(np.mean(adv)) if adv.size else 0.0)
            log.adv_L.append(float(adv[0]) if adv.size > 0 else 0.0)
            log.adv_R.append(float(adv[1]) if adv.size > 1 else 0.0)
            log.phase_L.append(cpg_in.phase_L)
            log.phase_R.append(cpg_in.phase_R)
            log.phase_ref_L.append(cpg_in.phase_ref_L)
            log.phase_ref_R.append(cpg_in.phase_ref_R)
            log.mus_E_L.append(cpg_out.mus_E_L)
            log.mus_F_L.append(cpg_out.mus_F_L)
            log.mus_E_R.append(cpg_out.mus_E_R)
            log.mus_F_R.append(cpg_out.mus_F_R)
            rg = (getattr(cpg_out, "extra", None) or {}).get("rg") or {}
            log.rg_e_L.append(float(rg.get("rg_e_L", 0.0)))
            log.rg_f_L.append(float(rg.get("rg_f_L", 0.0)))
            log.rg_e_R.append(float(rg.get("rg_e_R", 0.0)))
            log.rg_f_R.append(float(rg.get("rg_f_R", 0.0)))
            log.cut_L.append(cpg_in.cut_L)
            log.cut_R.append(cpg_in.cut_R)
            log.ia_len_E_L.append(cpg_in.ia_len_E_L)
            log.ia_vel_E_L.append(cpg_in.ia_vel_E_L)
            log.ia_rate_E_L.append(cpg_in.ia_rate_E_L)
            log.ia_len_F_L.append(cpg_in.ia_len_F_L)
            log.ia_vel_F_L.append(cpg_in.ia_vel_F_L)
            log.ia_rate_F_L.append(cpg_in.ia_rate_F_L)
            log.ia_len_E_R.append(cpg_in.ia_len_E_R)
            log.ia_vel_E_R.append(cpg_in.ia_vel_E_R)
            log.ia_rate_E_R.append(cpg_in.ia_rate_E_R)
            log.ia_len_F_R.append(cpg_in.ia_len_F_R)
            log.ia_vel_F_R.append(cpg_in.ia_vel_F_R)
            log.ia_rate_F_R.append(cpg_in.ia_rate_F_R)
            log.ib_act_E_L.append(cpg_in.ib_act_E_L)
            log.ib_force_E_L.append(cpg_in.ib_force_E_L)
            log.ib_rate_E_L.append(cpg_in.ib_rate_E_L)
            log.ib_act_F_L.append(cpg_in.ib_act_F_L)
            log.ib_force_F_L.append(cpg_in.ib_force_F_L)
            log.ib_rate_F_L.append(cpg_in.ib_rate_F_L)
            log.ib_act_E_R.append(cpg_in.ib_act_E_R)
            log.ib_force_E_R.append(cpg_in.ib_force_E_R)
            log.ib_rate_E_R.append(cpg_in.ib_rate_E_R)
            log.ib_act_F_R.append(cpg_in.ib_act_F_R)
            log.ib_force_F_R.append(cpg_in.ib_force_F_R)
            log.ib_rate_F_R.append(cpg_in.ib_rate_F_R)
            log.hip_model_L.append(cpg_in.hip_angle_m_L)
            log.hip_model_R.append(cpg_in.hip_angle_m_R)
            log.knee_model_L.append(cpg_in.knee_angle_m_L)
            log.knee_model_R.append(cpg_in.knee_angle_m_R)
            log.hip_ref_L.append(cpg_in.hip_angle_ref_L)
            log.hip_ref_R.append(cpg_in.hip_angle_ref_R)
            log.knee_ref_L.append(cpg_in.knee_angle_ref_L)
            log.knee_ref_R.append(cpg_in.knee_angle_ref_R)
            log.ankle_ref_L.append(cpg_in.ankle_angle_ref_L)
            log.ankle_ref_R.append(cpg_in.ankle_angle_ref_R)
            log.contact_m_L.append(cpg_in.contact_m_L)
            log.contact_m_R.append(cpg_in.contact_m_R)
            log.contact_ref_L.append(cpg_in.contact_ref_L)
            log.contact_ref_R.append(cpg_in.contact_ref_R)
            log.foot_x_m_L.append(cpg_in.foot_x_m_L)
            log.foot_x_m_R.append(cpg_in.foot_x_m_R)
            log.foot_z_m_L.append(cpg_in.foot_z_m_L)
            log.foot_z_m_R.append(cpg_in.foot_z_m_R)
            log.foot_z_ref_L.append(cpg_in.foot_z_ref_L)
            log.foot_z_ref_R.append(cpg_in.foot_z_ref_R)
            self._log_plastic(log, getattr(cpg_out, "extra", None) or {},
                              self.cpg.current_time_ms)

            if step_hook is not None:
                step_hook(step, log, cpg_in, cpg_out, stage)

            if self.log_every > 0 and (len(log.stage) % self.log_every == 0
                                       or len(log.stage) == 1):
                self._log_progress(log, stage)

            if stage == "pose":
                err = max(abs(cpg_in.ankle_angle_m_L - tgt_L),
                          abs(cpg_in.ankle_angle_m_R - tgt_R))
                pose_ok = pose_ok + 1 if err < self.eps_pose else 0
                pose_steps += 1
                if pose_ok >= self.pose_hold_steps or \
                        (pose_max_steps and pose_steps >= pose_max_steps):
                    stage = "track"
            else:
                if len(log.loss) >= self.loss_window and \
                        float(np.mean(log.loss[-self.loss_window:])) < self.loss_stop:
                    break

            if wall_clock_s is not None and time.monotonic() - t_start >= wall_clock_s:
                break

        if log.stdp:
            log.stdp["time_ms"] = np.asarray(log.time_ms, dtype=np.float64)
            log.stdp["sample_time_ms"] = np.asarray(log.stdp["sample_time_ms"], dtype=np.float64)
            for grp in ("delta", "n", "weight_mean", "weight_min", "weight_max"):
                log.stdp[grp] = {k: np.asarray(v, dtype=np.float64)
                                 for k, v in log.stdp[grp].items()}
            log.stdp["final_weights"] = self.registry.final_plastic_weights()

        return log.finalize()