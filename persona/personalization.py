from dataclasses import dataclass, field
from typing import Any, List

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
                 hip_L=None, knee_L=None, hip_R=None, knee_R=None):
        self.ankle_L = np.asarray(ankle_L, dtype=np.float64)
        self.ankle_R = np.asarray(ankle_R, dtype=np.float64)
        self.dt_ms = float(dt_ms)
        self.hip_L = None if hip_L is None else np.asarray(hip_L, dtype=np.float64)
        self.knee_L = None if knee_L is None else np.asarray(knee_L, dtype=np.float64)
        self.hip_R = None if hip_R is None else np.asarray(hip_R, dtype=np.float64)
        self.knee_R = None if knee_R is None else np.asarray(knee_R, dtype=np.float64)
        if pose is None:
            self._pose = (float(self.ankle_L[0]), float(self.ankle_R[0]))
        else:
            self._pose = (float(pose[0]), float(pose[1]))

    def _at(self, arr, t_ms):
        if arr is None:
            return None
        i = int(round(float(t_ms) / self.dt_ms))
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
        }


@dataclass
class RunLog:
    time_ms: List[float] = field(default_factory=list)
    ankle_model_L: List[float] = field(default_factory=list)
    ankle_model_R: List[float] = field(default_factory=list)
    ankle_target_L: List[float] = field(default_factory=list)
    ankle_target_R: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    advantage: List[float] = field(default_factory=list)
    phase_L: List[str] = field(default_factory=list)
    phase_R: List[str] = field(default_factory=list)
    stage: List[str] = field(default_factory=list)
    mus_E_L: List[float] = field(default_factory=list)
    mus_F_L: List[float] = field(default_factory=list)
    mus_E_R: List[float] = field(default_factory=list)
    mus_F_R: List[float] = field(default_factory=list)
    cut_L: List[float] = field(default_factory=list)
    ia_rate_E_L: List[float] = field(default_factory=list)
    ib_rate_E_L: List[float] = field(default_factory=list)
    ankle_ref_L: List[float] = field(default_factory=list)
    ankle_ref_R: List[float] = field(default_factory=list)
    knee_model_L: List[float] = field(default_factory=list)
    knee_model_R: List[float] = field(default_factory=list)
    hip_model_L: List[float] = field(default_factory=list)
    hip_model_R: List[float] = field(default_factory=list)
    ia_rate_F_L: List[float] = field(default_factory=list)

    _NUMERIC = ("time_ms", "ankle_model_L", "ankle_model_R", "ankle_target_L",
                "ankle_target_R", "loss", "advantage", "mus_E_L", "mus_F_L",
                "mus_E_R", "mus_F_R", "cut_L", "ia_rate_E_L", "ib_rate_E_L",
                "ankle_ref_L", "ankle_ref_R", "knee_model_L", "knee_model_R",
                "hip_model_L", "hip_model_R", "ia_rate_F_L")

    def finalize(self) -> "RunLog":
        for name in self._NUMERIC:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        return self

    def to_csv(self, path: str) -> None:
        import csv
        cols = list(self._NUMERIC) + ["phase_L", "phase_R", "stage"]
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
        self.verbose = bool(verbose)
        self.belt_mps = float(belt_mps)
        self.belt_start_step = int(belt_start_step)
        self.belt_ramp_steps = max(1, int(belt_ramp_steps))

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

    def run(self) -> RunLog:
        log = RunLog()
        cpg_in = CPGInput()
        stage = "pose"
        pose_ok = 0

        for step in range(self.max_steps):
            if step >= self.belt_start_step and hasattr(self.rig, "set_treadmill_surfacevel"):
                frac = min(1.0, (step - self.belt_start_step + 1) / self.belt_ramp_steps)
                self.rig.set_treadmill_surfacevel(self.belt_mps * frac)
            if stage == "pose":
                tgt_L, tgt_R = self.target_driver.pose()
                rig_kwargs = {"target_ankle_L": tgt_L, "target_ankle_R": tgt_R}
            else:
                rig_kwargs = self.target_driver.target(self.cpg.current_time_ms)
                tgt_L = rig_kwargs["target_ankle_L"]
                tgt_R = rig_kwargs["target_ankle_R"]

            actions = self.agent.sample_actions()
            self.registry.apply_explored_parameters(actions)

            self.cpg.update(cpg_in)
            cpg_out = self.cpg.step()
            cpg_in = self.rig.step(cpg_out, **rig_kwargs)

            res = self.agent.step(
                output=np.array([cpg_in.ankle_angle_m_L, cpg_in.ankle_angle_m_R]),
                target=np.array([tgt_L, tgt_R]),
                phase=[cpg_in.phase_L, cpg_in.phase_R],
            )

            log.time_ms.append(self.cpg.current_time_ms)
            log.ankle_model_L.append(cpg_in.ankle_angle_m_L)
            log.ankle_model_R.append(cpg_in.ankle_angle_m_R)
            log.ankle_target_L.append(tgt_L)
            log.ankle_target_R.append(tgt_R)
            log.loss.append(float(res["loss"]))
            log.advantage.append(float(np.mean(res["advantage_vector"])))
            log.phase_L.append(cpg_in.phase_L)
            log.phase_R.append(cpg_in.phase_R)
            log.stage.append(stage)
            log.mus_E_L.append(cpg_out.mus_E_L)
            log.mus_F_L.append(cpg_out.mus_F_L)
            log.mus_E_R.append(cpg_out.mus_E_R)
            log.mus_F_R.append(cpg_out.mus_F_R)
            log.cut_L.append(cpg_in.cut_L)
            log.ia_rate_E_L.append(cpg_in.ia_rate_E_L)
            log.ib_rate_E_L.append(cpg_in.ib_rate_E_L)
            log.ankle_ref_L.append(cpg_in.ankle_angle_ref_L)
            log.ankle_ref_R.append(cpg_in.ankle_angle_ref_R)
            log.knee_model_L.append(cpg_in.knee_angle_m_L)
            log.knee_model_R.append(cpg_in.knee_angle_m_R)
            log.hip_model_L.append(cpg_in.hip_angle_m_L)
            log.hip_model_R.append(cpg_in.hip_angle_m_R)
            log.ia_rate_F_L.append(cpg_in.ia_rate_F_L)

            if self.log_every > 0 and (len(log.stage) % self.log_every == 0
                                       or len(log.stage) == 1):
                self._log_progress(log, stage)

            if stage == "pose":
                err = max(abs(cpg_in.ankle_angle_m_L - tgt_L),
                          abs(cpg_in.ankle_angle_m_R - tgt_R))
                pose_ok = pose_ok + 1 if err < self.eps_pose else 0
                if pose_ok >= self.pose_hold_steps:
                    stage = "track"
            else:
                if len(log.loss) >= self.loss_window and \
                        float(np.mean(log.loss[-self.loss_window:])) < self.loss_stop:
                    break

        return log.finalize()