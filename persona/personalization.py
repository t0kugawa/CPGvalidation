from dataclasses import dataclass, field
from typing import Any, List

import numpy as np

from persona.registry import Registry
from persona.agent import Agent
from model.adapter import CPGInput


class MocapTargetDriver:
    """
    Serves the target-limb ankle trajectory to the rig. `pose()` returns the
    Stage-1 initial pose; `target(t_ms)` returns the Stage-2 trajectory value
    (interpolated at the simulation time).
    """
    def __init__(self, ankle_L, ankle_R, dt_ms: float = 10.0, pose=None):
        self.ankle_L = np.asarray(ankle_L, dtype=np.float64)
        self.ankle_R = np.asarray(ankle_R, dtype=np.float64)
        self.dt_ms = float(dt_ms)
        if pose is None:
            self._pose = (float(self.ankle_L[0]), float(self.ankle_R[0]))
        else:
            self._pose = (float(pose[0]), float(pose[1]))

    def pose(self) -> tuple[float, float]:
        return self._pose

    def target(self, t_ms: float) -> tuple[float, float]:
        n = len(self.ankle_L)
        i = int(round(float(t_ms) / self.dt_ms))
        i = min(max(i, 0), n - 1)
        return float(self.ankle_L[i]), float(self.ankle_R[i])


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

    def finalize(self) -> "RunLog":
        for name in ("time_ms", "ankle_model_L", "ankle_model_R",
                     "ankle_target_L", "ankle_target_R", "loss", "advantage"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        return self


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
                 max_steps: int = 1000):
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

    def run(self) -> RunLog:
        log = RunLog()
        cpg_in = CPGInput()
        stage = "pose"
        pose_ok = 0

        for _ in range(self.max_steps):
            if stage == "pose":
                tgt = self.target_driver.pose()
            else:
                tgt = self.target_driver.target(self.cpg.current_time_ms)
            tgt_L, tgt_R = tgt

            actions = self.agent.sample_actions()
            self.registry.apply_explored_parameters(actions)

            self.cpg.update(cpg_in)
            cpg_out = self.cpg.step()
            cpg_in = self.rig.step(cpg_out, target_ankle_L=tgt_L, target_ankle_R=tgt_R)

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