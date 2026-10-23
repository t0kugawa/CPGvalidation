"""ParallelPersonalizer loop with fake replicas (no NEST/MuJoCo).

This catches log-field and per-replica wiring errors without the simulators.
"""
import numpy as np

from model.adapter import CPGInput, CPGOutput
from persona.personalization import ParallelPersonalizer


class FakeCPG:
    def __init__(self, replicas):
        self.replicas = replicas
        self.current_time_ms = 0.0

    def update(self, cpg_inputs):
        assert len(cpg_inputs) == self.replicas

    def step(self):
        self.current_time_ms += 10.0
        return [
            CPGOutput(mus_E_L=0.2 + 0.1 * i, mus_F_L=0.3,
                      mus_E_R=0.4, mus_F_R=0.5,
                      t_ms=self.current_time_ms,
                      extra={"plastic": {"delta": {"p": 0.01},
                                         "n": {"p": 1}},
                             "rg": {"rg_e_L": 0.1, "rg_f_L": 0.2,
                                    "rg_e_R": 0.3, "rg_f_R": 0.4}})
            for i in range(self.replicas)
        ]


class FakeRig:
    def __init__(self, value):
        self.value = value

    def step(self, cpg_output, **kwargs):
        inp = CPGInput()
        inp.ankle_angle_m_L = self.value
        inp.ankle_angle_m_R = -self.value
        inp.phase_L, inp.phase_R = "stance", "swing"
        inp.phase_ref_L, inp.phase_ref_R = "stance", "swing"
        inp.cut_L = inp.cut_R = 5.0
        return inp


class FakeAgent:
    def __init__(self, replicas):
        self.replicas = replicas
        self.steps = 0
        self.closed = False

    def sample_actions(self, t_ms=None):
        return [{} for _ in range(self.replicas)]

    def apply_actions(self, actions):
        assert len(actions) == self.replicas

    def step(self, outputs, targets, ref_phases, model_phases=None, t_ms=None):
        self.steps += 1
        return {
            "loss": 0.1,
            "advantage_vector": np.zeros(2),
            "losses": [0.1] * self.replicas,
            "advantages": [np.zeros(2) for _ in range(self.replicas)],
            "merged": False,
        }

    def close(self):
        self.closed = True


class FakeRegistry:
    def __init__(self):
        self.last_apply_errors = []

    def sample_plastic_weight_stats(self):
        return {}

    def final_plastic_weights(self):
        return {}


class FakeDriver:
    def target(self, t_ms):
        return {
            "target_ankle_L": 0.1, "target_ankle_R": -0.1,
            "target_hip_L": 0.0, "target_knee_L": 0.0,
            "target_hip_R": 0.0, "target_knee_R": 0.0,
            "target_hipx_L": 0.0, "target_hipz_L": 0.0,
            "target_hipx_R": 0.0, "target_hipz_R": 0.0,
        }


def test_parallel_loop_logs_mean_values():
    replicas = 2
    personalizer = ParallelPersonalizer(
        registries=[FakeRegistry() for _ in range(replicas)],
        cpg=FakeCPG(replicas),
        rigs=[FakeRig(0.2), FakeRig(-0.2)],
        agent=FakeAgent(replicas),
        target_driver=FakeDriver(),
        max_steps=2,
        log_every=0,
        stdp_every=0,
        eps_pose=0.0,
        pose_hold_steps=1,
    )
    try:
        log = personalizer.run()
    finally:
        personalizer.close()

    assert len(log.stage) == 2
    assert np.allclose(log.ankle_model_L, [0.0, 0.0])
    assert np.allclose(log.mus_E_L, [0.25, 0.25])
    assert np.allclose(log.rg_e_L, [0.1, 0.1])
    assert np.allclose(log.loss, [0.1, 0.1])
    assert np.allclose(log.cut_L, [5.0, 5.0])
