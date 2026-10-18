import numpy as np
import pytest

from persona.parameter import PersonaParam
from persona.population import Population
from persona.registry import Registry
from persona.agent import Agent
from persona.strategy import PhaseBaselines, FisherMatrix


@pytest.fixture
def make_agent(nest_kernel):
    def _make(n: int = 5):
        reg = Registry()
        p = PersonaParam(mean=1.0, sigma=0.1, min=0.0, max=2.0)
        reg.create_population(
            Population(model="izhikevich", n=n, params={"a": p}, name="pop")
        )
        return Agent(registry=reg), reg

    return _make


def test_per_component_phase_independent():
    fm = FisherMatrix(param_names=["p"], window_size=10)
    fm.baselines = PhaseBaselines(alpha=0.0)
    fm.compute_advantage(np.array([2.0, 2.0]), ("stance", "swing"))
    a = fm.compute_advantage(np.array([1.0, 2.0]), ("stance", "swing"))
    assert np.isclose(a[0], 1.0) and np.isclose(a[1], 0.0)


def test_baselines_are_per_component():
    fm = FisherMatrix(param_names=["p"], window_size=10)
    fm.baselines = PhaseBaselines(alpha=0.0)
    # L stance (loss 2), R swing (loss 5)
    fm.compute_advantage(np.array([2.0, 5.0]), ("stance", "swing"))
    # L now swings: no prior L-swing baseline; R swing baseline is 5
    a = fm.compute_advantage(np.array([3.0, 4.0]), ("swing", "swing"))
    assert np.isclose(a[0], -3.0)
    assert np.isclose(a[1], 1.0)
    # L back to stance: its stance baseline (2) is used, not R's
    a2 = fm.compute_advantage(np.array([1.0, 4.0]), ("stance", "swing"))
    assert np.isclose(a2[0], 1.0)


def test_output_and_target_share_smoother(make_agent):
    agent, _ = make_agent()
    for _ in range(4):
        agent.sample_actions()
        agent.step(output=np.array([0.5, 0.5]), target=np.array([0.5, 0.5]),
                   phase="stance")
    assert np.allclose(agent.history["loss"], 0.0)


def test_agent_accepts_phase_vector(make_agent):
    agent, _ = make_agent()
    agent.sample_actions()
    agent.step(np.zeros(2), np.ones(2), phase=["stance", "swing"])
    assert agent.history["phase"][-1] == ("stance", "swing")


def test_agent_broadcasts_string_phase(make_agent):
    agent, _ = make_agent()
    agent.sample_actions()
    agent.step(np.zeros(2), np.ones(2), phase="stance")
    assert agent.history["phase"][-1] == "stance"