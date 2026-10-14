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
        p = PersonaParam(mean=1.0, sigma_param=0.01, sigma_ctrl=0.1, min=0.0, max=2.0)
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