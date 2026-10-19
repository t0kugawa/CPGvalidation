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


def test_advantage_is_baseline_minus_loss():
    ph = PhaseBaselines(alpha=0.0)
    fm = FisherMatrix(param_names=["p"])
    fm.baselines = ph
    a1 = fm.compute_advantage(np.array([1.0]), "stance")
    a2 = fm.compute_advantage(np.array([0.4]), "stance")
    assert np.isclose(a1[0], -1.0) and np.isclose(a2[0], 0.6)


def test_agent_logs_loss(make_agent):
    agent, _ = make_agent()
    for _ in range(3):
        agent.sample_actions()
        out = agent.step(output=np.array([0.0, 0.0]),
                         target=np.array([1.0, 1.0]), phase="stance")
    assert "loss" in out and len(agent.history["loss"]) == 3
    assert np.isfinite(agent.history["loss"]).all()


def test_cold_start_no_nan(make_agent):
    agent, _ = make_agent()
    for _ in range(2):
        agent.sample_actions()
        agent.step(output=np.array([0.0, 0.0]),
                   target=np.array([1.0, 1.0]), phase="stance")
    for _, p in agent.params.items():
        assert np.isfinite(p.mean)
        assert np.isfinite(p.sigma) and p.sigma > 0.0