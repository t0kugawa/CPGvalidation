import numpy as np
import pytest

from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.agent import Agent


@pytest.fixture
def make_agent(nest_kernel):
    def _make():
        reset_rng_stream(0)
        reg = Registry()
        reg.create_population(Population("izhikevich", 5,
                              {"a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0)},
                              "pop"))
        return Agent(registry=reg, lr_mu=0.2, lr_sigma_ctrl=0.0,
                     window_size=40), reg
    return _make


def test_loss_decreases_on_coupled_target(make_agent):
    agent, reg = make_agent()
    key = "pop_pop_a"
    for _ in range(200):
        agent.sample_actions()
        reg.apply_explored_parameters(agent.current_actions)
        a = agent.current_actions[key]
        out = np.array([abs(a - 2.0)] * 2)   # plant y = |a-2|; optimum at a=2.0
        agent.step(output=out, target=np.array([0.0, 0.0]),
                   phase=["stance", "swing"])
    assert abs(agent.params[key].mean - 2.0) < 0.5


def test_influence_logged(make_agent):
    agent, reg = make_agent(); agent.sample_actions()
    agent.step(np.zeros(2), np.ones(2), ["stance", "swing"])
    assert "influence" in agent.history and len(agent.history["influence"]) == 1