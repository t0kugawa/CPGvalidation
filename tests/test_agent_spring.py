import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry


def _agent(warmup=0):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
    }, "pop"))
    return Agent(registry=reg, lr_mu=0.0, lr_sigma=0.0, warmup=warmup,
                 dt_ms=10.0), reg


def test_custom_spring_moves_mean_and_is_logged(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.create_spring("s", key, power=0.5)
    agent.params[key].mean = 2.0
    agent.sample_actions(t_ms=0.0)
    res = agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert agent.params[key].mean == pytest.approx(1.5)
    assert res["delta_means"][key] == pytest.approx(-0.5)
    assert agent.history["spring"][-1][key]["force"] == pytest.approx(-0.5)


def test_spring_does_not_apply_during_warmup(nest_kernel):
    agent, reg = _agent(warmup=3)
    key = "pop_pop_a"
    reg.create_spring("s", key, power=0.5)
    agent.params[key].mean = 2.0
    for t in range(3):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=t * 10.0)
    assert agent.params[key].mean == 2.0
    assert agent.history["spring"][-1] == {}
    agent.sample_actions(t_ms=30.0)
    agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=30.0)
    assert agent.params[key].mean == pytest.approx(1.5)


def test_spring_skips_frozen_mean(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    agent.params[key].mean = 2.0
    reg.freeze(key, attrs=("mean",))          # snaps back to the creation anchor
    reg.create_spring("s", key, power=0.5)
    agent.sample_actions(t_ms=0.0)
    res = agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert agent.params[key].mean == 1.0
    assert res["delta_means"][key] == 0.0
    assert agent.history["spring"][-1] == {}
