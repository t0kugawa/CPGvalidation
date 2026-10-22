import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry


def _agent():
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
    }, "pop"))
    return Agent(registry=reg, lr_mu=0.5, lr_sigma=0.5, warmup=0,
                 dt_ms=10.0, tau_mem_s=0.2), reg


def test_frozen_mean_holds_and_sigma_still_updates(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.freeze(key, attrs=("mean",))
    sigma0 = agent.params[key].sigma
    last = None
    for t in range(20):
        agent.sample_actions(t_ms=t * 10.0)
        last = agent.step(np.zeros(2), np.ones(2) * t, phase="stance",
                          t_ms=t * 10.0)
    assert agent.params[key].mean == 1.0
    assert agent.params[key].sigma != sigma0
    assert last["delta_means"][key] == 0.0


def test_frozen_sigma_holds_and_trace_records_zero(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.freeze(key, attrs=("sigma",))
    sigma0 = agent.params[key].sigma
    seen = {}
    orig = agent.strategy.step

    def spy(sites, noise, advantage, **kwargs):
        seen.update(sites)
        return orig(sites, noise, advantage, **kwargs)

    agent.strategy.step = spy
    for t in range(20):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * t, phase="stance",
                   t_ms=t * 10.0)
    assert agent.params[key].sigma == sigma0
    assert seen[key][2] == 0.0
    assert agent.params[key].mean != 1.0


def test_freezing_both_then_unfreezing_resumes_updates(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.freeze(key, attrs=("mean", "sigma"))
    sigma0 = agent.params[key].sigma
    for t in range(20):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * t, phase="stance",
                   t_ms=t * 10.0)
    assert (agent.params[key].mean, agent.params[key].sigma) == (1.0, sigma0)
    reg.unfreeze(key)
    for t in range(20, 25):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * t, phase="stance", t_ms=t * 10.0)
    assert (agent.params[key].mean, agent.params[key].sigma) != (1.0, sigma0)
