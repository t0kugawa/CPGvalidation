import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.schedule import CosineLR


def _agent(schedule=None):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 3,
                          {"a": PersonaParam(mean=1.0, sigma=0.1)}, "pop"))
    return Agent(registry=reg, lr_mu=0.2, lr_sigma=0.0, warmup=0,
                 dt_ms=10.0, tau_mem_s=0.2, lr_schedule=schedule), reg


def test_schedule_drives_strategy_lr(nest_kernel):
    agent, _ = _agent(CosineLR(start=0.2, end=0.0, total_steps=4))
    seen = []
    orig = agent.strategy.step

    def spy(sites, noise, advantage, **kwargs):
        seen.append(agent.strategy.lr_mu)
        return orig(sites, noise, advantage, **kwargs)

    agent.strategy.step = spy
    for t in range(4):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=t * 10.0)
    assert seen == pytest.approx([0.2, 0.15, 0.05, 0.0])
    assert agent.history["lr_mu"] == pytest.approx([0.2, 0.15, 0.05, 0.0])


def test_no_schedule_keeps_fixed_lr_and_no_history_key(nest_kernel):
    agent, _ = _agent()
    assert agent.current_lr_mu == 0.2
    agent.sample_actions(t_ms=0.0)
    agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert agent.strategy.lr_mu == 0.2
    assert "lr_mu" not in agent.history
