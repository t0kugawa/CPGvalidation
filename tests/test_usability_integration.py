import numpy as np

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.schedule import CosineLR


def test_freeze_spring_and_cosine_coexist(nest_kernel):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
    }, "pop"))
    reg.freeze("pop_pop_a", attrs=("mean",))
    reg.create_spring("s", "pop_pop_b", power=0.5)
    agent = Agent(registry=reg, lr_mu=0.2, lr_sigma=0.0, warmup=0,
                  dt_ms=10.0, lr_schedule=CosineLR(0.2, 0.0, 5))
    agent.params["pop_pop_b"].mean = 2.0
    for t in range(5):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=t * 10.0)
    assert agent.params["pop_pop_a"].mean == 1.0
    assert agent.params["pop_pop_b"].mean < 2.0
    assert len(agent.history["lr_mu"]) == 5
    assert "pop_pop_b" in agent.history["spring"][-1]


def test_default_agent_has_no_new_history_keys(nest_kernel):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 3,
                          {"a": PersonaParam(mean=1.0, sigma=0.1)}, "pop"))
    agent = Agent(registry=reg, warmup=0)
    agent.sample_actions(t_ms=0.0)
    agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert "lr_mu" not in agent.history
    assert "spring" not in agent.history
