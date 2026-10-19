import numpy as np

from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.agent import Agent


def _agent(lr_mu=0.05, lr_sigma=0.0, tau_mem_s=2.0):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5,
                          {"a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0)},
                          "pop"))
    return Agent(registry=reg, lr_mu=lr_mu, lr_sigma=lr_sigma,
                 tau_mem_s=tau_mem_s, dt_ms=10.0), reg


def test_loss_decreases_and_sigma_recorded(nest_kernel):
    # τ_mem = 0.2 s keeps the trace-based credit's effective gain (1−β) workable
    # in-test (see test_credit_gain_is_ewma_weighted).
    agent, reg = _agent(lr_mu=1.0, tau_mem_s=0.2)
    key = "pop_pop_a"
    for t in range(300):
        agent.sample_actions(t_ms=t * 10.0)
        actions = dict(agent.current_actions)
        reg.apply_explored_parameters(actions)
        rec = reg.site_recorder(key)
        assert rec.latest_sample() == (actions[key], agent.params[key].sigma)
        out = np.array([abs(actions[key] - 2.0)] * 2)   # plant y = |a-2|; optimum at a=2
        agent.step(output=out, target=np.zeros(2), phase=["stance", "swing"],
                   t_ms=t * 10.0)
    assert abs(agent.params[key].mean - 2.0) < 0.5
    assert len(agent.history["sigmas"][key]) == 300


def test_agent_passes_fresh_producer_deltas_and_declared_scale(nest_kernel):
    agent, reg = _agent()
    rec = reg.producer_recorder("stdp:x", scale=2.0)
    captured = {}
    orig = agent.strategy.step

    def spy(sites, noise, advantage):
        captured.clear()
        captured.update(noise)
        return orig(sites, noise, advantage)

    agent.strategy.step = spy
    rec.record_delta(0.0, 5.0)
    agent.sample_actions(t_ms=0.0)
    agent.step(np.zeros(2), np.zeros(2), phase="stance", t_ms=10.0)
    assert captured["producer:stdp:x"][0] == 5.0     # fresh value read
    assert captured["producer:stdp:x"][1] == 2.0     # declared scale, no warm-up
    rec.record_delta(10.0, 3.0)
    agent.sample_actions(t_ms=10.0)
    agent.step(np.zeros(2), np.zeros(2), phase="stance", t_ms=20.0)
    assert captured["producer:stdp:x"][0] == 3.0     # fresh value read
    assert captured["producer:stdp:x"][1] == 2.0     # scale is fixed, not estimated
    agent.sample_actions(t_ms=20.0)
    agent.step(np.zeros(2), np.zeros(2), phase="stance", t_ms=30.0)
    assert captured["producer:stdp:x"][0] == 0.0     # never the stale 3.0


def test_unscaled_producer_has_zero_declared_scale(nest_kernel):
    agent, reg = _agent()
    rec = reg.producer_recorder("stdp:no_scale")     # declaration omitted
    captured = {}
    orig = agent.strategy.step

    def spy(sites, noise, advantage):
        captured.update(noise)
        return orig(sites, noise, advantage)

    agent.strategy.step = spy
    rec.record_delta(0.0, 5.0)
    agent.sample_actions(t_ms=0.0)
    agent.step(np.zeros(2), np.zeros(2), phase="stance", t_ms=10.0)
    assert captured["producer:stdp:no_scale"] == (5.0, 0.0)


def test_sigma_frozen_during_warmup_then_updates(nest_kernel):
    agent, reg = _agent(lr_sigma=0.5)
    key = "pop_pop_a"
    s0 = agent.params[key].sigma
    for t in range(3):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * (1.0 + 0.3 * t), phase="stance",
                   t_ms=t * 10.0)
    assert agent.params[key].sigma == s0             # warm-up accumulates only
    agent.sample_actions(t_ms=30.0)
    agent.step(np.zeros(2), np.ones(2) * 2.5, phase="stance", t_ms=30.0)
    assert agent.params[key].sigma != s0             # applied from warmup+1


def test_new_site_after_construction_is_picked_up(nest_kernel):
    agent, reg = _agent()
    reg.create_population(Population("izhikevich", 3,
                          {"a": PersonaParam(mean=0.5, sigma=0.1)}, "late"))
    agent.sample_actions(t_ms=0.0)
    assert "pop_late_a" in agent.current_actions
    agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert "pop_late_a" in agent.history["means"]
    assert agent.strategy.n_sites == 2


def test_removed_site_is_loud(nest_kernel):
    agent, reg = _agent()
    del reg.population_cfgs["pop"]
    try:
        agent.sample_actions(t_ms=0.0)
    except RuntimeError as e:
        assert "disappeared" in str(e)
    else:
        raise AssertionError("a vanished site must raise, not silently drop")


def test_influence_logged(nest_kernel):
    agent, reg = _agent()
    agent.sample_actions(t_ms=0.0)
    agent.step(np.zeros(2), np.ones(2), phase=["stance", "swing"], t_ms=10.0)
    assert "influence" in agent.history and len(agent.history["influence"]) == 1
    assert agent.history["influence"][-1].shape == (2 * agent.strategy.n_sites,)


def test_recorders_created_later_get_the_agent_beta(nest_kernel):
    agent, reg = _agent(tau_mem_s=0.2)           # beta = 1 − 0.01/0.2 = 0.95
    rec = reg.producer_recorder("stdp:late")
    assert rec.beta == 0.995                     # NoiseRecorder default until seen
    agent.sample_actions(t_ms=0.0)
    assert rec.beta == agent.beta
