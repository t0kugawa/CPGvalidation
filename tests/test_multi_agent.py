"""Multi-replica agent wiring without NEST/MuJoCo.

The fake registries expose the small surface MultiAgent needs. The test covers
different samples, shared-policy sync, per-replica strategy accumulation, the
phase-boundary merge, and advantage weighting.
"""
import numpy as np

from persona.multi_agent import MultiAgent
from persona.parameter import PersonaParam, reset_rng_stream

SITES = ("a", "b", "c")


class FakeRegistry:
    def __init__(self, params):
        self.params = params
        self.applied = None
        self.last_apply_errors = []

    def get_all_optimizable_params(self):
        return self.params

    def producer_recorders(self):
        return {}

    def frozen_attrs(self, name):
        return frozenset()

    def springs(self):
        return {}

    def spring_state(self, means, sigmas, lr_t):
        return {}

    def apply_explored_parameters(self, actions):
        self.applied = dict(actions)


def _registries(seed: int, replicas: int = 3):
    reset_rng_stream(seed)
    return [
        FakeRegistry({
            name: PersonaParam(mean=0.0, sigma=0.2, min=-5.0, max=5.0)
            for name in SITES
        })
        for _ in range(replicas)
    ]


def _agent(seed: int = 0, replicas: int = 3) -> MultiAgent:
    return MultiAgent(_registries(seed, replicas), lr_mu=0.1, lr_sigma=0.01,
                      warmup=0, dt_ms=10.0)


def _step(agent, phase, value=0.5):
    n = agent.n_replicas
    outputs = [np.array([value, -value]) for _ in range(n)]
    targets = [np.zeros(2) for _ in range(n)]
    return agent.step(outputs, targets, [phase for _ in range(n)])


def test_replicas_sample_different_actions():
    agent = _agent()
    actions = agent.sample_actions()
    assert len(actions) == agent.n_replicas
    for site in SITES:
        assert len({actions[i][site] for i in range(agent.n_replicas)}) > 1


def test_sampling_is_reproducible_from_the_base_seed():
    first = _agent(seed=5).sample_actions()
    second = _agent(seed=5).sample_actions()
    assert first == second


def test_apply_actions_reaches_every_replica():
    agent = _agent()
    actions = agent.sample_actions()
    agent.apply_actions(actions)
    for registry, expected in zip(agent.registries, actions):
        assert registry.applied == expected


def test_first_step_accumulates_without_merging():
    agent = _agent()
    agent.sample_actions()
    result = _step(agent, ("stance", "stance"))
    assert result["merged"] is False
    assert agent.strategy.steps == 0
    assert all(s.steps == 1 for s in agent.replica_strategies)
    assert np.any(agent.replica_strategies[0].F != 0.0)


def test_phase_change_merges_and_updates_the_shared_policy_once():
    agent = _agent()
    agent.sample_actions()
    _step(agent, ("stance", "stance"))
    before = {name: (p.mean, p.sigma) for name, p in agent.params.items()}

    agent.sample_actions()
    result = _step(agent, ("swing", "swing"))

    assert result["merged"] is True
    assert agent.strategy.steps == 2
    assert all(s.steps == 2 for s in agent.replica_strategies)
    assert np.allclose(agent.replica_strategies[0].F, agent.strategy.F)
    assert any(
        agent.params[name].mean != before[name][0] for name in SITES
    )
    # The policy mean is shared by every replica.
    for registry in agent.registries[1:]:
        for name in SITES:
            assert registry.params[name].mean == agent.params[name].mean


def test_merge_weights_favour_the_better_advantage():
    agent = _agent()
    agent.replica_adv = np.array([1.0, 0.0, -1.0])
    weights = agent._merge_weights()
    assert weights[0] > weights[1] > weights[2]
    assert np.isclose(weights.sum(), 1.0)


def test_merge_copies_state_back_to_every_replica():
    agent = _agent()
    agent.sample_actions()
    _step(agent, ("stance", "stance"))
    agent.sample_actions()
    _step(agent, ("swing", "swing"))

    for replica in agent.replica_strategies:
        assert np.allclose(replica.F, agent.strategy.F)
        assert np.allclose(replica.Xbar, agent.strategy.Xbar)
        assert np.allclose(replica.G, agent.strategy.G)
        assert replica.steps == agent.strategy.steps
