import numpy as np

from persona.agent import Agent
from persona.connection import Connection
from persona.game import GameConfig
from persona.parameter import PersonaParam
from persona.population import Population
from persona.strategy import FisherMatrix


class FakeRegistry:
    def __init__(self, populations, connections, counts=None):
        self.population_cfgs = populations
        self.connection_cfgs = connections
        self.population_nodes = {name: object() for name in populations}
        self.noise_recorders = {}
        self._syn_counts = dict(counts or {})

    def get_all_optimizable_params(self):
        out = {}
        for pop_name, pop in self.population_cfgs.items():
            for p_name, param in pop.params.items():
                out[f"pop_{pop_name}_{p_name}"] = param
        for conn_key, conn in self.connection_cfgs.items():
            for p_name, param in conn.parameters.items():
                out[f"conn_{conn_key}_{p_name}"] = param
        return out

    def producer_recorders(self):
        return {}

    def synapse_count(self, key):
        return int(self._syn_counts.get(key, 0))


def _registry():
    pop = Population("izhikevich", 4,
                     {"a": PersonaParam(mean=1.0, sigma=0.2,
                                        min=0.0, max=3.0)}, "A")
    pop_b = Population("izhikevich", 3,
                       {"b": PersonaParam(mean=0.2, sigma=0.05,
                                          min=0.0, max=1.0)}, "B")
    conn = Connection(
        "A", "B", model="static_synapse",
        parameters={"weight": PersonaParam(mean=0.5, sigma=0.1,
                                           min=0.0, max=1.0)},
        key="A_B")
    return FakeRegistry({"A": pop, "B": pop_b}, {"A_B": conn},
                        counts={"A_B": 12})


def _run(agent, steps=12, seed=4):
    rng = np.random.default_rng(seed)
    for t in range(steps):
        agent.sample_actions(t_ms=t * 10.0)
        target = np.array([0.5 + 0.1 * np.sin(t / 3.0),
                           0.5 + 0.1 * np.cos(t / 3.0)])
        output = np.array([0.4, 0.6]) + rng.normal(0.0, 0.02, 2)
        agent.step(output=output, target=target, phase="stance",
                   t_ms=(t + 1) * 10.0)


def test_agent_default_has_no_game():
    agent = Agent(registry=_registry(), lr_mu=0.03, dt_ms=10.0)
    assert agent.game is None
    assert "game" not in agent.history
    assert agent.strategy.game is None


def test_disabled_game_config_is_not_built():
    agent = Agent(registry=_registry(), lr_mu=0.03, dt_ms=10.0,
                  game=GameConfig(enabled=False))
    assert agent.game is None
    assert "game" not in agent.history


def test_agent_enabled_builds_game_and_logs_diagnostics():
    agent = Agent(registry=_registry(), lr_mu=0.03, dt_ms=10.0,
                  warmup=0, game=GameConfig(enabled=True, burnin_steps=0))
    assert agent.game is not None
    assert agent.strategy.game is agent.game
    assert len(agent.game.stacks) == 3
    _run(agent, steps=12)
    assert len(agent.history["game"]) == 12
    last = agent.history["game"][-1]
    assert np.isclose(last["stacks"].sum(), 1.0)
    for key in ("throttle", "contribution", "entropy", "pot",
                "transfer_sum"):
        assert key in last


def test_existing_strategy_gets_the_game():
    strat = FisherMatrix(
        param_names=["pop_A_a", "pop_B_b", "conn_A_B_weight"],
        lr_mu=0.03, warmup=0)
    agent = Agent(registry=_registry(), strategy=strat,
                  game=GameConfig(enabled=True, burnin_steps=0))
    assert agent.strategy is strat
    assert strat.game is agent.game
