import pytest

from persona.game import GameConfig
from persona.homeostasis import HomeostaticConfig
from persona.partition import PlayerPartition
from persona.parameter import PersonaParam
from persona.population import Population


class _Registry:
    def __init__(self):
        self.population_cfgs = {
            "A": Population("izhikevich", 4,
                            {"a": PersonaParam(1.0, 0.1, 0.0, 2.0)}, "A"),
            "B": Population("izhikevich", 4,
                            {"b": PersonaParam(0.2, 0.05, 0.0, 1.0)}, "B"),
        }
        self.connection_cfgs = {}
        self.noise_recorders = {}

    def get_all_optimizable_params(self):
        out = {}
        for name, pop in self.population_cfgs.items():
            for param in pop.params:
                out[f"pop_{name}_{param}"] = pop.params[param]
        return out

    def producer_recorders(self):
        return {}

    def relay_links(self):
        return {}


def _partition():
    return PlayerPartition(((0,), (1,)))


def test_agent_uses_the_player_partition():
    pytest.importorskip("nest")
    from persona.agent import Agent

    registry = _Registry()
    agent = Agent(registry=registry, game=GameConfig.current_best(),
                  player_map=_partition())
    assert agent.game.n_players == 2
    assert set(agent.game._site_names) == set(agent.strategy.param_names)
    assert agent.game.physical_map.n_players == 2


def test_agent_rejects_evo_with_homeostasis():
    pytest.importorskip("nest")
    from persona.agent import Agent

    with pytest.raises(ValueError, match="mutually exclusive"):
        Agent(registry=_Registry(), game=GameConfig.current_best(),
              homeostasis=HomeostaticConfig(enabled=True),
              player_map=_partition())
