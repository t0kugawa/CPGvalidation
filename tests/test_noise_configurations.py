import pytest

from persona.noise_recorder import select_producer_names

NAMES = [
    "producer:poisson:cut_L",
    "producer:poisson:ia_e_L",
    "producer:stdp:cut_rge_L",
    "producer:stdp:ia_rge_L",
]


def test_select_producer_names_all_none_poisson_stdp():
    assert select_producer_names(NAMES, "all") == NAMES
    assert select_producer_names(NAMES, "none") == []
    assert select_producer_names(NAMES, "poisson") == NAMES[:2]
    assert select_producer_names(NAMES, "stdp") == NAMES[2:]


def test_select_producer_names_one_relay_accepts_prefix_or_plain():
    assert select_producer_names(NAMES, "one-relay:cut_L") == [NAMES[0]]
    assert select_producer_names(
        NAMES, "one-relay:producer:poisson:cut_L") == [NAMES[0]]


def test_select_producer_names_rejects_unknown():
    with pytest.raises(ValueError):
        select_producer_names(NAMES, "banana")


class _FakeRegistry:
    def __init__(self):
        from persona.parameter import PersonaParam
        from persona.noise_recorder import NoiseRecorder

        self.population_cfgs = {}
        self.connection_cfgs = {}
        self.noise_recorders = {
            "producer:poisson:cut_L": NoiseRecorder(
                "producer:poisson:cut_L", scale=10.0),
            "producer:stdp:x": NoiseRecorder(
                "producer:stdp:x", scale=1.0),
        }
        self._params = {
            "pop_A_a": PersonaParam(mean=1.0, sigma=0.1, min=0.0, max=2.0)
        }

    def get_all_optimizable_params(self):
        return dict(self._params)

    def producer_recorders(self):
        return {n: r for n, r in self.noise_recorders.items()
                if n.startswith("producer:")}


def _agent(noise_config):
    pytest.importorskip("nest")
    from persona.agent import Agent

    return Agent(registry=_FakeRegistry(), noise_config=noise_config)


def test_agent_noise_filter_changes_strategy_dimensions():
    agent = _agent("none")
    assert agent.strategy.noise_names == []
    assert agent.strategy.n_total == 2
    agent = _agent("poisson")
    assert agent.strategy.noise_names == ["producer:poisson:cut_L"]
    assert agent.strategy.n_total == 3


def test_agent_reads_a_producer_delta_exactly_once():
    import numpy as np

    agent = _agent("poisson")
    rec = agent.registry.noise_recorders["producer:poisson:cut_L"]
    rec.record_delta(0.0, 0.5)
    captured = []
    original = agent.strategy.step

    def spy(sites, noise, advantage, **kwargs):
        captured.append(dict(noise))
        return original(sites, noise, advantage, **kwargs)

    agent.strategy.step = spy
    for _ in range(2):
        agent.step(output=np.array([0.0, 0.0]), target=np.array([1.0, 1.0]),
                   phase="stance")
    assert captured[0]["producer:poisson:cut_L"] == (0.5, 10.0)
    assert captured[1]["producer:poisson:cut_L"] == (0.0, 10.0)
    assert all(np.all(agent.strategy.G[2 * agent.strategy.n_sites:, :] == 0.0)
               for _ in [0])
