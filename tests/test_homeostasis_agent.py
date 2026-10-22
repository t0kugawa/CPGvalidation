from types import SimpleNamespace

import numpy as np

from persona.agent import Agent
from persona.connection import Connection
from persona.homeostasis import HomeostaticConfig
from persona.parameter import PersonaParam
from persona.population import Population


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


def _registry(weight_bounds=(0.0, 1.0), weight_mean=0.5, conn_model="static_synapse"):
    pop = Population("izhikevich", 4,
                     {"a": PersonaParam(mean=1.0, sigma=0.2,
                                        min=0.0, max=3.0)},
                     "A")
    pop_b = Population("izhikevich", 3,
                       {"b": PersonaParam(mean=0.2, sigma=0.05,
                                          min=0.0, max=1.0)},
                       "B")
    conn = Connection(
        "A", "B", model=conn_model,
        parameters={"weight": PersonaParam(
            mean=weight_mean, sigma=0.1,
            min=weight_bounds[0], max=weight_bounds[1])},
        key="A_B")
    return FakeRegistry({"A": pop, "B": pop_b}, {"A_B": conn},
                        counts={"A_B": 12})


def _run(agent, steps=12, seed=4):
    rng = np.random.default_rng(seed)
    for t in range(steps):
        agent.sample_actions(t_ms=t * 10.0)
        # A moving target keeps the plant from settling at one exact point.
        target = np.array([0.5 + 0.1 * np.sin(t / 3.0),
                           0.5 + 0.1 * np.cos(t / 3.0)])
        output = np.array([0.4, 0.6]) + rng.normal(0.0, 0.02, 2)
        agent.step(output=output, target=target, phase="stance",
                   t_ms=(t + 1) * 10.0)


def test_agent_default_disables_homeostasis_history():
    agent = Agent(registry=_registry(), lr_mu=0.03, dt_ms=10.0)
    assert agent.homeostasis is None
    assert "homeostasis" not in agent.history
    assert agent.strategy.homeostasis is None


def test_agent_enabled_builds_map_and_logs_diagnostics():
    reg = _registry()
    cfg = HomeostaticConfig(enabled=True, h_burnin_steps=0)
    agent = Agent(registry=reg, lr_mu=0.03, dt_ms=10.0,
                  warmup=0, homeostasis=cfg)
    assert agent.homeostasis is not None
    assert agent.strategy.homeostasis is agent.homeostasis
    assert "homeostasis" in agent.history
    _run(agent, steps=4)
    assert len(agent.history["homeostasis"]) == 4
    last = agent.history["homeostasis"][-1]
    assert last["stage"] == "active"
    assert last["mode"] == "advantage"
    assert "c" in last and "u_H" in last and "z" in last
    assert last["player_names"]


def test_agent_disabled_config_is_the_same_as_none():
    for cfg in (None, HomeostaticConfig(enabled=False)):
        agent = Agent(registry=_registry(), lr_mu=0.03, dt_ms=10.0,
                      homeostasis=cfg)
        assert agent.homeostasis is None
        assert "homeostasis" not in agent.history


def test_agent_warmup_applies_no_updates_with_homeostasis_enabled():
    agent = Agent(registry=_registry(), lr_mu=1.0, dt_ms=10.0, warmup=3,
                  homeostasis=HomeostaticConfig(enabled=True,
                                                h_burnin_steps=0))
    before = {k: (p.mean, p.sigma) for k, p in agent.params.items()}
    _run(agent, steps=3)
    for key, (mean, sigma) in before.items():
        assert agent.params[key].mean == mean
        assert agent.params[key].sigma == sigma


def test_agent_hard_bounds_hold_for_excitatory_and_inhibitory_weights():
    for bounds, mean in (((0.0, 1.0), 0.5), ((-1.0, 0.0), -0.5)):
        agent = Agent(registry=_registry(weight_bounds=bounds, weight_mean=mean),
                      lr_mu=1.0, dt_ms=10.0, warmup=0,
                      homeostasis=HomeostaticConfig(
                          enabled=True, h_burnin_steps=0, lambda_H=1000.0))
        _run(agent, steps=25, seed=5)
        for name, param in agent.params.items():
            assert param.min <= param.mean <= param.max
            if "weight" in name:
                if bounds[0] < 0.0:
                    assert param.mean <= 0.0
                else:
                    assert param.mean >= 0.0


def test_agent_stdp_hyperparameters_stay_positive_and_in_bounds():
    pop = Population("izhikevich", 4,
                     {"a": PersonaParam(mean=1.0, sigma=0.2,
                                        min=0.0, max=3.0)}, "A")
    conn = Connection(
        "A", "B", model="stdp_synapse",
        parameters={"Wmax": PersonaParam(mean=100.0, sigma=10.0,
                                         min=1.0, max=1000.0)},
        key="A_B_stdp", weight_recorder=object())
    reg = FakeRegistry({"A": pop}, {"A_B_stdp": conn},
                       counts={"A_B_stdp": 5})
    agent = Agent(registry=reg, lr_mu=1.0, dt_ms=10.0, warmup=0,
                  homeostasis=HomeostaticConfig(
                      enabled=True, h_burnin_steps=0, lambda_H=1000.0))
    _run(agent, steps=20, seed=7)
    wmax = agent.params["conn_A_B_stdp_Wmax"]
    assert 1.0 <= wmax.mean <= 1000.0
    ptype = agent.homeostasis.physical_map.specs[
        agent.homeostasis.physical_map.player_of["conn_A_B_stdp_Wmax"]].ptype
    assert ptype == "stdp"
