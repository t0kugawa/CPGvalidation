import numpy as np
import pytest

from persona.strategy import FisherMatrix
from persona.strategy_trace import rebuild_strategy_from_trace


def _trace(T=20, S=3, N=2):
    rng = np.random.default_rng(0)
    return {
        "dtheta": rng.normal(0, 0.1, (T, S)),
        "sigma": rng.uniform(0.2, 0.8, (T, S)),
        "dlogsigma": rng.normal(0, 0.01, (T, S)),
        "advantage": rng.normal(0, 0.3, (T, 2)),
        "noise_delta": rng.normal(0, 0.05, (T, N)),
        "noise_scale": np.ones((T, N)),
        "param_names": [f"p{i}" for i in range(S)],
        "noise_names": [f"n{i}" for i in range(N)],
        "beta": 0.9,
        "lambda_reg": 0.5,
        "warmup": 0,
        "refresh_every": 10,
        "dtype": "float64",
    }


def test_rebuild_from_trace_is_exact():
    trace = _trace()
    fm = FisherMatrix(param_names=trace["param_names"],
                      noise_names=trace["noise_names"], beta=0.9,
                      lambda_reg=0.5, warmup=0, refresh_every=10)
    for t in range(len(trace["dtheta"])):
        sites = {name: (trace["dtheta"][t, i], trace["sigma"][t, i],
                        trace["dlogsigma"][t, i])
                 for i, name in enumerate(trace["param_names"])}
        noise = {name: (trace["noise_delta"][t, j],
                        trace["noise_scale"][t, j])
                 for j, name in enumerate(trace["noise_names"])}
        fm.step(sites, noise, trace["advantage"][t], accumulate_only=True)
    fm._refresh_inverse()

    rebuilt = rebuild_strategy_from_trace(trace)
    assert np.array_equal(rebuilt.F, fm.F)
    assert np.array_equal(rebuilt.Xbar, fm.Xbar)
    assert np.array_equal(rebuilt.G, fm.G)


class _FakeRegistry:
    def __init__(self):
        from persona.parameter import PersonaParam

        self.population_cfgs = {}
        self.connection_cfgs = {}
        self.noise_recorders = {}
        self._params = {
            "pop_A_a": PersonaParam(mean=1.0, sigma=0.1, min=0.0, max=2.0)
        }

    def get_all_optimizable_params(self):
        return dict(self._params)

    def producer_recorders(self):
        return {}


def _run_agent(agent, steps=6):
    for t in range(steps):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(output=np.array([0.2, 0.1]),
                   target=np.array([1.0, 1.0]), phase="stance",
                   t_ms=(t + 1) * 10.0)


def test_agent_trace_rebuilds_the_strategy_exactly():
    pytest.importorskip("nest")
    from persona.agent import Agent

    agent = Agent(registry=_FakeRegistry(), trace_strategy=True)
    _run_agent(agent)
    trace = agent.history["strategy_trace"]
    assert trace["dtype"] == "float64"
    rebuilt = rebuild_strategy_from_trace(trace)
    assert np.array_equal(rebuilt.F, agent.strategy.F)
    assert np.array_equal(rebuilt.Xbar, agent.strategy.Xbar)
    assert np.array_equal(rebuilt.G, agent.strategy.G)


def test_trace_rejects_a_site_added_mid_run():
    pytest.importorskip("nest")
    from persona.agent import Agent

    registry = _FakeRegistry()
    agent = Agent(registry=registry, trace_strategy=True)
    _run_agent(agent, steps=2)
    from persona.parameter import PersonaParam
    registry._params["pop_B_b"] = PersonaParam(
        mean=0.5, sigma=0.1, min=0.0, max=1.0)
    with pytest.raises(RuntimeError, match="pop_B_b"):
        agent.refresh_params()


def test_float32_trace_is_approximate_only():
    pytest.importorskip("nest")
    from persona.agent import Agent

    agent = Agent(registry=_FakeRegistry(), trace_strategy=True,
                  trace_float32=True)
    _run_agent(agent)
    trace = agent.history["strategy_trace"]
    assert trace["dtype"] == "float32"
    rebuilt = rebuild_strategy_from_trace(trace)
    assert np.allclose(rebuilt.F, agent.strategy.F, rtol=1e-3, atol=1e-6)
