import numpy as np
import pytest

from persona.strategy import FisherMatrix


def _fm(**kwargs):
    options = dict(param_names=["a", "b"], noise_names=["n"],
                   beta=0.9, lambda_reg=0.5, warmup=0, refresh_every=10)
    options.update(kwargs)
    return FisherMatrix(**options)


def test_snapshot_restores_mid_refresh_exactly():
    fm = _fm()
    for _ in range(3):
        fm.step({"a": (0.1, 0.5, 0.0), "b": (-0.2, 0.4, 0.01)},
                {"n": (0.05, 1.0)}, np.array([0.3, -0.1]))
    snapshot = fm.snapshot()
    clone = FisherMatrix.from_snapshot(snapshot)

    assert np.array_equal(clone.F, fm.F)
    assert np.array_equal(clone.Xbar, fm.Xbar)
    assert np.array_equal(clone.G, fm.G)
    assert np.array_equal(clone.P, fm.P)
    assert clone._steps_since_refresh == fm._steps_since_refresh
    assert clone.steps == fm.steps

    sites = {"a": (0.05, 0.5, 0.0), "b": (0.03, 0.4, 0.0)}
    noise = {"n": (0.01, 1.0)}
    advantage = np.array([0.2, 0.1])
    left = fm.step(sites, noise, advantage)
    right = clone.step(sites, noise, advantage)
    assert left.keys() == right.keys()
    for key in left:
        assert np.isclose(left[key], right[key])


def test_snapshot_is_self_contained_and_not_a_full_agent_resume():
    fm = _fm()
    fm.step({"a": (0.1, 0.5, 0.0), "b": (-0.2, 0.4, 0.0)},
            {"n": (0.05, 1.0)}, np.array([0.3, -0.1]))
    snapshot = fm.snapshot()
    for key in ("F", "Xbar", "G", "P", "steps", "_steps_since_refresh",
                "beta", "lambda_reg", "warmup", "refresh_every", "lr_mu",
                "dt_s", "param_names", "noise_names", "noise_scales"):
        assert key in snapshot
    assert snapshot["homeostasis"] is None
    assert snapshot["game"] is None
    clone = FisherMatrix.from_snapshot(snapshot)
    assert clone.param_names == fm.param_names
    assert clone.noise_names == fm.noise_names


class _SnapRegistry:
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


def test_agent_snapshots_every_n_steps():
    pytest.importorskip("nest")
    from persona.agent import Agent

    agent = Agent(registry=_SnapRegistry(), snapshot_strategy=2)
    for t in range(5):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(output=np.array([0.2, 0.1]),
                   target=np.array([1.0, 1.0]), phase="stance",
                   t_ms=(t + 1) * 10.0)
    steps = [s["steps"] for s in agent.history["strategy_checkpoints"]]
    assert steps == [2, 4]
