import numpy as np

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
