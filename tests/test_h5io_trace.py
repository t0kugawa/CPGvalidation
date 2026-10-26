import numpy as np

from persona.h5io import load_run, save_run
from persona.personalization import RunLog


class _Agent:
    history = {
        "means": {"p": [0.0, 0.1]},
        "sigmas": {"p": [0.2, 0.2]},
        "strategy_trace": {
            "dtheta": np.zeros((2, 1)),
            "sigma": np.ones((2, 1)),
            "dlogsigma": np.zeros((2, 1)),
            "advantage": np.zeros((2, 2)),
            "noise_delta": np.zeros((2, 0)),
            "noise_scale": np.zeros((2, 0)),
            "param_names": ["p"],
            "noise_names": [],
            "beta": 0.995,
            "lambda_reg": 1e-3,
            "warmup": 3,
            "refresh_every": 10,
            "dtype": "float64",
        },
        "strategy_checkpoints": [
            {"steps": 2, "F": np.eye(2), "Xbar": np.zeros(2),
             "G": np.zeros((2, 2)), "P": np.eye(2),
             "_P_valid": True, "_steps_since_refresh": 2},
        ],
    }


def test_save_load_trace_and_checkpoints(tmp_path):
    path = tmp_path / "run.h5"
    save_run(str(path), RunLog(), _Agent())
    data, _ = load_run(str(path))
    trace = data["agent"]["strategy_trace"]
    assert trace["beta"] == 0.995
    assert trace["param_names"] == ["p"]
    assert np.array_equal(trace["sigma"], np.ones((2, 1)))
    checkpoint = data["agent"]["strategy_checkpoints"][0]
    assert checkpoint["steps"] == 2
    assert np.array_equal(checkpoint["F"], np.eye(2))


def test_missing_trace_groups_do_not_crash(tmp_path):
    class _Plain:
        history = {"means": {}, "sigmas": {}}

    path = tmp_path / "plain.h5"
    save_run(str(path), RunLog(), _Plain())
    data, _ = load_run(str(path))
    assert "strategy_trace" not in data["agent"]
    assert "strategy_checkpoints" not in data["agent"]
