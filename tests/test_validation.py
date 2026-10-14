import matplotlib
matplotlib.use("Agg")

import numpy as np

from persona.personalization import RunLog
from persona.validation import ValidationReport, biological_acceptance


def synthetic_runlog(n=100):
    t = np.arange(n) * 10.0
    log = RunLog()
    log.time_ms = t
    log.ankle_model_L = 0.3 + 0.1 * np.sin(t / 100.0)
    log.ankle_model_R = 0.3 - 0.1 * np.sin(t / 100.0)
    log.ankle_target_L = 0.3 + 0.1 * np.sin(t / 100.0)
    log.ankle_target_R = 0.3 - 0.1 * np.sin(t / 100.0)
    log.loss = np.linspace(0.5, 0.05, n)
    log.advantage = np.linspace(-0.2, 0.1, n)
    log.phase_L = ["stance"] * n
    log.phase_R = ["swing"] * n
    log.stage = ["pose"] * (n // 2) + ["track"] * (n - n // 2)
    return log


def test_figures_written(tmp_path):
    log = synthetic_runlog(n=100)
    rep = ValidationReport()
    rep.plot_tracking(log)
    rep.plot_learning(log)
    rep.save_all(tmp_path)
    assert (tmp_path / "tracking.png").exists()
    assert (tmp_path / "learning.png").exists()


def test_angle_acceptance_rule():
    err = np.array([0.01] * 70 + [0.3] * 30)
    res = biological_acceptance({}, stats=None, angle_errors=err,
                                phase_mask=np.ones(100, bool),
                                margin=0.05, coverage=0.70)
    assert np.isclose(res["angle_coverage"], 0.70)
    assert res["acceptable"] is True


def test_angle_acceptance_fails_below_coverage():
    err = np.array([0.3] * 100)
    res = biological_acceptance({}, stats=None, angle_errors=err,
                                phase_mask=np.ones(100, bool),
                                margin=0.05, coverage=0.70)
    assert res["acceptable"] is False