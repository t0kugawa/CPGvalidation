import numpy as np

from loader.stats import StatsLoader, LocomotionMetrics, in_distribution


def test_distribution_reads_all_rats():
    dl = StatsLoader("data/zenodo/DatasetStatisticalAnalysis/StatisticalDataTable.csv")
    d = dl.distribution("LH_duty_factor")
    assert d.size >= 5 and 0.0 < d.mean() < 1.0


def test_distribution_per_rat():
    dl = StatsLoader("data/zenodo/DatasetStatisticalAnalysis/StatisticalDataTable.csv")
    full = dl.distribution("LH_duty_factor")
    ratA = dl.distribution("LH_duty_factor", rat_ids=["RatA"])
    assert 0 < ratA.size < full.size


def test_metrics_from_synthetic_gait():
    n = 400
    dt = 10.0
    t = np.arange(n) * dt
    contact = ((t // 500) % 2 == 0).astype(float)  # 50% duty
    m = LocomotionMetrics.from_run(np.zeros(n), np.zeros(n), contact, contact, dt)
    assert abs(m["duty_factor_L"] - 0.5) < 0.15
    assert abs(m["duty_factor_R"] - 0.5) < 0.15
    assert 0.0 <= m["double_support"] <= 1.0


def test_in_distribution():
    assert in_distribution(0.6, np.array([0.55, 0.6, 0.65]))
    assert not in_distribution(99.0, np.array([0.55, 0.6, 0.65]))