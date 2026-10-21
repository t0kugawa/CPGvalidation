"""Multi-site plantar CUT: contact -> site, active buffer, aggregate, phase.

The tests use synthetic contact points and site positions, so they run without
NEST. The last test checks the XML site discovery on the real MuJoCo model.
"""
import numpy as np
import pytest

from model.adapter import CPGInput
from model.cut import (CUTAfferent, ContactPhase, PlantarCut,
                       aggregate_cut_hz)


def test_contact_assigns_to_nearest_site_and_sums_force():
    sites = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0]])
    cut = PlantarCut(("heel", "mid", "toe"), dt_ms=10.0, tau_ms=1.0,
                     active_force_n=0.01)
    out = cut.update(
        contact_positions=np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        contact_forces=np.array([0.2, 0.3]),
        site_positions=sites,
    )
    assert np.allclose(out["forces"], [0.2, 0.3, 0.0])
    assert out["active"].tolist() == [True, True, False]
    assert out["rates"][0] > 0.0 and out["rates"][1] > 0.0
    assert out["rates"][2] == 0.0


def test_inactive_site_is_zero_and_state_is_reset():
    cut = PlantarCut(("heel",), dt_ms=10.0, tau_ms=1.0,
                     active_force_n=0.01, r_peak_hz=120.0)
    sites = np.zeros((1, 3))
    low = cut.update(np.array([[0.0, 0.0, 0.0]]), np.array([0.005]), sites)
    assert low["active"].tolist() == [False]
    assert low["rates"].tolist() == [0.0]
    assert low["aggregate_hz"] == 0.0

    high = cut.update(np.array([[0.0, 0.0, 0.0]]), np.array([0.5]), sites)
    assert high["rates"][0] > 0.0
    # A site that unloads must start clean, not decay a stale rate.
    gone = cut.update(np.empty((0, 3)), np.empty(0), sites)
    assert gone["rates"].tolist() == [0.0]
    assert gone["active"].tolist() == [False]


def test_aggregate_is_area_density_weighted_mean_of_active():
    rates = np.array([100.0, 20.0, 5.0])
    areas = np.array([1.0, 3.0, 1.0])
    density = np.array([1.0, 2.0, 1.0])
    active = np.array([True, True, False])
    # (100*1*1 + 20*3*2) / (1*1 + 3*2)
    assert aggregate_cut_hz(rates, areas, density, active) == pytest.approx(220 / 7)
    assert aggregate_cut_hz(rates, areas, density, np.zeros(3, bool)) == 0.0
    assert aggregate_cut_hz(rates, areas, density,
                            np.array([True, False, False])) == pytest.approx(100.0)


def test_aggregate_stays_in_cut_scale():
    for _ in range(20):
        n = 8
        rates = np.random.default_rng(0).uniform(0, 120, n)
        areas = np.random.default_rng(1).uniform(0.2, 2.0, n)
        density = np.random.default_rng(2).uniform(0.2, 2.0, n)
        active = np.random.default_rng(3).random(n) > 0.3
        out = aggregate_cut_hz(rates, areas, density, active)
        assert 0.0 <= out <= 120.0


def test_cut_afferent_area_and_density_scale_rate():
    site = CUTAfferent(dt_ms=2.0, tau_ms=20.0)
    r1 = site.get_cut_hz(0.5, area=1.0, density=1.0)
    site.reset()
    r2 = site.get_cut_hz(0.5, area=2.0, density=1.0)
    site.reset()
    r3 = site.get_cut_hz(0.5, area=1.0, density=2.0)
    assert r2 > r1
    assert r3 > r1
    assert max(r2, r3) <= 120.0


def test_phase_default_debounce_is_50ms():
    # The 4 s probe showed 20 ms still chattered at 5-7 transitions/s; 50 ms
    # is the smallest setting that brought both legs into the 2-4/s band
    # without changing the stance fraction.
    ph = ContactPhase(initial="stance")
    for _ in range(4):
        assert ph.update(0.004, 10.0) == "stance"
    assert ph.update(0.004, 10.0) == "swing"


def test_phase_debounces_contact_and_uses_hysteresis():
    ph = ContactPhase(on_n=0.010, off_n=0.005, debounce_ms=20.0,
                      initial="stance")
    # 10 ms of no contact is not enough to leave stance.
    assert ph.update(0.004, 10.0) == "stance"
    assert ph.update(0.004, 10.0) == "swing"
    # Between thresholds while swinging: stay swinging.
    assert ph.update(0.006, 10.0) == "swing"
    assert ph.update(0.006, 10.0) == "swing"
    # A contact pulse shorter than the debounce must not switch.
    assert ph.update(0.015, 10.0) == "swing"
    assert ph.update(0.015, 10.0) == "stance"
    # A short loss of contact does not immediately leave stance.
    assert ph.update(0.004, 10.0) == "stance"
    assert ph.update(0.004, 10.0) == "swing"


def test_cpg_input_carries_per_site_rates_and_active_list():
    cpg_in = CPGInput(
        cut_L=42.0,
        cut_sites_L=(1.0, 2.0),
        cut_active_L=(True, False),
        cut_sites_R=(3.0, 4.0),
        cut_active_R=(False, True),
        cut_site_names_L=("rct_heel_0", "rct_toe_0"),
        cut_site_names_R=("rct_heel_0", "rct_toe_0"),
    )
    assert cpg_in.cut_L == 42.0
    assert cpg_in.cut_sites_L == (1.0, 2.0)
    assert cpg_in.cut_active_R == (False, True)


def test_leg_discovers_rct_sites_from_the_model():
    from model.leg import Leg
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    for side in "LR":
        names = leg.cut_site_names[side]
        assert len(names) == 8
        assert all(n.startswith("rct_") for n in names)
        assert len(set(names)) == 8
        assert leg.cut_plantar[side].n_sites == 8


def test_leg_step_returns_per_site_cut_and_phase():
    from model.leg import Leg
    from model.adapter import CPGOutput
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    out = leg.step(CPGOutput())
    for side in "LR":
        assert len(getattr(out, f"cut_sites_{side}")) == 8
        assert len(getattr(out, f"cut_active_{side}")) == 8
        assert getattr(out, f"cut_site_names_{side}") == leg.cut_site_names[side]
    assert out.phase_L in ("stance", "swing")
    assert out.phase_R in ("stance", "swing")
