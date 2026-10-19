"""Afferent models (CUT / Ia / Ib) must actually report the stance phase.

The MVP's only muscle-driven joint is the ankle, so the spindles and the Golgi
tendon organ are the CPG's entire window on load and length. A spindle that
reads a constant rate cannot signal stance, which leaves the ankle with no
counterforce while the belt drags it.
"""
import numpy as np

from model.adapter import CPGOutput
from model.cut import IaAfferent, IbAfferent
from model.leg import Leg


def test_ia_rate_rises_with_tendon_load():
    ia = IaAfferent(l_0=0.014)
    unloaded = ia.get_ia_hz(0.014, 0.0, force=0.0)
    loaded = ia.get_ia_hz(0.014, 0.0, force=4.0)
    assert loaded > unloaded + 20.0


def test_ia_stretch_term_is_alive_over_the_real_tendon_range():
    ia = IaAfferent(l_0=0.014)
    slack = ia.get_ia_hz(0.012, 0.0)          # shorter than the reference
    rest = ia.get_ia_hz(0.014, 0.0)
    stretched = ia.get_ia_hz(0.020, 0.0)      # longer than the reference
    assert np.isclose(slack, rest)            # no stretch below the reference
    assert stretched > rest + 5.0


def test_ia_reference_length_matches_the_rig_geometry():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    for s in ("L", "R"):
        # A reference far outside the tendon's real excursion (e.g. 40 mm for a
        # ~3-20 mm tendon) kills the stretch term; guard against that.
        assert 0.002 < leg.ia_sol[s].l_0 < 0.025
        assert 0.001 < leg.ia_ta[s].l_0 < 0.025


def test_ib_rate_rises_with_load_and_activation():
    ib = IbAfferent()
    quiet = ib.get_ib_hz(0.0, 0.0)
    loaded = ib.get_ib_hz(5.0, 0.9)
    assert loaded > quiet + 50.0


def test_soleus_ia_reports_load_when_the_muscle_contracts():
    """Driving SOL must raise the Ia rate well above its unloaded baseline."""
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)

    def run(drive, n=60):
        leg.reset()
        sen = None
        for _ in range(n):
            out = CPGOutput(mus_E_L=drive, mus_F_L=0.0, mus_E_R=0.0,
                            mus_F_R=0.0, t_ms=leg.time_ms)
            sen = leg.step(out, target_ankle_L=0.3, target_ankle_R=0.3,
                           target_hip_L=-0.2, target_knee_L=0.6,
                           target_hip_R=-0.2, target_knee_R=0.6)
        return sen

    off = run(0.0)
    on = run(1.0)
    assert on.ib_force_E_L > 1.0, "SOL must build force"
    assert on.ia_rate_E_L > off.ia_rate_E_L + 10.0
