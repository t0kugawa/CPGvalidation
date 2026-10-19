import numpy as np

from persona.registry import Registry
from t2xcpg.leg import Leg


def test_sensory_deltas_are_read_and_clear(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    leg.update_sensory_feedback(cut_hz=12.0, ia_e_hz=20.0, ia_f_hz=20.0,
                                ib_e_hz=8.0, ib_f_hz=8.0, current_time_ms=10.0)
    cut = reg.producer_recorder("poisson:cut_L")
    ia_e = reg.producer_recorder("poisson:ia_e_L")
    assert np.isclose(cut.take_delta(), 12.0)          # 0 -> 12 Hz
    assert np.isclose(ia_e.take_delta(), 10.0)         # 10 -> 20 Hz
    # a repeated rate in the next step is a fresh zero, never a stale 12
    leg.update_sensory_feedback(cut_hz=12.0, ia_e_hz=20.0, ia_f_hz=20.0,
                                ib_e_hz=8.0, ib_f_hz=8.0, current_time_ms=20.0)
    assert cut.take_delta() == 0.0
    assert cut.take_delta() is None


def test_left_and_right_groups_do_not_share_recorders(nest_kernel):
    reg = Registry()
    leg_l = Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    leg_r = Leg("R", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    leg_l.update_sensory_feedback(5.0, 0.0, 0.0, 0.0, 0.0, current_time_ms=10.0)
    leg_r.update_sensory_feedback(25.0, 0.0, 0.0, 0.0, 0.0, current_time_ms=10.0)
    assert np.isclose(reg.producer_recorder("poisson:cut_L").take_delta(), 5.0)
    assert np.isclose(reg.producer_recorder("poisson:cut_R").take_delta(), 25.0)
