import numpy as np
import pandas as pd
import pytest

from loader.mocap import decode_trial_name, markers_to_angles, treadmill_surfacevel

MARKERS = ["IC", "GT", "KN", "EA", "MH"]
COLLINEAR_COLUMNS = [f"{m}_{s}_{ax}" for s in ("L", "R") for m in MARKERS for ax in ("x", "y", "z")]


def _straight_leg_frame(scale: float = 1.0):
    """Markers laid out along the x-axis (straight leg) => all joint angles 0."""
    d = {}
    for side in ("L", "R"):
        for i, m in enumerate(MARKERS):
            d[f"{m}_{side}_x"] = [i * scale]
            d[f"{m}_{side}_y"] = [0.0]
            d[f"{m}_{side}_z"] = [0.0]
    return d


def test_decode():
    m = decode_trial_name("RatA_x1p75_Rfast_T2.csv")
    assert (m.rat_id, m.speed_ratio, m.fast_belt, m.condition, m.trial) == ("RatA", 1.75, "R", "T", 2)


def test_decode_decimal_d_form():
    m = decode_trial_name("RatC_x1d50_Lfast_W4.csv")
    assert (m.rat_id, m.speed_ratio, m.fast_belt, m.condition, m.trial) == ("RatC", 1.5, "L", "W", 4)


def test_markers_to_angles_straight_leg():
    a = markers_to_angles(pd.DataFrame(_straight_leg_frame()), rest_pose=None)
    assert set(a) == {"hip_L", "knee_L", "ankle_L", "hip_R", "knee_R", "ankle_R"}
    for v in a.values():
        assert np.isfinite(v).all()
        assert np.isclose(v[0], 0.0, atol=1e-9)


def test_rest_pose_zero_alignment():
    a = markers_to_angles(pd.DataFrame(_straight_leg_frame()), rest_pose={"hip_L": 0.5})
    assert np.isclose(a["hip_L"][0], -0.5)


def test_bad_filename_raises():
    with pytest.raises(ValueError):
        decode_trial_name("nonsense.csv")


def test_nan_markers_raise():
    d = {c: [np.nan] for c in COLLINEAR_COLUMNS}
    with pytest.raises(ValueError):
        markers_to_angles(pd.DataFrame(d), rest_pose=None)


def test_treadmill_ratio():
    assert treadmill_surfacevel(decode_trial_name("RatA_x2p00_Lfast_S1.csv"))[1] == 0.4


def test_treadmill_tied_is_symmetric():
    slow, fast = treadmill_surfacevel(decode_trial_name("RatA_x1p50_Lfast_T1.csv"))
    assert slow == fast