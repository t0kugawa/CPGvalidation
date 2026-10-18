import numpy as np
import pandas as pd
import pytest

from loader.mocap import (
    FOOT_REST_ANG,
    SHANK_REST_ANG,
    THIGH_REST_ANG,
    decode_trial_name,
    markers_to_angles,
    treadmill_surfacevel,
)

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
    a = markers_to_angles(pd.DataFrame(_straight_leg_frame()))
    assert set(a) == {"hip_L", "knee_L", "ankle_L", "hip_R", "knee_R", "ankle_R",
                      "hip_x_L", "hip_z_L", "hip_x_R", "hip_z_R"}
    # A collinear (straight) leg maps to the rig's rest segment directions.
    for s in ("L", "R"):
        assert np.isclose(a[f"hip_{s}"][0], THIGH_REST_ANG)
        assert np.isclose(a[f"knee_{s}"][0], SHANK_REST_ANG - THIGH_REST_ANG)
        assert np.isclose(a[f"ankle_{s}"][0], FOOT_REST_ANG - SHANK_REST_ANG)
    # Hip position is the GT marker (2nd marker => x=1.0, z=0.0).
    assert np.isclose(a["hip_x_L"][0], 1.0)
    assert np.isclose(a["hip_z_L"][0], 0.0)


def test_point_based_angles_round_trip():
    # Segments rotated by known angles: thigh -0.5, knee +0.3, ankle -0.2.
    d = {c: [0.0] for c in COLLINEAR_COLUMNS}
    phi_t, phi_k, phi_a = -0.5, 0.3, -0.2
    lengths = {"GT_KN": 0.030, "KN_EA": 0.036, "EA_MH": 0.021}
    for side in ("L", "R"):
        gt = np.array([0.0, 0.0])
        kn = gt + lengths["GT_KN"] * np.array([np.cos(phi_t), np.sin(phi_t)])
        ea = kn + lengths["KN_EA"] * np.array([np.cos(phi_k), np.sin(phi_k)])
        mh = ea + lengths["EA_MH"] * np.array([np.cos(phi_a), np.sin(phi_a)])
        for m, p in (("GT", gt), ("KN", kn), ("EA", ea), ("MH", mh)):
            d[f"{m}_{side}_x"] = [p[0]]
            d[f"{m}_{side}_z"] = [p[1]]
    a = markers_to_angles(pd.DataFrame(d))
    for s in ("L", "R"):
        assert np.isclose(a[f"hip_{s}"][0], THIGH_REST_ANG - phi_t)
        knee = (SHANK_REST_ANG - phi_k) - (THIGH_REST_ANG - phi_t)
        assert np.isclose(a[f"knee_{s}"][0], knee)
        assert np.isclose(a[f"ankle_{s}"][0], (FOOT_REST_ANG - phi_a) - (THIGH_REST_ANG - phi_t) - knee)


def test_bad_filename_raises():
    with pytest.raises(ValueError):
        decode_trial_name("nonsense.csv")


def test_nan_markers_raise():
    d = {c: [np.nan] for c in COLLINEAR_COLUMNS}
    with pytest.raises(ValueError):
        markers_to_angles(pd.DataFrame(d))


def test_treadmill_ratio():
    assert treadmill_surfacevel(decode_trial_name("RatA_x2p00_Lfast_S1.csv"))[1] == 0.4


def test_treadmill_tied_is_symmetric():
    slow, fast = treadmill_surfacevel(decode_trial_name("RatA_x1p50_Lfast_T1.csv"))
    assert slow == fast