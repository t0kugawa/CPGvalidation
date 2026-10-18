import re
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

MARKERS = ("IC", "GT", "KN", "EA", "MH")
SIDES = ("L", "R")

_TRIAL_RE = re.compile(
    r"^(?P<rat>Rat[A-Za-z]+)_x(?P<ratio>\d+)(?:p|d)(?P<frac>\d+)_(?P<belt>[LR])fast_"
    r"(?P<cond>[TSW])(?P<trial>\d+)\.csv$"
)

_CONDITIONS = {"T": "tied", "S": "split-belt", "W": "wash-out"}

SLOW_BELT_MPS = 0.2


@dataclass
class TrialMeta:
    rat_id: str
    speed_ratio: float
    fast_belt: str
    condition: str
    trial: int


def decode_trial_name(name: str) -> TrialMeta:
    """
    Parses a Zenodo mocap filename token, e.g. "RatA_x1p75_Rfast_T2.csv".
    The speed ratio is encoded with a 'p' or 'd' decimal separator
    (x1p50 or x1d50 = 1.5).
    """
    base = name.split("/")[-1].split("\\")[-1]
    m = _TRIAL_RE.match(base)
    if m is None:
        raise ValueError(f"Cannot decode mocap trial name: {name!r}")
    ratio = float(f"{m.group('ratio')}.{m.group('frac')}")
    return TrialMeta(
        rat_id=m.group("rat"),
        speed_ratio=ratio,
        fast_belt=m.group("belt"),
        condition=m.group("cond"),
        trial=int(m.group("trial")),
    )


def _marker_columns(side: str, marker: str) -> list[str]:
    return [f"{marker}_{side}_{ax}" for ax in ("x", "y", "z")]


# Rest-pose segment directions (rad) of the TARGET limb in
# dual_rat_hindlimb.xml, i.e. the sagittal orientation of each link in its own
# rest frame: thigh fromto (0.012, -0.032); shank fromto (-0.028, -0.024);
# ankle->metatarsal (toe-base) offset (0.022, -0.004). The target limb's link
# LENGTHS are set from the trial's own segment lengths, but these DIRECTIONS
# (hence the rest angles) are preserved. Keep in sync with the XML.
THIGH_REST_ANG = float(np.arctan2(-0.032, 0.012))
SHANK_REST_ANG = float(np.arctan2(-0.024, -0.028))
FOOT_REST_ANG = float(np.arctan2(-0.004, 0.022))


def _marker_xyz(df: pd.DataFrame, side: str, marker: str) -> np.ndarray:
    cols = _marker_columns(side, marker)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing marker columns {missing} for {marker}_{side}")
    arr = df[cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite marker values for {marker}_{side}")
    return arr


def markers_to_angles(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Point-based rigid-bone reconstruction of the target limb.

    The articulated chain is built from the marker points themselves: the
    absolute sagittal (x-z) orientation of each segment (thigh GT->KN, shank
    KN->EA, foot EA->MH) is mapped to the rig's hinge angles so the target limb
    reproduces the marker chain. Unlike unsigned segment-to-segment interior
    angles, this preserves orientation and sign, and yields the true relative
    joint angles (θ_hip = θ_thigh_rest - φ_thigh, θ_knee = (θ_shank_rest -
    φ_shank) - θ_hip, θ_ankle = (θ_foot_rest - φ_foot) - θ_hip - θ_knee).

    Also returns the hip (GT) position `hip_x_{side}`/`hip_z_{side}` in mm so the
    target limb's base can track the dataset; the mapping is absolute, referenced
    to the rig's rest directions.
    """
    out: Dict[str, np.ndarray] = {}
    for side in SIDES:
        gt = _marker_xyz(df, side, "GT")
        kn = _marker_xyz(df, side, "KN")
        ea = _marker_xyz(df, side, "EA")
        mh = _marker_xyz(df, side, "MH")

        phi_thigh = np.arctan2(kn[:, 2] - gt[:, 2], kn[:, 0] - gt[:, 0])
        phi_shank = np.arctan2(ea[:, 2] - kn[:, 2], ea[:, 0] - kn[:, 0])
        phi_foot = np.arctan2(mh[:, 2] - ea[:, 2], mh[:, 0] - ea[:, 0])

        hip = THIGH_REST_ANG - phi_thigh
        knee = (SHANK_REST_ANG - phi_shank) - hip
        ankle = (FOOT_REST_ANG - phi_foot) - hip - knee

        out[f"hip_{side}"] = hip
        out[f"knee_{side}"] = knee
        out[f"ankle_{side}"] = ankle
        out[f"hip_x_{side}"] = gt[:, 0]
        out[f"hip_z_{side}"] = gt[:, 2]

    return out


class MocapLoader:
    """
    Loads a Zenodo mocap trial, extracts per-side joint angles, and resamples
    them to the simulation timestep.
    """
    def __init__(self, path: str, dt_ms: float = 10.0, sample_hz: float = 500.0):
        self.path = path
        self.dt_ms = float(dt_ms)
        self.df = pd.read_csv(path)
        self.meta = decode_trial_name(path)
        self.angles = markers_to_angles(self.df)
        self._t_ms = np.arange(len(self.df)) * (1000.0 / sample_hz)
        self._resampled = self._resample()

    def _resample(self) -> Dict[str, np.ndarray]:
        n = len(self.df)
        if n == 0:
            raise ValueError(f"Empty mocap trial: {self.path}")
        t_end = self._t_ms[-1]
        n_out = max(1, int(round(t_end / self.dt_ms)) + 1)
        t_out = np.arange(n_out) * self.dt_ms
        return {k: np.interp(t_out, self._t_ms, v) for k, v in self.angles.items()}

    def resample(self) -> Dict[str, np.ndarray]:
        return self._resampled

    def ankle(self, side: str) -> np.ndarray:
        return self._resampled[f"ankle_{side}"]

    def __len__(self) -> int:
        return len(next(iter(self._resampled.values())))

    def __iter__(self):
        n = len(self)
        for i in range(n):
            yield {k: float(v[i]) for k, v in self._resampled.items()}


def treadmill_surfacevel(meta: TrialMeta, slow_mps: float = SLOW_BELT_MPS) -> tuple[float, float]:
    """
    Returns (slow_belt_mps, fast_belt_mps) for the MuJoCo treadmill from the
    trial metadata: tied => equal belts; split-belt => fast = ratio * slow.
    """
    if meta.condition == "T":
        return (slow_mps, slow_mps)
    return (slow_mps, slow_mps * meta.speed_ratio)