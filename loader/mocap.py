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


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = np.where((n1 * n2) > 1e-12, n1 * n2, 1.0)
    cos = np.clip(np.sum(v1 * v2, axis=-1) / denom, -1.0, 1.0)
    ang = np.arccos(cos)
    # Signed angle in the sagittal (x-z) plane for flexion/extension.
    cross_y = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
    return np.where(cross_y < 0.0, -ang, ang)


def markers_to_angles(df: pd.DataFrame, rest_pose: Dict[str, float] | None) -> Dict[str, np.ndarray]:
    """
    Converts marker coordinates (mm) into per-side joint angles (rad) using
    planar sagittal marker triples: hip = angle(IC->GT, GT->KN),
    knee = angle(GT->KN, KN->EA), ankle = angle(KN->EA, EA->MH).
    When a rest_pose is given, the rest-pose angle is subtracted (zero-aligned).
    """
    out: Dict[str, np.ndarray] = {}
    for side in SIDES:
        pts = {}
        for marker in MARKERS:
            cols = _marker_columns(side, marker)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing marker columns {missing} for {marker}_{side}")
            arr = df[cols].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Non-finite marker values for {marker}_{side}")
            pts[marker] = arr

        thigh = pts["GT"] - pts["IC"]
        shank = pts["KN"] - pts["GT"]
        foot = pts["EA"] - pts["KN"]
        toe = pts["MH"] - pts["EA"]

        hip = _angle_between(thigh, shank)
        knee = _angle_between(shank, foot)
        ankle = _angle_between(foot, toe)

        for joint, arr in (("hip", hip), ("knee", knee), ("ankle", ankle)):
            if rest_pose is not None and f"{joint}_{side}" in rest_pose:
                arr = arr - float(rest_pose[f"{joint}_{side}"])
            out[f"{joint}_{side}"] = arr

    return out


class MocapLoader:
    """
    Loads a Zenodo mocap trial, extracts per-side joint angles, and resamples
    them to the simulation timestep.
    """
    def __init__(self, path: str, rest_pose: Dict[str, float] | None = None,
                 dt_ms: float = 10.0, sample_hz: float = 500.0):
        self.path = path
        self.dt_ms = float(dt_ms)
        self.df = pd.read_csv(path)
        self.meta = decode_trial_name(path)
        self.angles = markers_to_angles(self.df, rest_pose)
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