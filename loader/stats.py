from typing import Dict, Sequence

import numpy as np
import pandas as pd


class StatsLoader:
    """
    Reads the Zenodo statistical table (duty factor, double support time,
    step/stride length, synergy contributions) and exposes per-field
    distributions across the five rats.
    """
    def __init__(self, path: str):
        self.path = path
        self.df = pd.read_csv(path)

    def distribution(self, field: str, rat_ids: Sequence[str] | None = None) -> np.ndarray:
        if field not in self.df.columns:
            raise KeyError(f"Field '{field}' not in statistical table")
        d = self.df
        if rat_ids is not None:
            d = d[d["rat_id"].isin(list(rat_ids))]
        vals = pd.to_numeric(d[field], errors="coerce").to_numpy(dtype=np.float64)
        return vals[np.isfinite(vals)]

    def rat_distributions(self, field: str) -> Dict[str, np.ndarray]:
        return {
            rat: self.distribution(field, rat_ids=[rat])
            for rat in sorted(self.df["rat_id"].dropna().unique())
        }


def _duty_factor(contact: np.ndarray, threshold: float = 0.5) -> float:
    on = np.asarray(contact, dtype=np.float64) > threshold
    return float(np.mean(on)) if on.size else 0.0


def _double_support(contact_L: np.ndarray, contact_R: np.ndarray, threshold: float = 0.5) -> float:
    both = (np.asarray(contact_L) > threshold) & (np.asarray(contact_R) > threshold)
    return float(np.mean(both)) if both.size else 0.0


def _excursion(signal: np.ndarray) -> float:
    s = np.asarray(signal, dtype=np.float64)
    return float(np.ptp(s)) if s.size else 0.0


class LocomotionMetrics:
    """
    Locomotion parameters computed from a simulation run. Duty factor and
    double support are dimensionless and directly comparable to the Zenodo
    statistics; step/stride length are horizontal foot excursions in mm when
    foot positions are supplied, otherwise ankle-angle excursions (rad) used
    as a kinematic proxy.
    """
    @staticmethod
    def from_run(ankle_L: np.ndarray, ankle_R: np.ndarray,
                 contact_L: np.ndarray, contact_R: np.ndarray,
                 dt_ms: float,
                 foot_pos_L: np.ndarray | None = None,
                 foot_pos_R: np.ndarray | None = None) -> Dict[str, float]:
        def _length(side_ankle, side_pos):
            if side_pos is not None:
                pos = np.asarray(side_pos, dtype=np.float64)
                x = pos[:, 0] if pos.ndim == 2 else pos
                return float(np.ptp(x) * 1000.0)  # m -> mm
            return _excursion(side_ankle)

        return {
            "duty_factor_L": _duty_factor(contact_L),
            "duty_factor_R": _duty_factor(contact_R),
            "double_support": _double_support(contact_L, contact_R),
            "step_length_L": _length(ankle_L, foot_pos_L),
            "step_length_R": _length(ankle_R, foot_pos_R),
            "stride_length_L": _length(ankle_L, foot_pos_L),
            "stride_length_R": _length(ankle_R, foot_pos_R),
        }


def in_distribution(value: float, dist: np.ndarray, k: float = 2.0) -> bool:
    """
    True if `value` lies within mean +/- k standard deviations of the
    reference distribution (a pragmatic proxy for the rat cluster).
    """
    d = np.asarray(dist, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return False
    if d.size == 1:
        return bool(np.isclose(value, d[0]))
    return bool(abs(value - d.mean()) <= k * d.std())