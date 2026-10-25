"""Fast, model-independent phase-gate metrics.

The gate is deliberately not a reference-tracking test. It asks whether the
current run produced a stable repeated pattern:

* at least ``min_cycles`` complete stance/swing cycles from contacts;
* a stable period;
* E/F anti-phase within a limb;
* stance onset aligned with the E burst;
* a stable, non-zero L/R offset (not forced to pi);
* periodic joint-angle cycles (hip, knee, ankle, both sides).

Everything is computed from logged arrays, so the same code can be used on a
lite gate run, a full run, or a unit-test trace. Reference matching remains a
PERSONA validation task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np

TAU = 2.0 * math.pi


@dataclass
class PhaseGateReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.passed:
            head = "phase gate: PASS"
        else:
            head = "phase gate: FAIL"
        m = self.metrics
        lines = [head]
        if "cycles" in m:
            lines.append(f"  cycles={m['cycles']} period={m['period_s']:.3f}s "
                         f"cv={m['period_cv']:.3f}")
        if "ef_phase" in m:
            lines.append(f"  E/F offset={m['ef_phase']:+.3f} rad "
                         f"std={m['ef_std']:.3f}")
        if "lr_phase" in m:
            lines.append(f"  L/R offset={m['lr_phase']:+.3f} rad "
                         f"drift={m['lr_drift']:.3f}")
        if "stance_e_phase" in m:
            lines.append(f"  stance/E={m['stance_e_phase']:+.3f} rad "
                         f"std={m['stance_e_std']:.3f}")
        if "angle_corr" in m:
            lines.append(f"  angle corr={m['angle_corr']:.3f}")
        for f in self.failures:
            lines.append(f"  - {f}")
        return "\n".join(lines)


def _wrap(x):
    return (np.asarray(x, dtype=float) + math.pi) % TAU - math.pi


def _circular(angles) -> tuple[float, float, int]:
    a = np.asarray(angles, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), 0
    z = np.exp(1j * a)
    mean = float(np.angle(np.mean(z)))
    r = float(abs(np.mean(z)))
    std = float(math.sqrt(max(-2.0 * math.log(max(r, 1e-12)), 0.0)))
    return mean, std, int(a.size)


def _rising_times(mask, dt_s: float) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0:
        return np.zeros(0, dtype=float)
    idx = np.flatnonzero((~m[:-1]) & m[1:]) + 1
    if m[0]:
        idx = np.r_[0, idx]
    return idx.astype(float) * float(dt_s)


def _debounce(times, min_gap_s: float) -> np.ndarray:
    t = np.asarray(times, dtype=float).ravel()
    if t.size <= 1 or min_gap_s <= 0:
        return t
    out = [float(t[0])]
    for x in t[1:]:
        if float(x) - out[-1] >= min_gap_s:
            out.append(float(x))
    return np.asarray(out, dtype=float)


def _stance_onsets(phase, dt_s: float, min_gap_s: float = 0.0) -> np.ndarray:
    p = np.asarray(phase, dtype=object).ravel()
    mask = np.asarray([str(x).strip().lower() == "stance" for x in p], dtype=bool)
    return _debounce(_rising_times(mask, dt_s), min_gap_s)


def _bursts(signal, dt_s: float, *, threshold: float = 0.5,
            min_gap_s: float = 0.04) -> np.ndarray:
    s = np.asarray(signal, dtype=float).ravel()
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if s.size == 0 or float(np.max(s)) <= 0.0:
        return np.zeros(0, dtype=float)
    # A constant co-contraction is not a burst pattern.
    if float(np.mean(s >= threshold)) > 0.90:
        return np.zeros(0, dtype=float)
    # If the trace is mostly above the nominal threshold, use its upper
    # quartile; the gate only needs repeated burst timing, not an absolute
    # activation unit.
    if float(np.mean(s >= threshold)) > 0.5:
        threshold = float(np.quantile(s, 0.75))
    return _debounce(_rising_times(s >= threshold, dt_s), min_gap_s)


def _phase_from_onsets(t, onsets, period_s: float) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    o = np.asarray(onsets, dtype=float).ravel()
    phase = np.full(t.shape, np.nan, dtype=float)
    if o.size == 0 or period_s <= 0:
        return phase
    idx = np.searchsorted(o, t, side="right") - 1
    valid = idx >= 0
    phase[valid] = (TAU * (t[valid] - o[idx[valid]]) / period_s) % TAU
    return phase


def _phase_of_events(events, reference, period_s: float) -> np.ndarray:
    e = np.asarray(events, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    if e.size == 0 or r.size == 0 or period_s <= 0:
        return np.zeros(0, dtype=float)
    # Nearest reference onset for each event.
    j = np.abs(e[:, None] - r[None, :]).argmin(axis=1)
    return _wrap(TAU * (e - r[j]) / period_s)


def _nearest_pairs(a, b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return np.zeros(0), np.zeros(0)
    j = np.abs(a[:, None] - b[None, :]).argmin(axis=1)
    return a, b[j]


def _cycle_correlation(trace, onsets, n_cycles: int, dt_s: float,
                       n: int = 64) -> float:
    x = np.asarray(trace, dtype=float).ravel()
    o = np.asarray(onsets, dtype=float).ravel()
    if x.size == 0 or o.size < 2 or dt_s <= 0:
        return float("nan")
    start = max(0, len(o) - (n_cycles + 1))
    cycles = []
    for k in range(start, len(o) - 1):
        i0 = int(round(o[k] / dt_s))
        i1 = int(round(o[k + 1] / dt_s))
        i0 = min(max(i0, 0), x.size - 2)
        i1 = min(max(i1, i0 + 2), x.size)
        cycles.append(np.interp(np.linspace(0, 1, n),
                                np.linspace(0, 1, i1 - i0), x[i0:i1]))
    if len(cycles) < 2:
        return float("nan")
    corrs = []
    for a, b in zip(cycles[:-1], cycles[1:]):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def _stuck_at_limit(q, target, lo: float, hi: float, dt_s: float, *,
                    margin: float = 0.06, target_eps: float = 0.15,
                    vel_eps: float = 0.01,
                    min_duration_s: float = 0.5) -> list[dict]:
    """Contiguous runs pinned on a joint stop while the target is far away."""
    q = np.asarray(q, dtype=float).ravel()
    t = np.asarray(target, dtype=float).ravel()
    if q.size < 3 or t.size != q.size:
        return []
    dq = np.diff(q, prepend=q[0])
    at = (q <= lo + margin) | (q >= hi - margin)
    far = np.abs(q - t) > target_eps
    slow = np.abs(dq) < vel_eps
    stuck = at & far & slow
    runs = []
    start = None
    for i, v in enumerate(stuck):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(stuck) - 1):
            end = i if v else i - 1
            dur = (end - start + 1) * dt_s
            if dur >= min_duration_s:
                limit = "upper" if q[start] >= hi - margin else "lower"
                runs.append({
                    "start_s": float(start * dt_s),
                    "duration_s": float(dur),
                    "limit": limit,
                    "q": float(q[start]),
                    "target": float(t[start]),
                })
            start = None
    return runs


def _foot_inverted_runs(sole_z, dt_s: float, *,
                        threshold: float = 0.05,
                        min_duration_s: float = 0.2) -> list[dict]:
    z = np.asarray(sole_z, dtype=float).ravel()
    up = z > threshold
    runs = []
    start = None
    for i, v in enumerate(up):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(up) - 1):
            end = i if v else i - 1
            dur = (end - start + 1) * dt_s
            if dur >= min_duration_s:
                runs.append({
                    "start_s": float(start * dt_s),
                    "duration_s": float(dur),
                    "max_sole_z": float(np.max(z[start:end + 1])),
                })
            start = None
    return runs


def _energy_phase(theta, signal) -> float:
    """Circular phase of a non-negative signal's energy over a cycle."""
    th = np.asarray(theta, dtype=float).ravel()
    s = np.asarray(signal, dtype=float).ravel()
    if th.size != s.size or th.size == 0:
        return float("nan")
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.maximum(s, 0.0)
    mask = np.isfinite(th) & (s > 0.0)
    total = float(np.sum(s[mask]))
    if not mask.any() or total <= 0.0:
        return float("nan")
    z = np.sum(s[mask] * np.exp(1j * th[mask])) / total
    return float(np.angle(z))


def _period_autocorr(trace, period_s: float, dt_s: float) -> float:
    """Correlation of a trace with itself one period later."""
    x = np.asarray(trace, dtype=float).ravel()
    if x.size < 4 or not np.isfinite(period_s) or period_s <= 0 or dt_s <= 0:
        return float("nan")
    lag = int(round(period_s / dt_s))
    if lag < 2 or lag >= x.size - 2:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def phase_gate_metrics(
    log: Mapping,
    *,
    dt_ms: float = 10.0,
    min_cycles: int = 5,
    min_cycle_s: float = 0.25,
    period_cv_max: float = 0.20,
    ef_tol: float = 0.35 * math.pi,
    stance_e_tol: float = 0.35 * math.pi,
    lr_drift_max: float = 0.10 * math.pi,
    angle_corr_min: float = 0.50,
    ankle_range: tuple[float, float] | None = None,
    render_path: str | None = None,
) -> PhaseGateReport:
    """Compute the mandatory phase gate from a run log mapping.

    ``log`` needs the same arrays as ``RunLog``/``run_log.csv``:
    ``phase_L/R``, ``mus_E/F_L/R`` or ``rg_e/f_L/R``, and
    ``hip_model_*``, ``knee_model_*``, ``ankle_model_*``.
    """
    dt_s = float(dt_ms) / 1000.0
    failures: list[str] = []
    metrics: dict = {}

    def arr(name):
        if name not in log:
            return np.zeros(0, dtype=float)
        return np.asarray(log[name], dtype=float)

    def phase_arr(name):
        if name not in log:
            return np.zeros(0, dtype=object)
        return np.asarray(log[name], dtype=object)

    # Contact cycles and period.
    cycles = {}
    periods = {}
    for s in "LR":
        on = _stance_onsets(phase_arr(f"phase_{s}"), dt_s,
                            min_gap_s=min_cycle_s)
        cycles[s] = on
        if on.size >= 2:
            periods[s] = np.diff(on)[-min_cycles:]
    all_periods = np.concatenate([v for v in periods.values() if v.size]) \
        if periods else np.zeros(0)
    if all_periods.size == 0:
        failures.append("no stance/swing cycles detected")
        period_s = float("nan")
        period_cv = float("nan")
        n_cycles = 0
    else:
        period_s = float(np.mean(all_periods))
        period_cv = float(np.std(all_periods) / max(period_s, 1e-12))
        n_cycles = int(min(max(0, len(cycles[s]) - 1) for s in "LR"))
    metrics.update(period_s=period_s, period_cv=period_cv,
                   cycles=n_cycles,
                   cycles_L=max(0, len(cycles["L"]) - 1),
                   cycles_R=max(0, len(cycles["R"]) - 1),
                   cycle_onsets={s: cycles[s].tolist() for s in "LR"})
    if n_cycles < min_cycles:
        failures.append(f"only {n_cycles} complete cycles; need {min_cycles}")
    if np.isfinite(period_cv) and period_cv > period_cv_max:
        failures.append(f"period CV {period_cv:.3f} > {period_cv_max:.3f}")

    # E/F anti-phase from the energy-weighted circular phase of the applied
    # motor commands. Burst thresholding is wrong for the smooth applied
    # envelope (and F may never cross a hard threshold).
    t = np.arange(len(phase_arr("phase_L")), dtype=float) * dt_s
    ef_offsets = []
    se_offsets = []
    for s in "LR":
        e = arr(f"applied_mus_E_{s}")
        f = arr(f"applied_mus_F_{s}")
        if not np.any(e):
            e = arr(f"mus_E_{s}")
        if not np.any(f):
            f = arr(f"mus_F_{s}")
        if not np.any(e):
            e = arr(f"rg_e_{s}")
        if not np.any(f):
            f = arr(f"rg_f_{s}")
        th = _phase_from_onsets(t, cycles[s], period_s)
        ep = _energy_phase(th, e)
        fp = _energy_phase(th, f)
        if np.isfinite(ep) and np.isfinite(fp):
            ef_offsets.append(float(_wrap(ep - fp)))
        if np.isfinite(ep):
            # E energy is expected to be centered in stance (pi/2 after the
            # stance onset), not at the onset itself.
            se_offsets.append(float(_wrap(ep - math.pi / 2.0)))
    ef_mean, ef_std, ef_n = _circular(ef_offsets)
    metrics.update(ef_phase=ef_mean, ef_std=ef_std, ef_n=ef_n)
    if ef_n < 2:
        failures.append("could not detect both E and F energy phases")
    else:
        if abs(abs(ef_mean) - math.pi) > ef_tol:
            failures.append(
                f"E/F offset {ef_mean:+.3f} rad is not anti-phase "
                f"(tol {ef_tol:.3f})")
        if ef_std > ef_tol:
            failures.append(f"E/F offset drift {ef_std:.3f} rad > {ef_tol:.3f}")

    se_mean, se_std, se_n = _circular(se_offsets)
    metrics.update(stance_e_phase=se_mean, stance_e_std=se_std, stance_e_n=se_n)
    if se_n < 2:
        failures.append("could not align E energy to stance onsets")
    else:
        if abs(se_mean) > stance_e_tol:
            failures.append(
                f"stance/E offset {se_mean:+.3f} rad outside "
                f"±{stance_e_tol:.3f}")
        if se_std > stance_e_tol:
            failures.append(
                f"stance/E drift {se_std:.3f} rad > {stance_e_tol:.3f}")

    # Stable L/R offset from the first-class clock phase. If the clock is not
    # logged (older runs), fall back to continuous onset phase. Nearest-onset
    # pairing is ambiguous near half a cycle, so compare phases at each sample.
    lr_offsets = []
    clk_l = arr("phase_clock_L")
    clk_r = arr("phase_clock_R")
    if clk_l.size and clk_l.size == clk_r.size:
        t = np.arange(clk_l.size, dtype=float) * dt_s
        start = 0.0
        for s in "LR":
            if len(cycles[s]) > min_cycles:
                start = max(start, float(cycles[s][-(min_cycles + 1)]))
        lr_offsets = _wrap(clk_r[t >= start] - clk_l[t >= start]).tolist()
    elif cycles["L"].size and cycles["R"].size and np.isfinite(period_s):
        t = np.arange(len(phase_arr("phase_L")), dtype=float) * dt_s
        start = 0.0
        for s in "LR":
            if len(cycles[s]) > min_cycles:
                start = max(start, float(cycles[s][-(min_cycles + 1)]))
        th_l = _phase_from_onsets(t, cycles["L"], period_s)
        th_r = _phase_from_onsets(t, cycles["R"], period_s)
        valid = (t >= start) & np.isfinite(th_l) & np.isfinite(th_r)
        lr_offsets = _wrap(th_l[valid] - th_r[valid]).tolist()
    lr_mean, lr_std, lr_n = _circular(lr_offsets)
    metrics.update(lr_phase=lr_mean, lr_drift=lr_std, lr_n=lr_n)
    if lr_n < 2:
        failures.append("could not evaluate L/R stance offset")
    else:
        if abs(lr_mean) < 0.02:
            failures.append("L/R offset is effectively zero")
        if lr_std > lr_drift_max:
            failures.append(
                f"L/R drift {lr_std:.3f} rad > {lr_drift_max:.3f}")
        # Peak-to-peak spread catches a slow monotone drift that a circular
        # std can understate.
        lo = float(np.percentile(lr_offsets, 10))
        hi = float(np.percentile(lr_offsets, 90))
        metrics["lr_spread"] = hi - lo
        if hi - lo > 2.0 * lr_drift_max:
            failures.append(
                f"L/R offset spread {hi - lo:.3f} rad > "
                f"{2.0 * lr_drift_max:.3f}")

    # Joint-angle periodicity. Reuse each side's contact onsets.
    corrs = []
    for s in "LR":
        per = arr(f"phase_period_{s}")
        p = float(per[-1]) if per.size else period_s
        for joint in ("hip", "knee", "ankle"):
            trace = arr(f"{joint}_model_{s}")
            if trace.size:
                c = _period_autocorr(trace, p, dt_s)
                if np.isfinite(c):
                    corrs.append(c)
    angle_corr = float(np.mean(corrs)) if corrs else float("nan")
    metrics.update(angle_corr=angle_corr, angle_corr_n=len(corrs))
    if not np.isfinite(angle_corr):
        failures.append("could not evaluate joint-angle periodicity")
    elif angle_corr < angle_corr_min:
        failures.append(
            f"joint-angle cycle correlation {angle_corr:.3f} < "
            f"{angle_corr_min:.3f}")

    # Ankle pinned on a stop while the target asks it to be elsewhere.
    stuck = {}
    if ankle_range is not None:
        lo, hi = (float(ankle_range[0]), float(ankle_range[1]))
        for s in "LR":
            runs = _stuck_at_limit(
                arr(f"ankle_model_{s}"), arr(f"ankle_target_{s}"),
                lo, hi, dt_s)
            stuck[s] = runs
            for run in runs:
                failures.append(
                    f"ankle {s} stuck at {run['limit']} stop for "
                    f"{run['duration_s']:.2f}s "
                    f"(q={run['q']:+.3f}, target={run['target']:+.3f})")
    metrics["ankle_stuck"] = stuck

    # Foot-sole inversion: the local -z axis of the foot body should not point
    # upward. This is recorded by the plant as `foot_sole_z_*`.
    inverted = {}
    for s in "LR":
        key = f"foot_sole_z_{s}"
        if key in log:
            runs = _foot_inverted_runs(arr(key), dt_s)
            inverted[s] = runs
            for run in runs:
                failures.append(
                    f"foot {s} sole points up for {run['duration_s']:.2f}s "
                    f"(max sole z={run['max_sole_z']:+.3f})")
    metrics["foot_inverted"] = inverted

    if render_path is not None:
        import os
        ok = os.path.exists(render_path) and os.path.getsize(render_path) > 0
        metrics["render_path"] = str(render_path)
        metrics["render_ok"] = bool(ok)
        if not ok:
            failures.append(f"3D render missing/empty: {render_path}")

    return PhaseGateReport(passed=not failures, failures=failures, metrics=metrics)
