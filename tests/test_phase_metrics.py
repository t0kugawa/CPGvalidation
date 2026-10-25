"""Fast synthetic tests for the phase gate; no MuJoCo or NEST."""
from __future__ import annotations

import numpy as np
import pytest

from persona.phase_metrics import phase_gate_metrics


def _synthetic_log(period_s=0.8, n_cycles=8, lr_offset=0.30,
                   ef_offset=0.5):
    dt = 0.01
    t = np.arange(0.0, n_cycles * period_s, dt)
    out = {"time_ms": t * 1000.0}
    for side, off in (("L", 0.0), ("R", lr_offset * period_s)):
        # stance for the first half of each cycle; R is offset.
        phase_pos = ((t - off) % period_s) / period_s
        stance = phase_pos < 0.5
        out[f"phase_{side}"] = np.where(stance, "stance", "swing").astype(object)
        # E at stance onset, F half a cycle later.
        e = np.zeros_like(t)
        f = np.zeros_like(t)
        for onset in np.arange(off, t[-1] + period_s, period_s):
            ie = int(round((onset + 0.25 * period_s) / dt))
            iff = int(round((onset + (0.25 + ef_offset) * period_s) / dt))
            if 0 <= ie < len(t):
                e[ie] = 1.0
            if 0 <= iff < len(t):
                f[iff] = 1.0
        out[f"rg_e_{side}"] = e
        out[f"rg_f_{side}"] = f
        for joint, amp in (("hip", 0.2), ("knee", 0.3), ("ankle", 0.1)):
            out[f"{joint}_model_{side}"] = amp * np.sin(
                2 * np.pi * (t - off) / period_s)
    return out


def test_synthetic_phase_gate_passes():
    report = phase_gate_metrics(_synthetic_log(), dt_ms=10.0)
    print(report.summary())
    assert report.passed, report.failures
    assert report.metrics["cycles"] >= 5
    assert abs(abs(report.metrics["ef_phase"]) - np.pi) < 0.35 * np.pi
    assert abs(report.metrics["stance_e_phase"]) < 0.35 * np.pi
    assert abs(report.metrics["lr_phase"]) > 0.02
    assert report.metrics["angle_corr"] > 0.5


def test_in_phase_muscles_fail_the_gate():
    report = phase_gate_metrics(_synthetic_log(ef_offset=0.0), dt_ms=10.0)
    assert not report.passed
    assert any("E/F" in f for f in report.failures)


def test_no_cycles_fails_the_gate():
    log = _synthetic_log()
    log["phase_L"] = np.full(len(log["phase_L"]), "swing", dtype=object)
    log["phase_R"] = np.full(len(log["phase_R"]), "swing", dtype=object)
    report = phase_gate_metrics(log, dt_ms=10.0)
    assert not report.passed
    assert any("cycles" in f for f in report.failures)


def test_missing_render_fails_the_gate():
    report = phase_gate_metrics(
        _synthetic_log(), dt_ms=10.0, render_path="/no/such/render.mp4")
    assert not report.passed
    assert any("render" in f.lower() for f in report.failures)


def test_unstable_lr_offset_fails_the_gate():
    log = _synthetic_log()
    # Rebuild R phases with a drifting offset.
    period = 0.8
    dt = 0.01
    t = np.asarray(log["time_ms"]) / 1000.0
    off = 0.30 + 0.50 * (t / max(t[-1], 1e-9))
    log["phase_R"] = np.where((((t - off * period) % period) / period) < 0.5,
                              "stance", "swing").astype(object)
    report = phase_gate_metrics(log, dt_ms=10.0)
    assert not report.passed
    assert any("L/R" in f for f in report.failures)


def test_stuck_ankle_at_a_limit_fails_the_gate():
    log = _synthetic_log()
    q = np.asarray(log["ankle_model_R"], dtype=float)
    tgt = np.asarray(log["ankle_model_R"], dtype=float) * 0.0 + 1.5
    q[-120:] = 0.19
    tgt[-120:] = 1.5
    log["ankle_model_R"] = q
    log["ankle_target_R"] = tgt
    report = phase_gate_metrics(log, dt_ms=10.0, ankle_range=(0.19, 1.76))
    assert not report.passed
    assert any("ankle R stuck" in f for f in report.failures)


def test_inverted_sole_fails_the_gate():
    log = _synthetic_log()
    sole = np.full(len(log["time_ms"]), -0.9)
    sole[-80:] = 0.3
    log["foot_sole_z_R"] = sole
    report = phase_gate_metrics(log, dt_ms=10.0)
    assert not report.passed
    assert any("sole points up" in f for f in report.failures)
