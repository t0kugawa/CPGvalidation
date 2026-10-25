"""Mandatory five-cycle lite phase gate.

This is intentionally a slow test: it runs the closed loop with a 1/5-scale
CPG, renders the 3D view, and checks the pattern metrics. It is the gate that
must pass before a phase/credit/game commit is approved; the synthetic
``test_phase_metrics.py`` checks the metric code itself in seconds.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from pathlib import Path

import pandas as pd
import pytest

import run_validation
from persona.phase_metrics import phase_gate_metrics
from model.model_cfg import ANKLE_RANGE_RAD

OUT = Path("archive/phase_gate_current")
LITE = dict(n_rg=20, n_mn=20, n_int=10, n_aff=20, n_bs=20)

pytestmark = pytest.mark.phase_gate


@pytest.fixture(scope="module")
def gate_run():
    OUT.mkdir(parents=True, exist_ok=True)
    return run_validation.run(
        duration_s=8.0,
        view="oblique_view",
        width=320,
        height=240,
        fps=20,
        out_dir=str(OUT),
        w_scale=1.0,
        ref_hz=run_validation.DEFAULT_REF_HZ,
        belt_mps=0.2,
        belt_start_s=0.5,
        belt_ramp_s=0.5,
        n_rg=LITE["n_rg"],
        n_mn=LITE["n_mn"],
        n_int=LITE["n_int"],
        n_aff=LITE["n_aff"],
        n_bs=LITE["n_bs"],
        render=True,
        budget_s=None,
        pose_max_s=1.0,
        belt_mode="dataset",
        snapshot_s=0,
        realize_every_k=1,
        resolution_ms=run_validation.DEFAULT_RESOLUTION_MS,
        cut_multisite=False,
        game=None,
        seed=0,
        nest_threads=1,
    )


def test_five_cycle_lite_phase_gate(gate_run):
    log = pd.read_csv(OUT / "run_log.csv")
    video = OUT / "locomotion_oblique_view.mp4"
    report = phase_gate_metrics(
        {c: log[c].to_numpy() for c in log.columns},
        dt_ms=10.0,
        ankle_range=ANKLE_RANGE_RAD,
        render_path=str(video),
    )
    print(report.summary())
    assert report.passed, report.summary()
