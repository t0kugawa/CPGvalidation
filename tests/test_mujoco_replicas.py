"""Two MuJoCo rigs are independent and can be stepped in a thread pool."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("mujoco")

from model.adapter import CPGOutput
from model.leg import Leg as MuJoCo_Leg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(ROOT, "dual_rat_hindlimb.xml")


def _rig():
    rig = MuJoCo_Leg(mujoco_cfg=XML, dt_ms=10.0)
    rig.reset()
    return rig


def _drive(extensor: float, flexor: float) -> CPGOutput:
    return CPGOutput(
        mus_E_L=extensor, mus_F_L=flexor,
        mus_E_R=extensor, mus_F_R=flexor,
    )


def _inputs(rig, drive):
    out = rig.step(drive)
    return np.array([
        out.ankle_angle_m_L, out.ankle_angle_m_R,
        out.hip_angle_m_L, out.knee_angle_m_L,
        out.cut_L, out.cut_R,
    ])


def test_two_rigs_have_independent_state():
    left, right = _rig(), _rig()
    left_in = _inputs(left, _drive(0.8, 0.1))
    right_in = _inputs(right, _drive(0.1, 0.8))
    assert not np.allclose(left_in, right_in)


def test_threaded_rig_steps_match_sequential():
    drive_a = _drive(0.8, 0.1)
    drive_b = _drive(0.1, 0.8)

    left, right = _rig(), _rig()
    sequential = [_inputs(left, drive_a), _inputs(right, drive_b)]

    left, right = _rig(), _rig()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_inputs, left, drive_a),
                   pool.submit(_inputs, right, drive_b)]
        threaded = [future.result() for future in futures]

    assert np.allclose(sequential[0], threaded[0])
    assert np.allclose(sequential[1], threaded[1])
    assert not np.allclose(threaded[0], threaded[1])
