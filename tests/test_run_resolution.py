"""The production loop must run at the `cpg_2legs_fast` resolution (0.2 ms).

At 1.0 ms the izhikevich network locks into a resolution-induced limit cycle:
at FINAL sizes with zero sensory input the rhythm generators sit at exactly
500 Hz and the motor pools fire every other step. The flexor drive then pins
at 1.0 regardless of the sampled weights, because the firing rate is limited
by the integration step rather than by the network. `ref_hz=700` was
calibrated at 0.2 ms, so the loop and the calibration must use the same step.
"""
import numpy as np

from model.adapter import CPGInput
from persona.parameter import reset_rng_stream
from persona.registry import Registry
from t2xcpg.bs import BaseDrive, BrainStem
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg


def _flexor_drive(resolution_ms, steps=40):
    import nest
    reset_rng_stream(0)
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": resolution_ms, "print_time": False,
                          "rng_seed": 12345})
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=10, bs_e_hz=60.0, bs_f_hz=60.0)
    base = BaseDrive(registry=reg, n_bs=10, base_hz=0.0)
    legs = {s: Leg(s, reg, n_rg=20, n_mn=20, n_int=10, n_aff=10, ref_hz=700.0)
            for s in "LR"}
    cpg = L2CPG(registry=reg, bs=bs, legs=legs, dt_ms=10.0, base=base)
    cpg_in = CPGInput()
    drive = []
    for _ in range(steps):
        cpg.update(cpg_in)
        drive.append(cpg.step().mus_F_L)
    return np.array(drive)


def test_production_resolution_is_the_reference_default():
    import run_validation
    assert run_validation.DEFAULT_RESOLUTION_MS == 0.2


def test_flexor_drive_stays_graded_at_0_2_ms(nest_kernel):
    drive = _flexor_drive(0.2)
    assert drive[-10:].max() < 0.99, (
        "flexor drive pinned at the production resolution; check the NEST "
        f"step (0.2 ms) and the network rates, got {drive[-10:]}")
