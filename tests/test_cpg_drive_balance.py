"""The tonic projections onto the rhythm generators must not saturate them.

Measured before the fix (100-neuron RG pool): the brainstem projection alone held
RG-E at 243 Hz and the cutaneous projection alone at 146 Hz, so `rg_e_L` read a
flat 1.0 for a whole trial and the muscle drive was pinned at 1.0.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from persona.parameter import reset_rng_stream
from persona.registry import Registry
from model.adapter import CPGInput
from t2xcpg.bs import BrainStem, BaseDrive
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg

DT_MS = 10.0
REF_HZ = 100.0
FAST = dict(n_rg=6, n_mn=5, n_int=5, n_aff=3, n_bs=5)


def _run(cut=0.0, ia=0.0, ib=0.0, bs_hz=40.0, steps=60, seed=0):
    """Build a two-leg CPG and return the per-step RG rate (units of ref_hz)."""
    import nest
    reset_rng_stream(seed)
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=FAST["n_bs"], bs_e_hz=bs_hz,
                   bs_f_hz=bs_hz)
    base = BaseDrive(registry=reg, n_bs=FAST["n_bs"], base_hz=2.0)
    legs = {s: Leg(s, reg, n_rg=FAST["n_rg"], n_mn=FAST["n_mn"],
                   n_int=FAST["n_int"], n_aff=FAST["n_aff"], ref_hz=REF_HZ)
            for s in "LR"}
    cpg = L2CPG(registry=reg, bs=bs, legs=legs, dt_ms=DT_MS, base=base)

    cpg_in = CPGInput()
    for side in "LR":
        setattr(cpg_in, f"cut_{side}", cut)
        for m in "EF":
            setattr(cpg_in, f"ia_rate_{m}_{side}", ia)
            setattr(cpg_in, f"ib_rate_{m}_{side}", ib)

    rates = []
    for _ in range(steps):
        cpg.update(cpg_in)
        out = cpg.step()
        rates.append(out.extra["rg"]["rg_e_L"])
    return np.asarray(rates, dtype=float)


def test_brainstem_alone_does_not_saturate_the_rhythm_generators(nest_kernel):
    rates = _run(cut=0.0, ia=0.0, ib=0.0, bs_hz=40.0)
    # ref_hz=100 -> 0.6 is 60 Hz. The RG must be driven, but not pinned.
    assert rates.mean() < 0.6, f"BS alone pins the RG at {rates.mean() * REF_HZ:.0f} Hz"
