"""Ib interneurons must inhibit the antagonist half-centre; mutual excitation is the defect, not a tuning choice."""
import pytest

from persona.registry import Registry
from t2xcpg.leg import Leg


@pytest.mark.parametrize("leg_name", ["L", "R"])
def test_ib_inhibits_antagonist_rg(nest_kernel, leg_name):
    reg = Registry()
    Leg(leg_name, reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)

    w = lambda k: reg.connection_cfgs[k].parameters["weight"].mean

    # Second hop of the Ib motif: excitatory Ib interneurons must not close a
    # positive loop onto the antagonist rhythm generator.
    for agonist, antagonist in (("e", "f"), ("f", "e")):
        assert w(f"in_ib_{agonist}_{leg_name}_rg_{antagonist}_{leg_name}") < 0.0

        # Guard against a future re-inversion: every in_ib_* projection onto
        # the antagonist RG population must be inhibitory.
        target = f"rg_{antagonist}_{leg_name}"
        for key, cfg in reg.connection_cfgs.items():
            if cfg.source.startswith(f"in_ib_{agonist}_{leg_name}") and cfg.target == target:
                assert cfg.parameters["weight"].mean < 0.0, key
