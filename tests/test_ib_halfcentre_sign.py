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


@pytest.mark.parametrize("leg_name", ["L", "R"])
def test_rg_does_not_drive_ib_interneurons(nest_kernel, leg_name):
    """The RG->Ib hop is project-authored and held the network at the step cap."""
    reg = Registry()
    Leg(leg_name, reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)

    rg_pops = {f"rg_e_{leg_name}", f"rg_f_{leg_name}"}
    ib_pops = {f"in_ib_e_{leg_name}", f"in_ib_f_{leg_name}"}

    # Guard: no rhythm-generator population may drive an Ib interneuron. Every
    # nonzero weight (1 to 8) of this projection pinned RG at 1.7-2.5 kHz.
    for key, cfg in reg.connection_cfgs.items():
        if cfg.source in rg_pops and cfg.target in ib_pops:
            raise AssertionError(
                f"RG -> Ib drive still present: {key} "
                f"(weight={cfg.parameters['weight'].mean})"
            )

    # Guard the other way: the Ib reflex arc must survive the removal, so a
    # future edit cannot delete the pathway instead of the drive.
    def count(src_prefix: str, tgt_prefix: str) -> int:
        return sum(
            1
            for cfg in reg.connection_cfgs.values()
            if cfg.source.startswith(src_prefix)
            and cfg.target.startswith(tgt_prefix)
        )

    for ib, mn, rg, aff in (
        ("e", "m_e", "rg_f", "ib_e_in"),
        ("f", "m_f", "rg_e", "ib_f_in"),
    ):
        for src_prefix, tgt_prefix, label in (
            (aff, f"in_ib_{ib}_{leg_name}", "GTO -> InIb"),
            (f"ia_{ib}_in_{leg_name}", f"in_ib_{ib}_{leg_name}", "Ia -> InIb"),
            (f"in_ib_{ib}_{leg_name}", f"{mn}_{leg_name}", "InIb -> MN"),
            (f"in_ib_{ib}_{leg_name}", f"{rg}_{leg_name}", "InIb -> antagonist RG"),
        ):
            assert count(src_prefix, tgt_prefix) > 0, label
