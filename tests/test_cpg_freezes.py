"""Structural CPG parameters are pinned through the native PERSONA registry.

PERSONA's homeostatic update drove the Renshaw recurrent inhibition and the M-F
Izhikevich parameters away from reference until the flexor pool went silent.
These parameters stay in the network with their normal bounds, but the registry
freeze holds their learned attributes, so the optimizer no longer maximises
them.
"""
import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import reset_rng_stream
from persona.registry import Registry
from t2xcpg.leg import Leg, M_F_IZH_REF

RENSHAW = {
    ("m_e", "r_e"): 3.0,
    ("r_e", "m_e"): -4.0,
    ("r_e", "ia_e"): -2.4,
    ("m_f", "r_f"): 3.0,
    ("r_f", "m_f"): -4.0,
    ("r_f", "ia_f"): -2.4,
}


def _leg(side="L"):
    reset_rng_stream(0)
    return Leg(side, Registry(), n_rg=8, n_mn=8, n_int=6, n_aff=6)


def _renshaw_key(src, tgt):
    return f"conn_{src}_L_{tgt}_L_weight"


def test_renshaw_weights_are_frozen_on_mean(nest_kernel):
    leg = _leg()
    for (src, tgt), value in RENSHAW.items():
        key = _renshaw_key(src, tgt)
        p = leg.registry.connection_cfgs[f"{src}_L_{tgt}_L"].parameters["weight"]
        assert leg.registry.is_frozen(key, "mean")
        assert not leg.registry.is_frozen(key, "sigma")
        assert p.mean == pytest.approx(value)
        # The native freeze holds the mean; it does not collapse the bounds.
        assert p.min < p.max
        assert p.sample_exploration() != pytest.approx(value)


def test_mf_izh_is_frozen_on_mean_and_sigma(nest_kernel):
    leg = _leg()
    for name, value in M_F_IZH_REF.items():
        key = f"pop_m_f_L_{name}"
        p = leg.registry.population_cfgs["m_f_L"].params[name]
        assert leg.registry.is_frozen(key, "mean")
        assert leg.registry.is_frozen(key, "sigma")
        assert p.mean == pytest.approx(value)
        assert p.sigma == pytest.approx(0.0)
        # The post-creation sigma=0 mutation is the captured anchor, and the
        # bounds keep their normal Izhikevich width.
        assert leg.registry.defaults(key)["mean"] == pytest.approx(value)
        assert leg.registry.defaults(key)["sigma"] == pytest.approx(0.0)
        assert p.min < p.max


def _mf_drive_keys(side="L"):
    return [f"conn_rg_f_{side}_m_f_{side}_weight",
            f"conn_ia_f_in_{side}_m_f_{side}_weight"]


@pytest.mark.parametrize("side", ("L", "R"))
def test_mf_excitatory_drive_is_seeded_above_reference(nest_kernel, side):
    """C1: seed the flexor pool's excitatory inputs at 4x the port reference.

    At the reduced population sizes the RG-F rate is a fraction of FINAL; with
    the 2.75/12 seed PERSONA suppressed the R inputs further and musF_R fell to
    exactly 0. The seed stays optimizable, so FINAL can adapt it down.
    """
    leg = _leg(side)
    for key, value in zip(_mf_drive_keys(side), (11.0, 48.0)):
        p = leg.registry.get_all_optimizable_params()[key]
        assert not leg.registry.is_frozen(key, "mean"), key
        assert p.mean == pytest.approx(value)
        assert p.min < p.max
    # Explicit 4x override of the cpg_2legs_fast/port reference values.
    assert 11.0 == pytest.approx(4.0 * 2.75)
    assert 48.0 == pytest.approx(4.0 * 12.0)


def test_agent_step_cannot_move_the_frozen_sites(nest_kernel):
    leg = _leg()
    agent = Agent(registry=leg.registry, lr_mu=0.5, lr_sigma=0.5, warmup=0,
                  dt_ms=10.0, tau_mem_s=0.2)
    renshaw_keys = [_renshaw_key(src, tgt) for src, tgt in RENSHAW]
    mf_keys = [f"pop_m_f_L_{name}" for name in M_F_IZH_REF]
    before = {
        key: (agent.params[key].mean, agent.params[key].sigma)
        for key in renshaw_keys + mf_keys
    }
    for t in range(20):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * t,
                   phase="stance", t_ms=t * 10.0)
    for key in renshaw_keys:
        # Mean is pinned; sigma is not part of the Renshaw freeze.
        assert agent.params[key].mean == pytest.approx(before[key][0])
    for key in mf_keys:
        assert (agent.params[key].mean, agent.params[key].sigma) == (
            pytest.approx(before[key][0]), pytest.approx(before[key][1]))


def test_freezes_can_be_disabled_for_ab(nest_kernel, monkeypatch):
    monkeypatch.setenv("CPG_RENSHAW_CLAMP", "0")
    monkeypatch.setenv("CPG_MF_IZH_FREEZE", "0")
    leg = _leg()
    for src, tgt in RENSHAW:
        assert leg.registry.frozen_attrs(_renshaw_key(src, tgt)) == frozenset()
    for name in M_F_IZH_REF:
        assert leg.registry.frozen_attrs(f"pop_m_f_L_{name}") == frozenset()
    p = leg.registry.connection_cfgs["r_f_L_m_f_L"].parameters["weight"]
    assert p.min < p.max
    p = leg.registry.population_cfgs["m_f_L"].params["d"]
    assert p.min < p.max
    assert p.sigma > 0.0
