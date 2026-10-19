"""
Integration smoke (spec §8, plan T6): the full PERSONA stack on a tiny two-leg
CPG — registry realization with no silent failures, site + producer recorders,
the persisted F/X̄/G strategy and the sigma bounds.
"""
import numpy as np

from persona.agent import Agent
from persona.parameter import reset_rng_stream
from persona.registry import Registry
from model.adapter import CPGInput
from t2xcpg.bs import BrainStem, BaseDrive
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg as CPG_Leg

DT_MS = 10.0


def _build_tiny(nest_kernel):
    reset_rng_stream(0)
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=4, bs_e_hz=30.0, bs_f_hz=30.0)
    base = BaseDrive(registry=reg, n_bs=4, base_hz=2.0)
    leg_l = CPG_Leg("L", reg, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    leg_r = CPG_Leg("R", reg, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": leg_l, "R": leg_r},
                dt_ms=DT_MS, base=base)
    agent = Agent(registry=reg, dt_ms=DT_MS)
    return reg, cpg, agent


def test_stack_smoke(nest_kernel):
    reg, cpg, agent = _build_tiny(nest_kernel)
    rng = np.random.default_rng(0)
    cpg_in = CPGInput()
    target = np.array([0.2, -0.1])
    steps = 12

    for step in range(steps):
        # synthetic sensory drive, varying per step -> poisson producer deltas
        cpg_in.cut_L = 20.0 + 5.0 * float(rng.uniform())
        cpg_in.cut_R = 20.0 + 5.0 * float(rng.uniform())
        cpg_in.ia_rate_E_L = 60.0 + 20.0 * float(rng.uniform())
        cpg_in.ia_rate_F_L = 60.0 + 20.0 * float(rng.uniform())
        cpg_in.ia_rate_E_R = 60.0 + 20.0 * float(rng.uniform())
        cpg_in.ia_rate_F_R = 60.0 + 20.0 * float(rng.uniform())
        cpg_in.ib_rate_E_L = 10.0 + 5.0 * float(rng.uniform())
        cpg_in.ib_rate_F_L = 10.0 + 5.0 * float(rng.uniform())
        cpg_in.ib_rate_E_R = 10.0 + 5.0 * float(rng.uniform())
        cpg_in.ib_rate_F_R = 10.0 + 5.0 * float(rng.uniform())

        actions = agent.sample_actions(t_ms=step * DT_MS)
        reg.apply_explored_parameters(actions)
        cpg.update(cpg_in)
        cpg.step()  # STDP + sensory producers feed their recorders

        # synthetic tracking output: a varying error keeps advantage/credit alive
        output = target + 0.02 * np.sin(0.5 * step) + 0.01 * rng.standard_normal(2)
        phase = ["stance", "stance"] if step < steps // 2 else ["swing", "swing"]
        agent.step(output=output, target=target, phase=phase,
                   t_ms=(step + 1) * DT_MS)

    # realization never silently failed (spec §5.5); the only surfaced
    # anomalies are structurally-empty projections (zero-synapse sites)
    assert all("zero-synapse" in msg for _, msg in reg.last_apply_errors)

    # persisted metric/credit: finite, symmetric, alive
    F = agent.strategy.F
    assert F.shape == (F.shape[0], F.shape[0])
    assert np.all(np.isfinite(F))
    assert np.allclose(F, F.T, atol=1e-12)
    assert np.any(F != 0.0)
    assert np.all(np.isfinite(agent.strategy.Xbar))
    assert np.any(agent.strategy.G != 0.0)

    # every site was sampled/applied; sigma is finite and within its bounds
    params = reg.get_all_optimizable_params()
    assert params
    for name, p in params.items():
        sample, sigma = reg.site_recorder(name).latest_sample()
        assert np.isfinite(sample), name
        assert np.isfinite(sigma), name
        assert p.sigma_min <= p.sigma <= p.sigma_max, name

    # every producer saw the run (6 STDP projections + 10 sensory relays)
    producers = reg.producer_recorders()
    assert len(producers) == 16
    for name, rec in producers.items():
        assert rec.stats()["n"] > 0, name
