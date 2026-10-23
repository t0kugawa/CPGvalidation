"""NEST integration for N independent CPG replicas in one kernel.

Skipped when NEST is not installed. The cross-replica contract is now:
samples and SNN outputs are intentionally different; a fixed
(seed, thread count) is reproducible.
"""
import os
import time

import numpy as np
import pytest

nest = pytest.importorskip("nest")

from model.adapter import CPGInput
from persona.multi_agent import MultiAgent
from persona.parameter import reset_rng_stream
from persona.registry import Registry
from persona.runtime import configure_kernel
from t2xcpg.bs import BaseDrive, BrainStem
from t2xcpg.cpg import L2CPG, MultiCPG
from t2xcpg.leg import Leg as CPG_Leg

DT_MS = 10.0


def _build(n: int = 2, seed: int = 0,
           n_rg: int = 6, n_mn: int = 6, n_int: int = 6, n_aff: int = 6):
    configure_kernel(nest, resolution_ms=0.2, seed=seed, threads=n)
    reset_rng_stream(seed)
    registries, cpgs, brainstems = [], [], []
    for i in range(n):
        reg = Registry(namespace=f"r{i}_")
        bs = BrainStem(registry=reg, n_bs=4, bs_e_hz=30.0, bs_f_hz=30.0)
        base = BaseDrive(registry=reg, n_bs=4, base_hz=2.0)
        leg_l = CPG_Leg("L", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int,
                        n_aff=n_aff)
        leg_r = CPG_Leg("R", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int,
                        n_aff=n_aff)
        cpgs.append(L2CPG(registry=reg, bs=bs,
                          legs={"L": leg_l, "R": leg_r},
                          dt_ms=DT_MS, base=base))
        registries.append(reg)
        brainstems.append(bs)
    return registries, cpgs, brainstems, MultiCPG(cpgs)


def test_kernel_reports_requested_threads():
    _build(3, n_rg=2, n_mn=2, n_int=2, n_aff=2)
    assert int(nest.GetKernelStatus("local_num_threads")) == 3


def test_replicas_scale_nodes_and_sites():
    regs_one, _, _, _ = _build(1, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    nodes_one = len(regs_one[0].population_nodes)
    sites_one = sorted(regs_one[0].get_all_optimizable_params())

    regs_two, _, _, _ = _build(2, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    assert len(regs_two) == 2
    for reg in regs_two:
        assert len(reg.population_nodes) == nodes_one
        assert sorted(reg.get_all_optimizable_params()) == sites_one


def test_one_simulate_advances_every_replica():
    _, cpgs, _, multi = _build(2, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    multi.update([CPGInput(), CPGInput()])
    outputs = multi.step()

    assert len(outputs) == 2
    assert multi.current_time_ms == DT_MS
    assert all(cpg.current_time_ms == DT_MS for cpg in cpgs)
    assert float(nest.GetKernelStatus("biological_time")) == DT_MS


def test_different_drive_produces_different_outputs():
    _, _, drivers, multi = _build(
        2, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    drivers[0].set_rates(10.0, 10.0)
    drivers[1].set_rates(50.0, 50.0)

    totals = [0.0, 0.0]
    for _ in range(20):
        multi.update([CPGInput(), CPGInput()])
        outputs = multi.step()
        totals[0] += outputs[0].mus_E_L + outputs[0].mus_F_L
        totals[1] += outputs[1].mus_E_L + outputs[1].mus_F_L

    assert totals[0] != totals[1]


def test_persona_samples_differ_and_merge_on_phase_change():
    regs, _, _, multi = _build(2, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    agent = MultiAgent(regs, lr_mu=0.1, lr_sigma=0.01, warmup=0)
    actions = agent.sample_actions()
    assert actions[0] != actions[1]

    agent.apply_actions(actions)
    multi.update([CPGInput(), CPGInput()])
    outputs = multi.step()

    first = agent.step(
        outputs=[np.array([o.mus_E_L, o.mus_F_L]) for o in outputs],
        targets=[np.zeros(2) for _ in outputs],
        ref_phases=[("stance", "stance") for _ in outputs],
    )
    assert first["merged"] is False

    actions = agent.sample_actions()
    agent.apply_actions(actions)
    multi.update([CPGInput(), CPGInput()])
    outputs = multi.step()
    second = agent.step(
        outputs=[np.array([o.mus_E_L, o.mus_F_L]) for o in outputs],
        targets=[np.zeros(2) for _ in outputs],
        ref_phases=[("swing", "swing") for _ in outputs],
    )
    assert second["merged"] is True
    assert np.allclose(agent.replica_strategies[0].F, agent.strategy.F)


def _signature(n: int = 2, steps: int = 5):
    _, cpgs, _, multi = _build(
        n, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    out = []
    for _ in range(steps):
        multi.update([CPGInput() for _ in cpgs])
        outputs = multi.step()
        out.extend(
            (round(o.mus_E_L, 12), round(o.mus_F_L, 12),
             round(o.mus_E_R, 12), round(o.mus_F_R, 12))
            for o in outputs
        )
    return out


def test_fixed_thread_count_is_reproducible():
    assert _signature(2) == _signature(2)


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >= 2 cores")
def test_multi_replica_speed_is_bounded(capsys):
    def timed(n: int, steps: int = 10) -> float:
        _, cpgs, _, multi = _build(
            n, n_rg=10, n_mn=10, n_int=8, n_aff=8)
        start = time.perf_counter()
        for _ in range(steps):
            multi.update([CPGInput() for _ in cpgs])
            multi.step()
        return time.perf_counter() - start

    single = timed(1)
    two = timed(2)
    per_sample = two / 2.0
    with capsys.disabled():
        print(f"\n[NEST replicas] 1 replica={single * 1e3:.1f} ms "
              f"2 replicas={two * 1e3:.1f} ms "
              f"per-sample ratio={per_sample / single:.2f}x")

    assert per_sample <= single * 3.0
