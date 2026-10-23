"""NEST-thread integration: accuracy, reproducibility, and bounded speed.

These tests need the real NEST simulator and are skipped when it is absent.
The cross-thread contract for NEST is weaker than for PERSONA: a fixed thread
count is reproducible, but different thread counts may realize the network
with different RNG streams. PERSONA samples, by contrast, are identical for
every thread count (see test_parallel_sampling.py).
"""
import os
import time

import pytest

nest = pytest.importorskip("nest")

from persona.agent import Agent
from persona.parameter import reset_rng_stream
from persona.registry import Registry
from persona.runtime import configure_kernel
from t2xcpg.bs import BaseDrive, BrainStem
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg as CPG_Leg

DT_MS = 10.0


def _build(threads: int, *, seed: int = 0,
           n_rg: int = 6, n_mn: int = 6, n_int: int = 6, n_aff: int = 6):
    configure_kernel(nest, resolution_ms=0.2, seed=seed, threads=threads)
    reset_rng_stream(seed)
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=4, bs_e_hz=30.0, bs_f_hz=30.0)
    base = BaseDrive(registry=reg, n_bs=4, base_hz=2.0)
    leg_l = CPG_Leg("L", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff)
    leg_r = CPG_Leg("R", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": leg_l, "R": leg_r},
                dt_ms=DT_MS, base=base)
    return reg, cpg


def test_kernel_reports_requested_threads():
    _build(4, n_rg=2, n_mn=2, n_int=2, n_aff=2)
    assert int(nest.GetKernelStatus("local_num_threads")) == 4


def test_thread_count_keeps_site_and_node_counts():
    reg_one, _ = _build(1, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    sites_one = sorted(reg_one.get_all_optimizable_params())
    nodes_one = len(reg_one.population_nodes)

    reg_two, _ = _build(2, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    assert sorted(reg_two.get_all_optimizable_params()) == sites_one
    assert len(reg_two.population_nodes) == nodes_one


def test_persona_sampling_is_identical_on_real_registry():
    reg_one, _ = _build(1, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    agent_one = Agent(registry=reg_one, sampling_threads=1)

    reg_four, _ = _build(4, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    agent_four = Agent(registry=reg_four, sampling_threads=4)

    try:
        sequential = [agent_one.sample_actions(t_ms=t * DT_MS)
                      for t in range(8)]
        parallel = [agent_four.sample_actions(t_ms=t * DT_MS)
                    for t in range(8)]
    finally:
        agent_one.close()
        agent_four.close()

    assert sequential == parallel
    assert agent_one._snap == agent_four._snap


def _run_steps(threads: int, steps: int = 5):
    _, cpg = _build(threads, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    return [cpg.step() for _ in range(steps)]


def _signature(outputs):
    return [(round(o.mus_E_L, 12), round(o.mus_F_L, 12),
             round(o.mus_E_R, 12), round(o.mus_F_R, 12))
            for o in outputs]


def test_fixed_thread_count_is_reproducible():
    assert _signature(_run_steps(2)) == _signature(_run_steps(2))


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs >= 2 cores")
def test_threaded_nest_speed_is_bounded(capsys):
    """Two NEST threads must not make a run pathologically slower."""
    steps = 10
    sizes = {"n_rg": 10, "n_mn": 10, "n_int": 8, "n_aff": 8}

    _, cpg_one = _build(1, **sizes)
    start = time.perf_counter()
    for _ in range(steps):
        cpg_one.step()
    single = time.perf_counter() - start

    _, cpg_two = _build(2, **sizes)
    start = time.perf_counter()
    for _ in range(steps):
        cpg_two.step()
    multi = time.perf_counter() - start

    with capsys.disabled():
        print(f"\n[NEST speed] steps={steps} 1-thread={single * 1e3:.1f} ms "
              f"2-thread={multi * 1e3:.1f} ms ratio={multi / single:.2f}x")

    assert multi <= single * 5.0
