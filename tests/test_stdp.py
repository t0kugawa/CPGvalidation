import numpy as np

from persona.registry import Registry
from t2xcpg.leg import Leg


def test_plastic_projections_exist(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    assert any("cut" in k and "rge" in k for k in leg.plastic_projections)
    assert any("ia" in k for k in leg.plastic_projections), \
        "Ia->RG must be a plastic (STDP) projection (spec D12)"
    for k in leg.plastic_projections:
        assert reg.connection_cfgs[k].model.startswith("stdp")


def test_plastic_params_are_hyperparams_not_weight(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    params = reg.get_all_optimizable_params()
    k = leg.plastic_projections[0]
    assert f"conn_{k}_weight" not in params, "STDP weight must not be optimized"
    assert f"conn_{k}_Wmax" in params, "STDP hyper-parameters must be optimized"


def test_weight_delta_sampled(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    deltas = reg.sample_plastic_weight_deltas(0.0, 10.0)
    assert isinstance(deltas, dict)
    assert len(deltas) > 0
    assert all(np.isfinite(v) for v in deltas.values())


def test_plastic_delta_is_per_synapse_increment():
    from persona.registry import plastic_delta_from_events

    events = {"senders": np.array([1, 1, 2]), "targets": np.array([5, 5, 6]),
              "weights": np.array([11.0, 12.0, 9.0])}
    last = {(1, 5): 10.5}                     # synapse (1,5) seen before; (2,6) first sight
    delta, last2 = plastic_delta_from_events(events, last, seed_weight=10.0, n_syn=4)
    # (11-10.5) + (12-11) + (9-10 seed)
    assert np.isclose(delta, (0.5 + 1.0 - 1.0) / 4)
    assert last2[(1, 5)] == 12.0 and last2[(2, 6)] == 9.0


def test_plastic_delta_uses_seed_for_first_sighting():
    from persona.registry import plastic_delta_from_events

    events = {"senders": np.array([7]), "targets": np.array([9]),
              "weights": np.array([13.0])}
    delta, _ = plastic_delta_from_events(events, {}, seed_weight=14.0, n_syn=2)
    assert np.isclose(delta, -0.5)            # (13 - 14 seed) / N=2


def test_stdp_producer_recorders_are_wired_per_projection(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    for key in leg.plastic_projections:
        assert f"producer:stdp:{key}" in reg.producer_recorders()


def test_sample_plastic_deltas_feed_recorders_read_and_clear(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    key = leg.plastic_projections[0]
    seed = float(reg.connection_cfgs[key].extra_syn_spec["weight"])
    reg._plastic_n_syn[key] = 4               # N_syn is cached at finalize; pin it here
    reg.read_plastic_events = lambda clear=True: {
        key: {"times": np.array([10.0, 20.0]),
              "weights": np.array([seed + 1.0, seed + 3.0]),
              "senders": np.array([1, 1]),
              "targets": np.array([2, 2])}}
    deltas = reg.sample_plastic_weight_deltas(0.0, 10.0)
    assert np.isclose(deltas[key], 3.0 / 4)   # (seed+1 - seed) + (seed+3 - (seed+1))
    rec = reg.producer_recorder(f"stdp:{key}")
    assert np.isclose(rec.take_delta(), 3.0 / 4)
    assert rec.take_delta() is None



def test_weight_recorder_is_drained_each_read(nest_kernel):
    import nest

    reg = Registry()
    Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    nest.Simulate(50.0)

    first = reg.read_plastic_events(clear=True)
    total = sum(int(ev["weights"].size) for ev in first.values())
    assert total > 0, "STDP must have produced weight-recorder events"
    for key, cfg in reg.connection_cfgs.items():
        if cfg.weight_recorder is not None:
            assert cfg.weight_recorder.get("events")["weights"].size == 0
    second = reg.read_plastic_events(clear=True)
    assert sum(int(ev["weights"].size) for ev in second.values()) == 0


def test_plastic_weight_stats_bracket_the_mean(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    stats = reg.sample_plastic_weight_stats()
    assert stats
    for key, s in stats.items():
        assert s["min"] <= s["mean"] <= s["max"]
    final = reg.final_plastic_weights()
    assert set(final.keys()) == set(stats.keys())


def test_deltas_are_shared_within_one_step(nest_kernel):
    """Both limbs read the same drained batch, so the second limb must not see zeros."""
    reg = Registry()
    Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    Leg("R", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    import nest

    nest.Simulate(50.0)
    a = reg.sample_plastic_weight_deltas(0.0, 50.0)
    b = reg.sample_plastic_weight_deltas(0.0, 50.0)
    assert a == b
    left = [k for k in a if k.endswith("_L")]
    assert len(left) > 0
    # the shared batch is recorded exactly once, so it is taken once
    rec = reg.producer_recorder(f"stdp:{left[0]}")
    assert rec.take_delta() is not None
    assert rec.take_delta() is None
