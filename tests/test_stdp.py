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


def test_recorded_weight_delta_is_not_differenced_twice(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    key = leg.plastic_projections[0]
    sampled = {k: (0.5 if k == key else 0.0) for k in leg.plastic_projections}
    reg.sample_plastic_weight_deltas = lambda a, b: dict(sampled)

    leg.update_plastic_weights(0.0, 10.0)
    leg.update_plastic_weights(10.0, 20.0)

    vals = [s.value for s in leg.noise_recorder.samples if s.producer_id == key]
    assert vals[-2:] == [0.5, 0.5]
    assert np.isclose(leg.noise_recorder.get_latest_step_deltas()[f"noise_{key}_weight"], 0.5)
