import numpy as np

from persona.parameter import PersonaParam
from persona.registry import Registry
from t2xcpg.leg import Leg


def test_site_recorder_per_site_unique_across_legs(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=3, n_mn=3, n_int=3, n_aff=3)
    Leg("R", reg, n_rg=3, n_mn=3, n_int=3, n_aff=3)
    params = reg.get_all_optimizable_params()
    for key in params:
        rec = reg.site_recorder(key)
        assert rec.name == f"site:{key}"
    # one recorder instance per site, and each carries its own sample state
    recs = {k: reg.site_recorder(k) for k in params}
    assert len({id(r) for r in recs.values()}) == len(params)


def test_site_recorder_holds_last_applied_sample(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=4, n_mn=4, n_int=4, n_aff=4)
    reg.apply_explored_parameters({"pop_rg_e_L_a": 0.05})
    assert reg.site_recorder("pop_rg_e_L_a").latest_sample()[0] == 0.05
    # untouched sites keep their seed sample
    sample, sigma = reg.site_recorder("pop_rg_f_L_a").latest_sample()
    assert np.isfinite(sample) and sigma > 0.0


def test_stdp_producer_recorders_are_per_projection(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    Leg("R", reg, n_rg=6, n_mn=6, n_int=5, n_aff=5)
    prods = set(reg.producer_recorders())
    for key in leg.plastic_projections:
        assert f"producer:stdp:{key}" in prods
    # no L/R collisions between legs
    assert "producer:stdp:cut_rge_L" in prods and "producer:stdp:cut_rge_R" in prods
    assert "producer:poisson:cut_L" in prods and "producer:poisson:cut_R" in prods


def test_relay_applies_rate_and_records_delta(nest_kernel):
    reg = Registry()
    reg.create_relay("pg", "in", 4, rate=0.0, noise_name="cut_L")
    reg.set_relay_rate("pg", 12.0)
    rec = reg.producer_recorder("poisson:cut_L")
    assert np.isclose(rec.take_delta(), 12.0)          # 0 -> 12 Hz
    assert rec.take_delta() is None


def test_relay_rate_param_is_optimizable(nest_kernel):
    reg = Registry()
    p = PersonaParam(mean=5.0, sigma=1.0, min=0.0, max=50.0)
    reg.create_relay("bs_pg_e", "BS_in_e", 4, rate=5.0, rate_param=p, pop_name="BS_E")
    assert reg.get("bs_pg_e") is not None and reg.get("BS_in_e") is not None
    assert "pop_BS_E_rate" in reg.get_all_optimizable_params()
    assert "site:pop_BS_E_rate" in reg.noise_recorders
