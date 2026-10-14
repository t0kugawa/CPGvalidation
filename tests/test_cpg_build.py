from persona.registry import Registry
from t2xcpg.leg import Leg


def test_populations_registered(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    for name in ["rg_e_L", "rg_f_L", "m_e_L", "m_f_L", "r_e_L", "r_f_L",
                 "ia_e_L", "ia_f_L", "in_e_L", "in_f_L",
                 "v2a_L", "v0v_L", "v0d_L", "v3_f_L", "in1_L"]:
        assert reg.get(name) is not None


def test_afferent_generators_present(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    for name in ["cut_in_L", "ia_e_in_L", "ia_f_in_L", "ib_e_in_L", "ib_f_in_L"]:
        assert name in reg.population_nodes


def test_optimizable_params_nonempty(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    params = reg.get_all_optimizable_params()
    assert len(params) > 20
    assert all(k.startswith(("pop_", "conn_")) for k in params)


def test_zero_realized_connections_no_crash(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=4, n_mn=4, n_int=4, n_aff=4, p_scale=1e-6)
    assert len(reg.get_all_optimizable_params()) > 0