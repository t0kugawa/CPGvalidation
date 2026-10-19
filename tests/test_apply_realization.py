"""
Realization tests for `Registry.apply_explored_parameters` (spec §5.2):
every realized instance gets its own dispersion draw, and every failure is
surfaced in `last_apply_errors` instead of being swallowed.
"""
import numpy as np

from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.connection import Connection
from persona.registry import Registry


def _values(obj, key):
    """Reads one property of a node/synapse collection (tuple or dict)."""
    vals = obj.get(key)
    if isinstance(vals, dict):
        vals = vals[key]
    return np.asarray(vals, dtype=float)


def _tiny(nest_kernel):
    """Two 4-node Izhikevich populations plus an all-to-all static projection."""
    reset_rng_stream(0)
    reg = Registry()
    a = PersonaParam(mean=0.02, sigma=0.002, min=0.001, max=0.1)
    reg.create_population(Population("izhikevich", 4, {"a": a}, "src"))
    reg.create_population(Population("izhikevich", 4, {"a": a}, "tgt"))
    w = PersonaParam(mean=1.0, sigma=0.2, min=-150.0, max=150.0)
    reg.create_connection(Connection(
        source="src", target="tgt", model="static_synapse",
        parameters={"weight": w}, rule="all_to_all", p=1.0,
        extra_syn_spec={"delay": 1.0}, key="src_tgt"))
    reg.finalize_connections()
    return reg


def test_static_synapses_realize_per_synapse(nest_kernel):
    reg = _tiny(nest_kernel)
    reg.apply_explored_parameters({"conn_src_tgt_weight": 5.0})
    w = _values(reg.synapses("src_tgt"), "weight")
    assert w.size == 16
    assert np.unique(w).size == w.size               # one dispersion draw per synapse
    assert abs(w.mean() - 5.0) < 0.2                 # around the applied sample
    assert reg.site_recorder("conn_src_tgt_weight").latest_sample()[0] == 5.0


def test_poisson_device_realizes_per_node(nest_kernel):
    reg = Registry()
    p = PersonaParam(mean=100.0, sigma=20.0, min=0.0, max=1000.0)
    reg.create_relay("base_pg", "base_in", 6, rate=100.0, rate_param=p,
                     pop_name="base", noise_name="base")
    reg.apply_explored_parameters({"pop_base_rate": 200.0})
    rates = _values(reg.get("base_pg"), "rate")
    assert rates.size == 6
    assert np.unique(rates).size == 6                # per-node draws, not one scalar
    assert rates.min() >= 0.0 and rates.max() <= 1000.0
    assert abs(rates.mean() - 200.0) < 30.0
    assert reg.site_recorder("pop_base_rate").latest_sample()[0] == 200.0


def test_apply_failures_are_surfaced(nest_kernel):
    reg = _tiny(nest_kernel)

    class _Boom:
        def set(self, *args, **kwargs):
            raise RuntimeError("boom")

    reg.population_cfgs["src"].node_collection = _Boom()
    reg._synapses["src_tgt"] = _Boom()
    reg.apply_explored_parameters({"pop_src_a": 0.05, "conn_src_tgt_weight": 3.0})
    # both failures present, in order: earlier entries are not overwritten
    assert [k for k, _ in reg.last_apply_errors] == ["pop_src", "conn_src_tgt"]
    assert all(msg for _, msg in reg.last_apply_errors)
    assert any("boom" in msg for _, msg in reg.last_apply_errors)
    # a fresh call clears the list (each apply reports only its own failures)
    reg.apply_explored_parameters({})
    assert reg.last_apply_errors == []


def test_create_population_set_failure_is_surfaced(nest_kernel):
    reg = Registry()
    bad = PersonaParam(mean=5.0, sigma=1.0, min=0.0, max=50.0)
    # `rate` is not a property of izhikevich neurons: both realization paths
    # fail and the failure must be collected, not swallowed
    reg.create_population(Population("izhikevich", 3, {"rate": bad}, "bad_pop"))
    assert [k for k, _ in reg.last_apply_errors] == ["pop_bad_pop"]
    assert reg.last_apply_errors[0][1]
