"""NEURON adapter smoke tests (run in the WSL venv with NEURON installed)."""
import numpy as np
import pytest

pytest.importorskip("neuron")

from persona.connection import Connection
from persona.neuron_adapter import NeuronAdapter
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population


def _fresh(seed=1):
    NeuronAdapter.reset_kernel(resolution_ms=0.025, v_init=-65.0)
    reset_rng_stream(0)
    return NeuronAdapter(seed=seed)


def _population(name, n=2, model="interneuron", n_syn=8):
    return Population(
        model=model,
        n=n,
        params={
            "soma.gnabar_hh": PersonaParam(
                mean=0.25, sigma=0.01, min=0.0, max=1.0
            )
        },
        name=name,
        neuron={"n_synapses": n_syn},
    )


def test_kernel_advances_time():
    reg = _fresh()
    reg.initialize()
    t0 = reg.time_ms
    t1 = reg.simulate(1.0)
    assert abs(t1 - (t0 + 1.0)) < 0.05
    assert abs(reg.time_ms - t1) < 1e-9
    assert reg.dt_ms > 0.0


def test_static_exc_population_and_connection():
    reg = _fresh()
    src = reg.create_population(_population("src", 2))
    tgt = reg.create_population(_population("tgt", 2))
    conn = Connection(
        source="src",
        target="tgt",
        model="static_exc",
        parameters={
            "weight": PersonaParam(mean=2.0, sigma=0.1, min=0.0, max=5.0)
        },
        rule="all_to_all",
        key="e",
    )
    key = reg.create_connection(conn)
    assert key == "e"
    assert len(src) == 2 and len(tgt) == 2
    assert reg.synapse_count(key) == 4
    group = reg.synapses(key)
    assert group.model == "static_exc"
    assert np.allclose(group.get("weight"), 2.0)
    assert np.allclose(group.get("signed_weight"), 2.0)


def test_inhibitory_type_can_never_become_excitatory():
    reg = _fresh()
    reg.create_population(_population("src"))
    reg.create_population(_population("tgt"))
    p = PersonaParam(mean=4.0, sigma=0.1, min=-5.0, max=5.0)
    conn = Connection(
        source="src",
        target="tgt",
        model="static_inh",
        parameters={"weight": p},
        rule="all_to_all",
        key="i",
    )
    reg.create_connection(conn)
    # The special type wins over the positive magnitude: the site is signed
    # negative and the hardware conductance is positive.
    assert p.mean == -4.0
    assert p.max == 0.0
    assert np.allclose(reg.synapses("i").get("weight"), 4.0)
    assert np.allclose(reg.synapses("i").get("signed_weight"), -4.0)


def test_static_synapse_dispatches_on_sign():
    reg = _fresh()
    reg.create_population(_population("src"))
    reg.create_population(_population("tgt"))
    pos = Connection(
        source="src",
        target="tgt",
        model="static_synapse",
        parameters={
            "weight": PersonaParam(mean=1.0, sigma=0.01, min=-5.0, max=5.0)
        },
        rule="one_to_one",
        key="pos",
    )
    neg = Connection(
        source="src",
        target="tgt",
        model="static_synapse",
        parameters={
            "weight": PersonaParam(mean=-1.0, sigma=0.01, min=-5.0, max=5.0)
        },
        rule="one_to_one",
        key="neg",
    )
    reg.create_connection(pos)
    reg.create_connection(neg)
    assert reg.synapses("pos").model == "static_exc"
    assert reg.synapses("neg").model == "static_inh"
    assert np.allclose(reg.synapses("neg").get("weight"), 1.0)


def test_creation_only_synapse_values_are_applied():
    reg = _fresh()
    reg.create_population(_population("src"))
    reg.create_population(_population("tgt"))
    reg.create_connection(Connection(
        source="src", target="tgt", model="static_exc",
        parameters={"weight": PersonaParam(1.0, 0.01, 0.0, 5.0)},
        rule="one_to_one", key="exc_extra",
        extra_syn_spec={"tau": 5.0, "e": 10.0},
    ))
    assert np.allclose(reg.synapses("exc_extra").get("syn.tau"), 5.0)
    assert np.allclose(reg.synapses("exc_extra").get("syn.e"), 10.0)

    reg.create_connection(Connection(
        source="src", target="tgt", model="static_inh",
        parameters={"weight": PersonaParam(-1.0, 0.01, -5.0, 0.0)},
        rule="one_to_one", key="inh_extra",
        extra_syn_spec={"tau1": 1.0, "tau2": 3.0, "e": -80.0},
    ))
    assert np.allclose(reg.synapses("inh_extra").get("syn.tau1"), 1.0)
    assert np.allclose(reg.synapses("inh_extra").get("syn.tau2"), 3.0)
    assert np.allclose(reg.synapses("inh_extra").get("syn.e"), -80.0)


def test_weight_cv_never_flips_excitatory_or_inhibitory_identity():
    reg = _fresh()
    reg.create_population(_population("src", n=4))
    reg.create_population(_population("tgt", n=4))
    reg.create_connection(Connection(
        source="src", target="tgt", model="static_exc",
        parameters={"weight": PersonaParam(5.0, 0.1, 0.0, 20.0)},
        rule="all_to_all", key="e_cv", neuron={"weight_cv": 1.0},
    ))
    reg.create_connection(Connection(
        source="src", target="tgt", model="static_inh",
        parameters={"weight": PersonaParam(-5.0, 0.1, -20.0, 0.0)},
        rule="all_to_all", key="i_cv", neuron={"weight_cv": 1.0},
    ))
    assert np.all(reg.synapses("e_cv").get("signed_weight") >= 0.0)
    assert np.all(reg.synapses("i_cv").get("signed_weight") <= 0.0)
    assert np.all(reg.synapses("e_cv").get("weight") >= 0.0)
    assert np.all(reg.synapses("i_cv").get("weight") >= 0.0)


def test_static_synapse_zero_seed_is_ambiguous():
    reg = _fresh()
    reg.create_population(_population("src"))
    reg.create_population(_population("tgt"))
    conn = Connection(
        source="src",
        target="tgt",
        model="static_synapse",
        parameters={"weight": PersonaParam(mean=0.0, sigma=0.01)},
        rule="all_to_all",
        key="zero",
    )
    with pytest.raises(ValueError):
        reg.create_connection(conn)


def test_population_paths_and_apply_recorded_equals_applied():
    reg = _fresh()
    pop = _population("src")
    reg.create_population(pop)
    assert "soma.gnabar_hh" in reg.get("src").param_paths()
    # Creation realizes the biological dispersion around the seed mean.
    assert np.all(np.abs(reg.get("src").get("soma.gnabar_hh") - 0.25) < 0.1)
    key = "pop_src_soma.gnabar_hh"
    reg.apply_explored_parameters({key: 0.4})
    assert np.all(np.abs(reg.get("src").get("soma.gnabar_hh") - 0.4) < 0.1)
    sample, sigma = reg.site_recorder(key).latest_sample()
    assert sample == 0.4


def test_freeze_and_exact_key_resolution():
    reg = _fresh()
    pop = Population(
        "interneuron",
        2,
        {"dend[0].g_pas": PersonaParam(mean=0.0002, sigma=1e-5,
                                       min=0.0, max=1.0)},
        "src",
        neuron={"n_dend": 3, "n_synapses": 4},
    )
    reg.create_population(pop)
    key = "pop_src_dend[0].g_pas"
    assert reg.freeze(key) == (key,)
    assert reg.is_frozen(key, "mean")


def test_rate_relay_and_spike_recorder():
    reg = _fresh()
    rate = PersonaParam(mean=1000.0, sigma=10.0, min=0.0, max=2000.0)
    pg, relay = reg.create_relay(
        "drive", "drive_in", 2, rate=1000.0, rate_param=rate,
        pop_name="drive", noise_name="drive",
    )
    assert len(pg) == 2 and len(relay) == 2
    reg.create_population(_population("tgt", 2))
    reg.create_connection(
        Connection(
            source="drive_in", target="tgt", model="static_exc",
            parameters={"weight": PersonaParam(mean=2.0, sigma=0.1,
                                                min=0.0, max=5.0)},
            rule="one_to_one", key="drive_e",
        )
    )
    rec = reg.create_spike_recorder(reg.get("tgt"))
    reg.initialize()
    t0 = reg.time_ms
    reg.simulate(20.0)
    ev = rec.get("events")
    assert "times" in ev and "senders" in ev
    assert ev["times"].ndim == 1
    assert np.all(ev["times"] >= t0 - 1e-9)
    rec.set({"n_events": 0})
    assert len(rec) == 0
