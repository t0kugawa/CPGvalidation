"""Group and recorder contracts on a real (small) NEURON kernel."""
import numpy as np
import pytest

pytest.importorskip("neuron")

from persona.connection import Connection
from persona.neuron_adapter import NeuronAdapter
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population


@pytest.fixture()
def reg():
    NeuronAdapter.reset_kernel(resolution_ms=0.05, v_init=-65.0)
    reset_rng_stream(0)
    return NeuronAdapter(seed=11)


def _pop(name, n=1):
    return Population("interneuron", n, {}, name, neuron={"n_synapses": 8})


def _exc(weight=5.0):
    return PersonaParam(weight, 0.01, 0.0, 20.0)


def test_rate_group_roundtrip_and_zero_safe(reg):
    rate = PersonaParam(500.0, 0.0, 0.0, 2000.0)
    pg, relay = reg.create_relay(
        "drive", "drive_in", 2, rate=500.0, rate_param=rate,
        pop_name="drive", noise_name="drive",
    )
    assert len(pg) == len(relay) == 2
    assert np.allclose(pg.get("rate"), 500.0)
    assert np.allclose(pg.get("interval"), 2.0)
    pg.set({"rate": 0.0})
    assert np.allclose(pg.get("rate"), 0.0)
    assert np.allclose(pg.get("number"), 0.0)
    pg.set({"rate": 250.0})
    assert np.allclose(pg.get("rate"), 250.0)
    pg.set({"interval": 8.0})
    assert np.allclose(pg.get("rate"), 125.0)


def test_spike_recorder_and_clear(reg):
    reg.create_population(_pop("src", 2))
    reg.create_population(_pop("tgt", 2))
    pg, relay = reg.create_relay("drive", "drive_in", 2, rate=800.0,
                                 pop_name="drive")
    reg.create_connection(Connection(
        "drive_in", "src", model="static_exc",
        parameters={"weight": _exc(10.0)}, rule="one_to_one",
        key="drive_e",
    ))
    reg.create_connection(Connection(
        "src", "tgt", model="static_exc",
        parameters={"weight": _exc(10.0)}, rule="one_to_one", key="e",
    ))
    rec = reg.create_spike_recorder(reg.get("src"))
    reg.initialize()
    reg.simulate(50.0)
    events = rec.get("events")
    assert events["times"].shape == events["senders"].shape
    assert events["times"].ndim == 1
    assert int(rec.get("n_events")) == events["times"].size
    rec.set({"n_events": 0})
    assert len(rec) == 0
    assert rec.get("events")["times"].size == 0


def test_weight_recorder_step_sampling_and_clear(reg):
    reg.create_population(_pop("src"))
    reg.create_population(_pop("tgt"))
    reg.create_connection(Connection(
        "src", "tgt", model="static_exc",
        parameters={"weight": _exc(5.0)}, rule="one_to_one", key="e",
    ))
    wr = reg.create_weight_recorder()
    wr.attach(reg.synapses("e")[0])
    reg.initialize()
    reg.simulate(20.0)
    assert wr.get("n_events") == 0
    reg.synapses("e").set({"weight": 4.0})
    reg.simulate(20.0)
    events = wr.get("events")
    assert events["weights"].tolist() == [4.0]
    assert events["times"].tolist() == [20.0]
    assert events["senders"].size == events["targets"].size == 1
    wr.set({"n_events": 0})
    assert len(wr) == 0
    assert wr.get("events")["weights"].size == 0


def test_relay_source_and_tap_target_with_recorders(reg):
    reg.create_population(_pop("src", 2))
    pg, relay = reg.create_relay("drive", "drive_in", 2, rate=600.0,
                                 pop_name="drive")
    reg.create_connection(Connection(
        "drive_in", "src", model="static_exc",
        parameters={"weight": _exc(10.0)}, rule="one_to_one",
        key="drive_e",
    ))
    tap = reg.create_parrot_nodes("tap", 2)
    assert tap.can_target and not tap.can_source
    assert relay.can_source and not relay.can_target
    reg.create_connection(Connection(
        "src", "tap", model="static_exc",
        parameters={"weight": _exc(10.0)}, rule="one_to_one",
        key="tap_e",
    ))
    src_rec = reg.create_spike_recorder(reg.get("src"))
    tap_rec = reg.create_spike_recorder(tap)
    reg.initialize()
    reg.simulate(60.0)
    assert src_rec.get("n_events") > 0
    assert tap_rec.get("n_events") > 0
