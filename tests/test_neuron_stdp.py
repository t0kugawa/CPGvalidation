"""STDP path tests; skipped unless the reference ``stdp.mod`` is compiled."""
import numpy as np
import pytest

pytest.importorskip("neuron")
from pathlib import Path

from neuron import h

# Try the documented in-repo compile location before declaring the test
# unavailable; the adapter also auto-loads this path at creation time.
_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (
    _ROOT / "memcpg" / "mod_files" / "x86_64" / "libnrnmech.so",
    _ROOT / "memcpg" / "mod_files" / "x86_64" / "libnrnmech.dylib",
):
    if _candidate.exists():
        try:
            h.nrn_load_dll(str(_candidate))
        except Exception:
            pass
        break

from persona.connection import Connection
from persona.neuron_adapter import NeuronAdapter
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population


def test_stdp_projection_records_weight_changes():
    if not hasattr(h, "STDP"):
        pytest.skip(
            "NEURON mechanism 'STDP' is not loaded; compile "
            "memcpg/mod_files/stdp.mod with nrnivmodl and load the library")
    NeuronAdapter.reset_kernel(resolution_ms=0.05, v_init=-65.0)
    reset_rng_stream(0)
    reg = NeuronAdapter(seed=3)
    reg.create_population(Population("interneuron", 1, {}, "src",
                                     neuron={"n_synapses": 8}))
    reg.create_population(Population("interneuron", 1, {}, "tgt",
                                     neuron={"n_synapses": 8}))
    reg.create_relay("drive", "drive_in", 1, rate=800.0, pop_name="drive")
    reg.create_connection(Connection(
        "drive_in", "src", model="static_exc",
        parameters={"weight": PersonaParam(10.0, 0.1, 0.0, 20.0)},
        rule="one_to_one", key="drive_e",
    ))
    reg.create_connection(Connection(
        "drive_in", "tgt", model="static_exc",
        parameters={"weight": PersonaParam(12.0, 0.1, 0.0, 20.0)},
        rule="one_to_one", key="drive_t",
    ))
    wr = reg.create_weight_recorder()
    reg.create_connection(Connection(
        "src", "tgt", model="stdp_exc",
        parameters={
            "wmax": PersonaParam(15.0, 0.0, 0.0, 15.0),
            "tauhebb": PersonaParam(20.0, 0.0, 1.0, 100.0),
            "tauanti": PersonaParam(20.0, 0.0, 1.0, 100.0),
            "hebbwt": PersonaParam(0.01, 0.0, 0.0, 0.1),
            "antiwt": PersonaParam(0.01, 0.0, 0.0, 0.1),
        },
        rule="one_to_one", key="stdp", weight_recorder=wr,
        extra_syn_spec={"weight": 0.5},
    ))
    reg.initialize()
    for _ in range(20):
        reg.simulate(10.0)
    assert reg.synapses("stdp").get("weight")[0] > 0.5
    ev = wr.get("events")
    assert ev["times"].size > 0
    assert ev["weights"].size == ev["senders"].size == ev["targets"].size
    assert np.all(ev["weights"] <= 15.0)
    assert "stdp" in reg.sample_plastic_weight_stats()
    assert "stdp" in reg.final_plastic_weights()
