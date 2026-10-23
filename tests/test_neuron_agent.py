"""A minimal PERSONA learning loop on the NEURON adapter."""
import numpy as np
import pytest

pytest.importorskip("neuron")

from persona.agent import Agent
from persona.connection import Connection
from persona.neuron_adapter import NeuronAdapter
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population


def test_agent_runs_five_steps_on_neuron():
    NeuronAdapter.reset_kernel(resolution_ms=0.05, v_init=-65.0)
    reset_rng_stream(0)
    reg = NeuronAdapter(seed=1)
    reg.create_population(Population(
        "interneuron", 2,
        {"soma.gnabar_hh": PersonaParam(0.2, 0.01, 0.0, 1.0)},
        "src", neuron={"n_synapses": 4},
    ))
    reg.create_population(Population(
        "interneuron", 2,
        {"soma.gnabar_hh": PersonaParam(0.2, 0.01, 0.0, 1.0)},
        "tgt", neuron={"n_synapses": 4},
    ))
    reg.create_connection(Connection(
        "src", "tgt", model="static_exc",
        parameters={"weight": PersonaParam(1.0, 0.05, 0.0, 5.0)},
        rule="all_to_all", key="e",
    ))
    reg.finalize_connections()
    agent = Agent(reg, warmup=0, dt_ms=10.0)
    for _ in range(5):
        reg.apply_explored_parameters(agent.sample_actions())
        agent.step(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
    assert agent.current_step == 5
    assert len(agent.params) == 3
    assert np.isfinite(agent.params["conn_e_weight"].mean)
