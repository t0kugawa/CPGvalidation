"""The common adapter contract: one pure ABC, two concrete files.

These tests intentionally run without NEURON/NEST-specific behaviour. The
NEURON import is skipped when the package is unavailable; the real NEURON
suite runs in the WSL environment.
"""
import inspect
import importlib

import pytest

from persona.adapter import (
    Adapter,
    clamp_weight_sign,
    enforce_synaptic_sign,
    plastic_delta_from_events,
    resolve_site_keys,
)
from persona.nest_adapter import NestAdapter
from persona.parameter import PersonaParam
from persona.registry import Registry


COMMON_METHODS = {
    "reset_kernel",
    "set_resolution",
    "initialize",
    "simulate",
    "register_noise_recorder",
    "get_noise_recorder",
    "site_recorder",
    "producer_recorder",
    "producer_recorders",
    "recapture_defaults",
    "freeze",
    "unfreeze",
    "frozen_attrs",
    "is_frozen",
    "defaults",
    "create_spring",
    "remove_spring",
    "springs",
    "spring_state",
    "register_nodes",
    "create_parrot_nodes",
    "create_spike_recorder",
    "create_weight_recorder",
    "create_population",
    "create_connection",
    "create_relay",
    "set_relay_rate",
    "synapses",
    "synapse_count",
    "finalize_connections",
    "read_plastic_events",
    "sample_plastic_weight_deltas",
    "last_plastic_counts",
    "sample_plastic_weight_stats",
    "final_plastic_weights",
    "get",
    "get_population_cfg",
    "get_all_optimizable_params",
    "sample_all_actions",
    "apply_explored_parameters",
    "time_ms",
    "dt_ms",
}


def test_adapter_is_a_pure_abstract_interface():
    assert inspect.isabstract(Adapter)
    assert COMMON_METHODS <= set(Adapter.__abstractmethods__)
    source = inspect.getsource(inspect.getmodule(Adapter))
    assert "import nest" not in source
    assert "import neuron" not in source
    assert "from neuron" not in source


def test_concrete_adapters_live_in_separate_files():
    pytest.importorskip("neuron")
    assert NestAdapter.__module__ == "persona.nest_adapter"
    assert "persona.neuron_adapter" == importlib.import_module(
        "persona.neuron_adapter"
    ).NeuronAdapter.__module__


def test_registry_is_a_nest_adapter_alias_or_subclass():
    assert issubclass(Registry, NestAdapter)
    assert issubclass(Registry, Adapter)
    assert not inspect.isabstract(Registry)


def test_neuron_registry_is_a_neuron_adapter_alias_or_subclass():
    pytest.importorskip("neuron")
    from persona.neuron import NeuronRegistry
    from persona.neuron_adapter import NeuronAdapter

    assert issubclass(NeuronRegistry, NeuronAdapter)
    assert issubclass(NeuronRegistry, Adapter)
    assert not inspect.isabstract(NeuronRegistry)


def test_inhibitory_type_is_normalized_to_a_negative_site():
    p = PersonaParam(mean=4.0, sigma=0.1, min=-10.0, max=10.0)
    enforce_synaptic_sign(p, "inh")
    assert p.mean == -4.0
    assert p.max == 0.0
    assert p.min == -10.0


def test_inhibitory_magnitude_with_positive_bounds_becomes_negative_range():
    p = PersonaParam(mean=4.0, sigma=0.1, min=0.0, max=5.0)
    enforce_synaptic_sign(p, "inh")
    assert p.mean == -4.0
    assert p.min == -5.0
    assert p.max == 0.0


def test_inhibitory_site_cannot_be_flipped_positive():
    p = PersonaParam(mean=-4.0, sigma=0.1, min=-10.0, max=10.0)
    enforce_synaptic_sign(p, "inh")
    assert p.mean < 0.0
    assert p.max == 0.0


def test_excitatory_type_is_normalized_to_a_positive_site():
    p = PersonaParam(mean=-4.0, sigma=0.1, min=-10.0, max=10.0)
    enforce_synaptic_sign(p, "exc")
    assert p.mean == 4.0
    assert p.min == 0.0


def test_static_synapse_keeps_the_sign_identity():
    p = PersonaParam(mean=-2.0, sigma=0.1, min=-10.0, max=10.0)
    enforce_synaptic_sign(p, "signed")
    assert p.mean == -2.0
    assert p.max == 0.0


def test_resolve_site_keys_exact_bracket_beats_fnmatch():
    params = {
        "pop_x_dend[0].g_pas": object(),
        "pop_x_dend[1].g_pas": object(),
    }
    assert resolve_site_keys(params, "pop_x_dend[0].g_pas") == (
        "pop_x_dend[0].g_pas",
    )
    assert resolve_site_keys(params, "pop_x_*") == (
        "pop_x_dend[0].g_pas",
        "pop_x_dend[1].g_pas",
    )
