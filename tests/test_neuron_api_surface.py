"""API parity between the NEST and NEURON adapters."""
import inspect

import pytest

from persona.adapter import Adapter
from persona.nest_adapter import NestAdapter
from persona.registry import Registry

pytest.importorskip("neuron")
from persona.neuron import NeuronRegistry
from persona.neuron_adapter import NeuronAdapter


def _signature_shape(func):
    sig = inspect.signature(func)
    return [
        (p.name, p.kind, p.default)
        for p in sig.parameters.values()
        if p.name != "self"
    ]


def test_both_concrete_adapters_implement_the_abstract_surface():
    assert inspect.isabstract(Adapter)
    assert not inspect.isabstract(NestAdapter)
    assert not inspect.isabstract(NeuronAdapter)
    assert issubclass(NestAdapter, Adapter)
    assert issubclass(NeuronAdapter, Adapter)
    assert issubclass(Registry, NestAdapter)
    assert issubclass(NeuronRegistry, NeuronAdapter)
    assert set(Adapter.__abstractmethods__) <= set(dir(NestAdapter))
    assert set(Adapter.__abstractmethods__) <= set(dir(NeuronAdapter))


def test_shared_method_signatures_match():
    for name in sorted(Adapter.__abstractmethods__):
        if name == "__init__":
            continue
        nest_attr = getattr(NestAdapter, name)
        neuron_attr = getattr(NeuronAdapter, name)
        if isinstance(nest_attr, property):
            assert isinstance(neuron_attr, property), name
            continue
        assert _signature_shape(nest_attr) == _signature_shape(neuron_attr), name


def test_public_state_attributes_exist():
    instances = [NestAdapter(), NeuronAdapter(seed=1)]
    for instance in instances:
        for attr in ("population_cfgs", "connection_cfgs",
                     "population_nodes", "noise_recorders",
                     "last_apply_errors"):
            assert hasattr(instance, attr), (type(instance).__name__, attr)
