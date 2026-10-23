"""NEURON implementation helpers and compatibility exports.

The heavy adapter is imported lazily so ``import persona.neuron.groups`` does
not pull in the registry (and ``persona.neuron_adapter`` can import the helper
submodules without a circular import).
"""
from __future__ import annotations

from persona.neuron.cells import PathSpec, SectionCell, SpikeSource, parse_cell_path
from persona.neuron.groups import CellGroup, RateGroup, RelayGroup, SynapseGroup
from persona.neuron.recorders import SpikeRecorder, WeightRecorder
from persona.neuron.synapses import (
    StaticDispatcherFactory,
    StaticSynapseFactory,
    STDPSynapseFactory,
    SynapseContext,
    SynapseFactory,
    SynapseHandle,
    TapSynapseFactory,
    builtin_synapse_factories,
)
from persona.neuron.cells import builtin_cell_factories


def __getattr__(name: str):
    if name in {"NeuronAdapter", "NeuronRegistry"}:
        from persona.neuron_adapter import NeuronAdapter

        return NeuronAdapter
    raise AttributeError(name)


__all__ = [
    "NeuronAdapter", "NeuronRegistry",
    "PathSpec", "parse_cell_path", "SectionCell", "SpikeSource",
    "CellGroup", "SynapseGroup", "RateGroup", "RelayGroup",
    "SpikeRecorder", "WeightRecorder",
    "SynapseContext", "SynapseHandle", "SynapseFactory",
    "StaticSynapseFactory", "StaticDispatcherFactory",
    "STDPSynapseFactory", "TapSynapseFactory",
    "builtin_cell_factories", "builtin_synapse_factories",
]
