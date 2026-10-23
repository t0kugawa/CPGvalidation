"""PERSONA package.

The simulator adapters are imported lazily so lightweight pieces (parameters,
sampling, the CLI/runtime helpers) stay importable when NEST or NEURON is not
installed.  Attribute access is unchanged: ``from persona import Registry``
still works.
"""
from persona.adapter import Adapter
from persona.connection import Connection
from persona.noise_recorder import NoiseRecorder
from persona.parameter import PersonaParam
from persona.population import Population
from persona.schedule import CosineLR, resolve_lr_schedule
from persona.spring import SpringSpec

__all__ = [
    "Adapter",
    "Agent",
    "Connection",
    "CosineLR",
    "FisherMatrix",
    "MultiAgent",
    "NestAdapter",
    "NeuronAdapter",
    "NeuronRegistry",
    "NoiseRecorder",
    "PersonaParam",
    "PhaseBaselines",
    "Population",
    "Registry",
    "SpringSpec",
    "resolve_lr_schedule",
]

_LAZY = {
    "NestAdapter": ("persona.nest_adapter", "NestAdapter"),
    "Registry": ("persona.registry", "Registry"),
    "FisherMatrix": ("persona.strategy", "FisherMatrix"),
    "PhaseBaselines": ("persona.strategy", "PhaseBaselines"),
    "Agent": ("persona.agent", "Agent"),
    "MultiAgent": ("persona.multi_agent", "MultiAgent"),
    "NeuronAdapter": ("persona.neuron_adapter", "NeuronAdapter"),
    "NeuronRegistry": ("persona.neuron_adapter", "NeuronRegistry"),
}


def __getattr__(name: str):
    """Lazy simulator-backed exports; keep NEST/NEURON optional."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(name)
    import importlib

    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value
