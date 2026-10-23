"""Compatibility facade for the NEST registry.

The concrete implementation now lives in :mod:`persona.nest_adapter` as
``NestAdapter``.  ``Registry`` remains an import-compatible subclass so every
existing import (``persona.registry.Registry``) and the old module-level
helper imports continue to work unchanged.
"""
from persona.adapter import clamp_weight_sign, plastic_delta_from_events
from persona.nest_adapter import NestAdapter


class Registry(NestAdapter):
    """Backward-compatible name for the NEST adapter."""


__all__ = ["Registry", "NestAdapter", "clamp_weight_sign",
           "plastic_delta_from_events"]
