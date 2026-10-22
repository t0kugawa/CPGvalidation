import numpy as np

from persona.parameter import PersonaParam
from persona.registry import clamp_weight_sign


def test_excitatory_weight_cannot_cross_zero():
    p = PersonaParam(mean=0.5, sigma=0.1, min=-1.0, max=1.0)
    clamp_weight_sign(p)
    assert p.min == 0.0
    assert p.max == 1.0


def test_inhibitory_weight_cannot_cross_zero():
    p = PersonaParam(mean=-0.5, sigma=0.1, min=-1.0, max=1.0)
    clamp_weight_sign(p)
    assert p.min == -1.0
    assert p.max == 0.0


def test_zero_weight_is_left_untouched():
    p = PersonaParam(mean=0.0, sigma=0.1, min=-1.0, max=1.0)
    clamp_weight_sign(p)
    assert p.min == -1.0
    assert p.max == 1.0


def test_stricter_existing_bounds_are_not_widened():
    p = PersonaParam(mean=0.5, sigma=0.05, min=0.2, max=0.8)
    clamp_weight_sign(p)
    assert p.min == 0.2
    assert p.max == 0.8


def test_clamped_weight_stays_on_its_side_after_updates():
    p = PersonaParam(mean=0.5, sigma=0.1, min=-1.0, max=1.0)
    clamp_weight_sign(p)
    for _ in range(50):
        p.update_mean(-1.0)
    assert p.mean >= 0.0
    assert not np.any(np.asarray([p.mean]) < 0.0)
