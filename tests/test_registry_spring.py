import pytest

from persona.parameter import PersonaParam
from persona.population import Population
from persona.registry import Registry


def _reg():
    reg = Registry()
    reg.create_population(Population("izhikevich", 4, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=2.0, sigma=0.4, min=0.0, max=5.0),
        "c": PersonaParam(mean=0.0, sigma=0.0, min=0.0, max=1.0),
    }, "pop"))
    return reg


def test_custom_spring_force_and_anchor(nest_kernel):
    reg = _reg()
    reg.create_spring("s", "pop_pop_a", power=0.25)
    state = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.3}, 0.1)
    assert state["pop_pop_a"]["power"] == 0.25
    assert state["pop_pop_a"]["force"] == pytest.approx(-0.25)
    assert state["pop_pop_a"]["anchor"] == 1.0


def test_dynamic_power_uses_lr_and_variance(nest_kernel):
    reg = _reg()
    reg.create_spring("s", "pop_pop_a", power=None, gain=1.0, max_power=0.5)
    at_ref = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.3}, 0.1)["pop_pop_a"]["power"]
    tighter = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.15}, 0.1)["pop_pop_a"]["power"]
    looser = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.6}, 0.1)["pop_pop_a"]["power"]
    assert at_ref == pytest.approx(0.1)
    assert tighter > at_ref
    assert looser < at_ref
    assert tighter <= 0.5


def test_dynamic_power_zero_sigma_is_finite(nest_kernel):
    reg = _reg()
    reg.create_spring("s", "pop_pop_a", power=None, gain=1.0, max_power=0.5)
    state = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.0}, 1.0)
    assert state["pop_pop_a"]["power"] == pytest.approx(0.5)
    reg.create_spring("zero_ref", "pop_pop_c", power=None, gain=1.0, max_power=0.5)
    zero = reg.spring_state({"pop_pop_c": 0.0}, {"pop_pop_c": 0.0}, 1.0)
    assert zero["pop_pop_c"]["power"] == 0.0


def test_spring_skips_frozen_mean(nest_kernel):
    reg = _reg()
    reg.freeze("pop_pop_a", attrs=("mean",))
    reg.create_spring("s", "pop_pop_a", power=0.25)
    assert reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.3}, 0.1) == {}


def test_spring_overlap_and_validation(nest_kernel):
    reg = _reg()
    reg.create_spring("s1", "pop_*")
    with pytest.raises(ValueError):
        reg.create_spring("s2", "pop_pop_a")
    reg.remove_spring("s1")
    with pytest.raises(ValueError):
        reg.create_spring("s3", "pop_pop_b", power=0.9, max_power=0.5)
    assert reg.springs() == {}
