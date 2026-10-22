import pytest

from persona.parameter import PersonaParam
from persona.population import Population
from persona.registry import Registry


def _reg():
    reg = Registry()
    reg.create_population(Population("izhikevich", 4, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=2.0, sigma=0.4, min=0.0, max=5.0),
    }, "pop"))
    return reg


def test_defaults_captured_at_creation(nest_kernel):
    reg = _reg()
    assert reg.defaults("pop_pop_a")["mean"] == 1.0
    assert reg.defaults("pop_pop_a")["sigma"] == 0.3


def test_freeze_mean_snaps_and_pins_only_mean(nest_kernel):
    reg = _reg()
    p = reg.get_all_optimizable_params()["pop_pop_a"]
    p.mean, p.sigma = 2.5, 0.9
    reg.freeze("pop_pop_a", attrs=("mean",))
    assert p.mean == 1.0
    assert p.sigma == 0.9
    assert reg.is_frozen("pop_pop_a", "mean")
    assert not reg.is_frozen("pop_pop_a", "sigma")


def test_freeze_glob_and_unfreeze(nest_kernel):
    reg = _reg()
    assert reg.freeze("pop_*_a", attrs=("mean",)) == ("pop_pop_a",)
    assert reg.frozen_attrs("pop_pop_a") == frozenset({"mean"})
    reg.unfreeze("pop_pop_a")
    assert reg.frozen_attrs("pop_pop_a") == frozenset()


def test_freeze_unknown_pattern_and_attr(nest_kernel):
    reg = _reg()
    with pytest.raises(KeyError):
        reg.freeze("nope_*")
    with pytest.raises(ValueError):
        reg.freeze("pop_pop_a", attrs=("min",))


def test_recapture_refreshes_the_anchor(nest_kernel):
    reg = _reg()
    p = reg.get_all_optimizable_params()["pop_pop_a"]
    p.mean, p.sigma = 7.0, 0.0
    reg.recapture_defaults("pop_pop_a")
    assert reg.defaults("pop_pop_a")["mean"] == 7.0
    reg.freeze("pop_pop_a", attrs=("mean", "sigma"))
    assert (p.mean, p.sigma) == (7.0, 0.0)


def test_finalize_refreshes_defaults_after_post_create_mutation(nest_kernel):
    reg = _reg()
    p = reg.get_all_optimizable_params()["pop_pop_a"]
    p.mean, p.sigma = 5.0, 0.0
    reg.finalize_connections()
    assert reg.defaults("pop_pop_a")["mean"] == 5.0
    assert reg.defaults("pop_pop_a")["sigma"] == 0.0
