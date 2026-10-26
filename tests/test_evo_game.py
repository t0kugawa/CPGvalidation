import numpy as np

from persona.game import GameConfig, StackGame
from persona.homeostasis import PhysicalMap, PlayerSpec


def _map():
    return PhysicalMap.from_specs([
        PlayerSpec("p0", "population", ("a",), size=1),
        PlayerSpec("p1", "population", ("b",), size=1),
    ])


def test_zero_share_floor_is_valid_and_keeps_a_simplex():
    config = GameConfig(enabled=True, share_floor=0.0, burnin_steps=0)
    game = StackGame(_map(), config, site_names=["a", "b"])
    game.apply(np.array([1.0, -1.0]), np.array([0.0, 0.0]), steps=1000,
               sigmas=np.array([0.1, 0.1]))
    assert np.all(np.isfinite(game.stacks))
    assert np.isclose(game.stacks.sum(), 1.0)


def test_zero_fraction_is_reachable_without_a_floor():
    config = GameConfig(
        enabled=True, share_floor=0.0, throttle_mode="proportional",
        transfer_rate=1.0, burnin_steps=0)
    game = StackGame(_map(), config, site_names=["a", "b"])
    game.apply(np.array([1.0, 1.0]), np.array([1.0, 0.0]), steps=1000,
               sigmas=np.array([0.1, 0.1]))
    assert np.isclose(game.stacks.sum(), 1.0)
    assert np.min(game.stacks) == 0.0


def test_all_zero_transfer_guard_restores_previous_stacks():
    config = GameConfig(enabled=True, share_floor=0.0, burnin_steps=0)
    game = StackGame(_map(), config, site_names=["a", "b"])
    game.stacks = np.zeros(2)
    game.apply(np.array([1.0, -1.0]), np.array([0.0, 0.0]), steps=1000,
               sigmas=np.array([0.1, 0.1]))
    assert np.all(np.isfinite(game.stacks))
    assert game.last.get("error") == "all_zero"
