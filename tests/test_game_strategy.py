import numpy as np

from persona.game import GameConfig, StackGame
from persona.homeostasis import PhysicalMap, PlayerSpec
from persona.strategy import FisherMatrix


SITES = {"s0": (0.1, 0.1, 0.0), "s1": (-0.2, 0.1, 0.0)}
ADV = np.array([1.0, 1.0])


def _strategy(game=None):
    return FisherMatrix(param_names=["s0", "s1"], lr_mu=0.1, beta=0.9,
                        warmup=0, lambda_reg=1e-3, game=game, dt_s=1.0)


def _game(mode="inverse", transfer_rate=0.05, fee=0.0,
          base_seat="none", liveness=0.0):
    specs = [PlayerSpec("p0", "population", ("s0",), ptype="exc"),
             PlayerSpec("p1", "population", ("s1",), ptype="exc")]
    pm = PhysicalMap.from_specs(specs)
    cfg = GameConfig(enabled=True, throttle_mode=mode,
                     transfer_rate=transfer_rate, movement_fee=fee,
                     burnin_steps=0, base_seat=base_seat,
                     liveness_fraction=liveness)
    return StackGame(pm, cfg, site_names=["s0", "s1"], dt_s=1.0)


def test_strategy_without_game_is_unchanged():
    plain = _strategy()
    explicit_none = _strategy(game=None)
    for _ in range(5):
        a = plain.step(dict(SITES), {}, ADV)
        b = explicit_none.step(dict(SITES), {}, ADV)
        assert a == b
    np.testing.assert_allclose(plain.F, explicit_none.F)
    np.testing.assert_allclose(plain.G, explicit_none.G)


def test_strategy_with_game_throttles_player_deltas():
    base = _strategy()
    game = _game(transfer_rate=0.0)
    game.stacks = np.array([0.2, 0.8])
    throttled = _strategy(game=game)
    for _ in range(3):
        d0 = base.step(dict(SITES), {}, ADV)
        d1 = throttled.step(dict(SITES), {}, ADV)
    assert np.isclose(d1["s0"], d0["s0"])
    assert np.isclose(d1["s1"], 0.25 * d0["s1"])


def test_proportional_game_freezes_the_empty_player():
    base = _strategy()
    game = _game(mode="proportional", transfer_rate=0.0)
    game.stacks = np.array([0.0, 1.0])
    throttled = _strategy(game=game)
    for _ in range(3):
        d0 = base.step(dict(SITES), {}, ADV)
        d1 = throttled.step(dict(SITES), {}, ADV)
    assert abs(d1["s0"]) < 1e-9
    assert np.isclose(d1["s1"], d0["s1"])


def test_strategy_game_updates_stacks_and_stays_on_simplex():
    game = _game(transfer_rate=0.1)
    strat = _strategy(game=game)
    asym = {"s0": (0.4, 0.1, 0.0), "s1": (-0.02, 0.1, 0.0)}
    for _ in range(20):
        strat.step(dict(asym), {}, ADV)
    assert np.isclose(game.stacks.sum(), 1.0)
    assert not np.allclose(game.stacks, [0.5, 0.5])


def test_strategy_ensure_names_extends_game():
    game = _game()
    strat = _strategy(game=game)
    strat.ensure_names(["s0", "s1", "s2"])
    assert len(game.site_names) == 3
    assert len(game.stacks) == 3


def test_strategy_feeds_sigmas_to_the_uncertainty_base():
    game = _game(base_seat="uncertainty", liveness=0.1)
    strat = _strategy(game=game)
    strat.step({"s0": (0.1, 0.3, 0.0), "s1": (-0.2, 0.1, 0.0)}, {}, ADV)
    assert game.sigma_ref["s0"] == 0.3
    assert game.sigma_ref["s1"] == 0.1
