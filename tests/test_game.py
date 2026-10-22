import numpy as np
import pytest

from persona.game import GameConfig, StackGame
from persona.homeostasis import PhysicalMap, PlayerSpec


def _game(mode="inverse", fee=0.0, burnin=0, n=2, base_seat="none",
          liveness=0.0, transfer_rate=0.05, throttle_exponent=1.0):
    specs = [PlayerSpec(f"p{i}", "population", (f"s{i}",), ptype="exc")
             for i in range(n)]
    pm = PhysicalMap.from_specs(specs)
    cfg = GameConfig(enabled=True, throttle_mode=mode, movement_fee=fee,
                     burnin_steps=burnin, base_seat=base_seat,
                     liveness_fraction=liveness,
                     transfer_rate=transfer_rate,
                     throttle_exponent=throttle_exponent)
    return StackGame(pm, cfg, site_names=[f"s{i}" for i in range(n)],
                     dt_s=1.0)


def test_config_default_is_off():
    assert GameConfig().enabled is False


def test_current_best_preset_is_the_selected_a25_configuration():
    cfg = GameConfig.current_best()
    assert cfg.enabled is True
    assert cfg.throttle_mode == "proportional"
    assert cfg.base_seat == "uncertainty"
    assert cfg.liveness_fraction == 0.05
    assert cfg.throttle_exponent == 2.5
    assert cfg.transfer_rate == 0.05
    assert cfg.movement_fee == 0.0


def test_config_rejects_unknown_throttle_mode():
    with pytest.raises(ValueError):
        GameConfig(throttle_mode="sideways")


def test_uniform_stacks_are_neutral_in_both_modes():
    for mode in ("proportional", "inverse"):
        g = _game(mode)
        np.testing.assert_allclose(g.throttle(), np.ones(2))


def test_proportional_throttle_tracks_share():
    g = _game("proportional")
    g.stacks = np.array([0.2, 0.8])
    np.testing.assert_allclose(g.throttle(), [0.25, 1.0])


def test_inverse_throttle_tracks_lack_of_share():
    g = _game("inverse")
    g.stacks = np.array([0.2, 0.8])
    np.testing.assert_allclose(g.throttle(), [1.0, 0.25])


def test_throttle_scales_the_player_proposal():
    g = _game("inverse")
    g.stacks = np.array([0.2, 0.8])
    out = g.apply(np.ones(2), np.array([1.0, 2.0]), steps=1)
    np.testing.assert_allclose(out, [1.0, 0.5])


def test_transfer_is_zero_sum():
    g = _game("inverse")
    g.stacks = np.array([0.5, 0.5])
    g.apply(np.array([1.0, -1.0]), np.array([1.0, 1.0]), steps=1)
    assert abs(g.last["transfer_sum"]) < 1e-12
    assert np.isclose(g.stacks.sum(), 1.0)


def test_player_with_positive_contribution_gains_share():
    g = _game("inverse")
    g.stacks = np.array([0.5, 0.5])
    for t in range(10):
        g.apply(np.array([1.0, -1.0]), np.array([1.0, 1.0]), steps=t + 1)
    assert g.stacks[0] > 0.5
    assert g.stacks[1] < 0.5


def test_no_consume_arm_does_not_debit_movement():
    g = _game("inverse", fee=0.0)
    g.stacks = np.array([0.5, 0.5])
    g.apply(np.ones(2), np.array([0.1, 0.1]), steps=1)
    assert g.last["pot"] == 0.0
    np.testing.assert_allclose(g.last["fees"], [0.0, 0.0])


def test_pot_debit_equals_credit():
    g = _game("inverse", fee=0.05)
    g.stacks = np.array([0.5, 0.5])
    g.apply(np.ones(2), np.array([0.1, 0.3]), steps=1, F=np.eye(2))
    assert np.isclose(g.last["pot"], 0.05)
    assert np.isclose(g.last["fees"].sum(), g.last["pot"])
    assert np.isclose(g.stacks.sum(), 1.0)


def test_stacks_stay_on_simplex_over_many_steps():
    g = _game("inverse", fee=0.05, n=5)
    rng = np.random.default_rng(0)
    for t in range(200):
        g.apply(rng.normal(size=5), rng.normal(size=5), steps=t + 1,
                F=np.eye(5))
        assert np.isclose(g.stacks.sum(), 1.0)
        assert np.all(g.stacks >= 0.0)


def test_burnin_delays_transfer_but_not_throttle():
    g = _game("inverse", burnin=5)
    g.stacks = np.array([0.2, 0.8])
    out = g.apply(np.ones(2), np.array([1.0, 2.0]), steps=1)
    np.testing.assert_allclose(out, [1.0, 0.5])
    np.testing.assert_allclose(g.stacks, [0.2, 0.8])


def test_new_site_enters_with_average_share():
    g = _game(n=2)
    g.stacks = np.array([0.7, 0.3])
    g.ensure_params(["s0", "s1", "new"])
    assert g.physical_map.n_players == 3
    assert len(g.stacks) == 3
    assert np.isclose(g.stacks.sum(), 1.0)


def test_apply_rejects_wrong_length():
    g = _game()
    with pytest.raises(ValueError):
        g.apply(np.ones(3), np.ones(2), steps=1)


def test_config_rejects_unknown_base_seat():
    with pytest.raises(ValueError):
        GameConfig(base_seat="mass")


def test_config_rejects_nonpositive_throttle_exponent():
    with pytest.raises(ValueError):
        GameConfig(throttle_exponent=0.0)


def test_throttle_exponent_sharpens_the_contestable_part():
    g = _game("proportional", throttle_exponent=2.0)
    g.stacks = np.array([0.2, 0.8])
    np.testing.assert_allclose(g.throttle(), [0.0625, 1.0])


def test_throttle_exponent_one_matches_the_linear_curve():
    linear = _game("proportional")
    powered = _game("proportional", throttle_exponent=1.0)
    linear.stacks = np.array([0.2, 0.8])
    powered.stacks = np.array([0.2, 0.8])
    np.testing.assert_allclose(linear.throttle(), powered.throttle())


def test_config_rejects_liveness_outside_unit_interval():
    with pytest.raises(ValueError):
        GameConfig(liveness_fraction=1.5)


def test_zero_liveness_matches_no_base_seat():
    a = _game("proportional")
    b = _game("proportional", base_seat="uniform", liveness=0.0)
    a.stacks = np.array([0.0, 1.0])
    b.stacks = np.array([0.0, 1.0])
    out_a = a.apply(np.ones(2), np.array([1.0, 2.0]), steps=1)
    out_b = b.apply(np.ones(2), np.array([1.0, 2.0]), steps=1)
    np.testing.assert_allclose(out_a, out_b)


def test_uniform_base_keeps_a_zero_stack_player_moving():
    g = _game("proportional", base_seat="uniform", liveness=0.1,
              transfer_rate=0.0)
    g.stacks = np.array([0.0, 1.0])
    out = g.apply(np.ones(2), np.array([1.0, 2.0]), steps=1)
    np.testing.assert_allclose(out, [0.1, 2.0], atol=1e-9)


def test_uncertainty_base_gives_more_movement_to_an_uncertain_player():
    g = _game("proportional", base_seat="uncertainty", liveness=0.2,
              transfer_rate=0.0)
    g.stacks = np.array([0.0, 1.0])
    g.apply(np.ones(2), np.ones(2), steps=1, sigmas=np.array([0.1, 0.1]))
    out = g.apply(np.ones(2), np.ones(2), steps=2,
                  sigmas=np.array([0.2, 0.05]))
    np.testing.assert_allclose(out, [0.2, 0.85], atol=1e-9)


def test_uncertainty_base_reduces_a_confident_player():
    g = _game("proportional", base_seat="uncertainty", liveness=0.2,
              transfer_rate=0.0)
    g.stacks = np.array([0.0, 1.0])
    g.apply(np.ones(2), np.ones(2), steps=1, sigmas=np.array([0.1, 0.1]))
    out = g.apply(np.ones(2), np.ones(2), steps=2,
                  sigmas=np.array([0.05, 0.1]))
    np.testing.assert_allclose(out, [0.1, 1.0], atol=1e-9)


def test_base_seat_is_normalized_to_one():
    g = _game("proportional", base_seat="uncertainty", liveness=0.2)
    g.stacks = np.array([0.5, 0.5])
    g.apply(np.ones(2), np.ones(2), steps=1,
            sigmas=np.array([0.2, 0.1]))
    assert np.isclose(g.last["base_seat"].max(), 1.0)
