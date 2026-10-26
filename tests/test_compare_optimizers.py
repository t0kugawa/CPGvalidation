import numpy as np

from benchmarks.compare.variants import VariantFisherMatrix


def _fm(mode, blocks=None, game=None):
    return VariantFisherMatrix(
        param_names=["a", "b", "c"], noise_names=[],
        beta=0.9, lambda_reg=0.5, warmup=0, lr_mu=0.1,
        preconditioner=mode, blocks=blocks, game=game)


def _sites():
    return {"a": (0.1, 0.5, 0.0), "b": (-0.2, 0.4, 0.0),
            "c": (0.05, 0.3, 0.0)}


def test_block_with_one_block_equals_full():
    full = _fm("full")
    block = _fm("block", blocks=[np.arange(full.n_total)])
    advantage = np.array([0.3, -0.1])
    left = full.step(_sites(), {}, advantage)
    right = block.step(_sites(), {}, advantage)
    assert left.keys() == right.keys()
    for key in left:
        assert np.isclose(left[key], right[key], atol=1e-12)


def test_diagonal_on_diagonal_fisher_equals_full():
    sites = {"a": (0.1, 0.5, 0.0), "b": (0.0, 0.5, 0.0),
             "c": (0.0, 0.5, 0.0)}
    full = _fm("full")
    diag = _fm("diag")
    advantage = np.array([0.3, -0.1])
    for strategy in (full, diag):
        score, applied = strategy._score_vectors(sites, {})
        strategy._accumulate(score, applied,
                             strategy._pad_advantage(advantage))
        strategy.F = np.diag(np.diag(strategy.F))
    left = full._deltas(sites, advantage)
    right = diag._deltas(sites, advantage)
    for key in left:
        assert np.isclose(left[key], right[key], atol=1e-10)


def test_variants_are_game_off_by_default():
    assert _fm("diag").game is None
    assert _fm("block").game is None


def test_cmaes_finds_quadratic_minimum():
    from benchmarks.compare.cmaes_arm import run_cmaes

    bounds = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}
    result = run_cmaes(
        lambda params: (params["x"] - 0.3) ** 2
        + (params["y"] + 0.2) ** 2,
        bounds, {"x": 0.0, "y": 0.0}, n_trials=20, seed=0)
    assert result["best_value"] < 0.1
    assert len(result["values"]) == 20
