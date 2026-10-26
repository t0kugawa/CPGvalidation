import numpy as np

from benchmarks.compare.surrogate import (
    finite_difference_grad,
    surrogate_loss_and_grad,
)
from benchmarks.compare.toy_cpg import INITIAL, ToyConfig


def test_surrogate_matches_finite_difference():
    config = ToyConfig(steps=30, seed=0)
    loss, grad = surrogate_loss_and_grad(INITIAL, config)
    finite = finite_difference_grad(INITIAL, config, eps=1e-5, idx=(0, 1, 2))
    relative = (np.linalg.norm(grad[:3] - finite)
                / max(np.linalg.norm(finite), 1e-12))
    assert np.isfinite(loss)
    assert np.isfinite(grad).all()
    assert relative < 1e-4, relative
