"""Benchmark-only Fisher preconditioner variants."""
from __future__ import annotations

import numpy as np

from persona.strategy import FisherMatrix


class VariantFisherMatrix(FisherMatrix):
    """Full, diagonal, or block-diagonal natural-gradient preconditioner.

    The ``F/Xbar/G`` recurrences are unchanged. The variant only replaces the
    linear solve used for the μ proposal. It is constructed with
    ``refresh_every=1`` so ``_deltas`` calls :meth:`_solve_direct`.
    """

    def __init__(self, *args, preconditioner: str = "full", blocks=None,
                 **kwargs):
        if preconditioner not in ("full", "diag", "block"):
            raise ValueError(
                f"unknown preconditioner {preconditioner!r}")
        kwargs["refresh_every"] = 1
        super().__init__(*args, **kwargs)
        self.preconditioner = preconditioner
        if blocks is None:
            blocks = [np.arange(self.n_total, dtype=int)]
        self.blocks = [np.asarray(block, dtype=int) for block in blocks]

    def _solve_direct(self, g):
        g = np.asarray(g, dtype=np.float64).ravel()
        if self.preconditioner == "full":
            return super()._solve_direct(g)
        if self.preconditioner == "diag":
            diagonal = np.diag(self.F) + self.lambda_reg
            return g / np.maximum(diagonal, 1e-12)
        out = np.zeros_like(g)
        for block in self.blocks:
            matrix = self.F[np.ix_(block, block)] \
                + self.lambda_reg * np.eye(len(block))
            try:
                out[block] = np.linalg.solve(matrix, g[block])
            except np.linalg.LinAlgError:
                out[block] = np.linalg.pinv(matrix) @ g[block]
        return out
