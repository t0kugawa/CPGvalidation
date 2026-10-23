"""Deterministic threaded exploration sampling for PERSONA sites.

Each ``PersonaParam`` owns an independent RNG stream and is drawn exactly once
per step, so how the sites are distributed across worker threads cannot change
any site's draw sequence.  Single-threaded and threaded sampling are therefore
bit-identical.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Self

from persona.parameter import PersonaParam
from persona.runtime import validate_threads


def _sample_chunk(items) -> list[tuple[str, float]]:
    return [(name, float(param.sample_exploration()))
            for name, param in items]


class ReplicaSampler:
    """Draws one *different* exploratory sample per replica.

    The shared policy's ``PersonaParam`` objects draw ``N`` values in one
    vectorised call.  Replica ``i`` receives the ``i``-th draw, so the samples
    are different by construction and the sampling cost stays low.
    """

    def __init__(self, replicas: int = 1):
        self.replicas = validate_threads(replicas)

    def sample(self, params: Mapping[str, PersonaParam]):
        actions: list[dict[str, float]] = [{} for _ in range(self.replicas)]
        snapshots: list[dict[str, tuple[float, float, float]]] = [
            {} for _ in range(self.replicas)]
        for name, param in params.items():
            mu = float(param.mean)
            sigma = float(param.sigma)
            values = param.sample_explorations(self.replicas)
            for replica, value in enumerate(values):
                xhat = float(value)
                actions[replica][name] = xhat
                snapshots[replica][name] = (mu, sigma, xhat)
        return list(zip(actions, snapshots))

    def close(self) -> None:
        """Compatibility no-op: vectorised sampling owns no pool."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ExplorationSampler:
    """Samples a site mapping, optionally across a persistent thread pool."""

    def __init__(self, threads: int = 1):
        self.threads = validate_threads(threads)
        self._pool: ThreadPoolExecutor | None = None

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.threads,
                thread_name_prefix="persona-sample")
        return self._pool

    def sample(self, params: Mapping[str, PersonaParam]) -> dict[str, float]:
        items = list(params.items())
        if not items:
            return {}

        workers = min(self.threads, len(items))
        if workers <= 1:
            return {name: float(param.sample_exploration())
                    for name, param in items}

        chunk_size = math.ceil(len(items) / workers)
        chunks = [items[i:i + chunk_size]
                  for i in range(0, len(items), chunk_size)]
        out: dict[str, float] = {}
        for pairs in self._executor().map(_sample_chunk, chunks):
            out.update(pairs)
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
