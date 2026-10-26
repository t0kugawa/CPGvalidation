"""PERSONA-Evo: segment-matrix initialization, pruning, and training."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from persona.partition import PlayerPartition


def territory_partition(matrix, n_players: int, seed: int) -> PlayerPartition:
    """Grow ``n_players`` territories from random seed segments.

    Seeds are allocated per connected component (at least one per component)
    and every other segment joins its nearest seed. Ties are broken uniformly
    with the Evo RNG, so the partition is deterministic from ``(seed, P)``.
    """
    n_players = int(n_players)
    n = int(matrix.n_segments)
    if n_players < 1 or n_players > n:
        raise ValueError(
            f"n_players must be in [1, {n}], got {n_players}")
    if n == 0:
        raise ValueError("cannot partition an empty segment matrix")
    rng = np.random.default_rng(int(seed) ^ 0xE0)
    adjacency = np.asarray(matrix.adjacency)
    _, labels = connected_components(csr_matrix(adjacency > 0.0),
                                     directed=False)
    components = [np.flatnonzero(labels == label)
                  for label in range(int(labels.max()) + 1)]
    if n_players < len(components):
        raise ValueError(
            f"n_players={n_players} is smaller than the {len(components)} "
            "connected components; at least one seed per component is required")
    allocation = _seed_allocation(
        n_players, [len(component) for component in components])
    seeds = []
    for component, count in zip(components, allocation):
        seeds.extend(int(value) for value in
                     rng.choice(component, size=int(count), replace=False))
    seeds = np.asarray(seeds, dtype=int)
    owner = {int(segment): index for index, segment in enumerate(seeds)}
    groups: list[list[int]] = [[int(segment)] for segment in seeds]
    for segment in range(n):
        if segment in owner:
            continue
        distances = np.asarray(matrix.distance[segment, seeds], dtype=float)
        best = float(np.min(distances))
        candidates = np.flatnonzero(distances == best)
        chosen = int(rng.choice(candidates)) if len(candidates) > 1 else int(
            candidates[0])
        groups[chosen].append(segment)
    return PlayerPartition(tuple(tuple(group) for group in groups))


def random_partition(matrix, n_players: int, seed: int) -> PlayerPartition:
    """Pure-random assignment of segments to players, then fill empties."""
    n_players = int(n_players)
    n = int(matrix.n_segments)
    if n_players < 1 or n_players > n:
        raise ValueError(
            f"n_players must be in [1, {n}], got {n_players}")
    rng = np.random.default_rng(int(seed) ^ 0xE0)
    assignment = rng.integers(0, n_players, size=n)
    groups: list[list[int]] = [[] for _ in range(n_players)]
    for segment, player in enumerate(assignment):
        groups[int(player)].append(segment)
    while any(not group for group in groups):
        empty = next(i for i, group in enumerate(groups) if not group)
        donor = max(range(len(groups)),
                    key=lambda i: (len(groups[i]), -i))
        moved = int(rng.integers(0, len(groups[donor])))
        groups[empty].append(groups[donor].pop(moved))
    return PlayerPartition(tuple(tuple(group) for group in groups))


def _seed_allocation(n_players: int, component_sizes: list[int]) -> list[int]:
    """Allocate seeds proportionally with at least one per component."""
    sizes = np.asarray(component_sizes, dtype=float)
    if len(sizes) == 0:
        raise ValueError("no components")
    allocation = np.ones(len(sizes), dtype=int)
    remaining = int(n_players) - len(sizes)
    if remaining <= 0:
        return allocation.tolist()
    raw = remaining * sizes / sizes.sum()
    extra = np.floor(raw).astype(int)
    left = remaining - int(extra.sum())
    order = np.argsort(-(raw - extra), kind="stable")
    extra[order[:left]] += 1
    allocation += extra
    return allocation.tolist()
