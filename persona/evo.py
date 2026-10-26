"""PERSONA-Evo: segment-matrix initialization, pruning, and training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from persona.partition import PlayerPartition


@dataclass
class Redistribution:
    segment: int
    from_player: int
    to_player: int
    distance: float
    edge_types: tuple[str, ...]
    fallback: bool = False


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


def prune_and_redistribute(partition: PlayerPartition, fractions,
                           matrix, *, seed: int, impact_eps: float = 0.0,
                           min_players: int = 2
                           ) -> tuple[PlayerPartition, list[Redistribution], str]:
    """Remove zero-fraction players and reassign every orphan segment.

    The dead player's segments are assigned individually to the nearest
    surviving player by the segment-matrix distance. If no surviving player is
    reachable, the segment goes to the player with the highest incident edge
    weight (then the largest group) and the assignment is flagged.
    """
    partition.validate(matrix)
    fractions = np.asarray(fractions, dtype=np.float64).ravel()
    if fractions.size != len(partition.groups):
        raise ValueError(
            "fractions must match the number of players: "
            f"{fractions.size} != {len(partition.groups)}")
    min_players = max(1, int(min_players))
    threshold = max(float(impact_eps), 1e-12)
    dead = [index for index, value in enumerate(fractions)
            if float(value) <= threshold]
    if not dead:
        return partition, [], "stable"
    survivors = [index for index in range(len(partition.groups))
                 if index not in dead]
    if not survivors:
        keep = int(np.argmax(fractions))
        all_segments = tuple(sorted(
            segment for group in partition.groups for segment in group))
        return (PlayerPartition((all_segments,)), [], "all_dead")
    if len(survivors) < min_players:
        return partition, [], "min_players"

    rng = np.random.default_rng(int(seed) ^ 0xE2)
    groups = [list(partition.groups[index]) for index in survivors]
    log: list[Redistribution] = []
    for dead_index in dead:
        for segment in partition.groups[dead_index]:
            segment = int(segment)
            scores = []
            for index, members in enumerate(groups):
                distance, weight = matrix.player_link(segment, members)
                scores.append((distance, weight, index))
            finite = [entry for entry in scores if np.isfinite(entry[0])]
            fallback = not finite
            if fallback:
                best = max(
                    scores,
                    key=lambda entry: (entry[1], len(groups[entry[2]]),
                                       -entry[2]))
                target = int(best[2])
                distance = float("inf")
            else:
                best_distance = min(entry[0] for entry in finite)
                nearest = [entry for entry in finite
                           if np.isclose(entry[0], best_distance)]
                best_weight = max(entry[1] for entry in nearest)
                tied = [entry for entry in nearest
                        if np.isclose(entry[1], best_weight)]
                chosen = tied[int(rng.integers(0, len(tied)))]
                target = int(chosen[2])
                distance = float(best_distance)
            edge_types: set[str] = set()
            for member in groups[target]:
                key = (min(segment, int(member)), max(segment, int(member)))
                edge_types.update(matrix.edge_types.get(key, ()))
            groups[target].append(segment)
            log.append(Redistribution(
                segment=segment,
                from_player=int(dead_index),
                to_player=target,
                distance=distance,
                edge_types=tuple(sorted(edge_types)),
                fallback=bool(fallback),
            ))
    result = PlayerPartition(tuple(tuple(group) for group in groups))
    result.validate(matrix)
    return result, log, "pruned"


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
