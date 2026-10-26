"""PERSONA-Evo: segment-matrix initialization, pruning, and training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

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


@dataclass
class EvoConfig:
    n_players: int = 24
    generations: int = 6
    steps_per_generation: int = 400
    eval_window: int = 50
    init: str = "territory"
    stop: str = "both"
    patience: int = 2
    nash_tol: float = 1e-3
    impact_eps: float = 0.0
    min_players: int = 2
    seed: int = 0
    extension_steps: int = 0
    fresh_process: bool = True
    sizes: dict | None = field(default=None, compare=False)

    def __post_init__(self):
        self.n_players = int(self.n_players)
        self.generations = int(self.generations)
        self.steps_per_generation = int(self.steps_per_generation)
        self.eval_window = int(self.eval_window)
        self.patience = int(self.patience)
        self.nash_tol = float(self.nash_tol)
        self.impact_eps = float(self.impact_eps)
        self.min_players = int(self.min_players)
        self.seed = int(self.seed)
        self.extension_steps = int(self.extension_steps)
        if self.n_players < 1:
            raise ValueError("n_players must be >= 1")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.steps_per_generation < 1:
            raise ValueError("steps_per_generation must be >= 1")
        if self.eval_window < 1:
            raise ValueError("eval_window must be >= 1")
        if self.init not in ("territory", "random"):
            raise ValueError("init must be 'territory' or 'random'")
        if self.stop not in ("partition", "nash", "both"):
            raise ValueError("stop must be 'partition', 'nash', or 'both'")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.nash_tol < 0.0:
            raise ValueError("nash_tol must be >= 0")
        if self.impact_eps < 0.0:
            raise ValueError("impact_eps must be >= 0")
        if self.min_players < 1:
            raise ValueError("min_players must be >= 1")
        if self.extension_steps < 0:
            raise ValueError("extension_steps must be >= 0")


@dataclass
class GenerationResult:
    partition: PlayerPartition
    generation: int
    seed: int
    fractions: np.ndarray
    fraction_mean: np.ndarray
    contribution: np.ndarray
    transfer: np.ndarray
    replicator_residual: float
    movement: float
    metrics: dict
    wall_s: float


@dataclass
class EvoResult:
    final_partition: PlayerPartition
    history: list[dict]
    stop_reason: str
    equilibrium: dict


class EvoTrainer:
    """Restart-from-seed evolutionary outer loop over player partitions."""

    def __init__(self, matrix, config: EvoConfig, runner):
        self.matrix = matrix
        self.config = config
        self.runner = runner
        self.history: list[dict] = []

    def run(self) -> EvoResult:
        cfg = self.config
        if cfg.init == "territory":
            partition = territory_partition(self.matrix, cfg.n_players,
                                            cfg.seed)
        else:
            partition = random_partition(self.matrix, cfg.n_players, cfg.seed)
        stable = 0
        seen: dict[tuple, int] = {}
        previous_fractions = None
        last_key = None
        for generation in range(cfg.generations):
            result = self.runner(partition, generation, cfg.seed)
            partition = result.partition
            partition.validate(self.matrix)
            fractions = np.asarray(result.fractions, dtype=np.float64).ravel()
            if fractions.size != len(partition.groups):
                raise ValueError(
                    "generation fractions do not match the partition: "
                    f"{fractions.size} != {len(partition.groups)}")
            threshold = max(cfg.impact_eps, 1e-12)
            dead = fractions <= threshold
            fraction_stability = None
            if previous_fractions is not None and \
                    previous_fractions.size == fractions.size:
                fraction_stability = float(np.max(np.abs(
                    fractions - previous_fractions)))
            key = tuple(partition.groups)
            record = {
                "generation": int(generation),
                "partition": [list(group) for group in partition.groups],
                "fractions": fractions.tolist(),
                "fraction_mean": np.asarray(
                    result.fraction_mean, dtype=float).tolist(),
                "contribution": np.asarray(
                    result.contribution, dtype=float).tolist(),
                "transfer": np.asarray(result.transfer, dtype=float).tolist(),
                "replicator_residual": float(result.replicator_residual),
                "movement": float(result.movement),
                "fraction_stability": fraction_stability,
                "dead": dead.tolist(),
                "metrics": result.metrics,
                "wall_s": float(result.wall_s),
                "prune_reason": None,
                "redistribution": [],
            }
            self.history.append(record)

            no_dead = not bool(dead.any())
            stable_next = stable + 1 if no_dead else 0
            partition_ok = no_dead and stable_next >= cfg.patience
            nash_ok = no_dead and result.replicator_residual <= cfg.nash_tol

            if not np.any(~dead):
                reason = "all_dead"
            elif int(np.sum(~dead)) < cfg.min_players:
                reason = "min_players"
            elif cfg.stop == "partition":
                reason = "partition" if partition_ok else None
            elif cfg.stop == "nash":
                if nash_ok:
                    reason = "nash"
                elif no_dead:
                    reason = "partition_stable_nash_failed"
                else:
                    reason = None
            else:  # both
                if partition_ok and nash_ok:
                    reason = "both"
                elif no_dead and not nash_ok:
                    reason = "partition_stable_nash_failed"
                else:
                    reason = None
            if reason is None and key in seen and key != last_key:
                reason = "cycle"
            if reason is not None:
                if reason == "all_dead":
                    final = PlayerPartition((tuple(sorted(
                        segment for group in partition.groups
                        for segment in group)),))
                else:
                    final = partition
                return EvoResult(
                    final_partition=final,
                    history=self.history,
                    stop_reason=reason,
                    equilibrium=self._equilibrium(
                        final, result, no_dead, reason, generation + 1,
                        fraction_stability),
                )

            seen[key] = generation
            last_key = key
            stable = stable_next
            previous_fractions = fractions
            new_partition, redistribution, prune_reason = \
                prune_and_redistribute(
                    partition, fractions, self.matrix, seed=cfg.seed,
                    impact_eps=cfg.impact_eps, min_players=cfg.min_players)
            record["prune_reason"] = prune_reason
            record["redistribution"] = [asdict(entry)
                                        for entry in redistribution]
            if prune_reason in ("all_dead", "min_players"):
                return EvoResult(
                    final_partition=new_partition,
                    history=self.history,
                    stop_reason=prune_reason,
                    equilibrium=self._equilibrium(
                        new_partition, result, False, prune_reason,
                        generation + 1, fraction_stability),
                )
            partition = new_partition
        last = self.history[-1] if self.history else {}
        return EvoResult(
            final_partition=partition,
            history=self.history,
            stop_reason="max_generations",
            equilibrium={
                "stop_reason": "max_generations",
                "generations": len(self.history),
                "replicator_residual": last.get("replicator_residual"),
                "movement": last.get("movement"),
                "fraction_stability": last.get("fraction_stability"),
                "partition_stable": bool(
                    self.history and not any(self.history[-1]["dead"])),
                "floor_hit": False,
            },
        )

    def _equilibrium(self, partition, result, partition_stable, reason,
                     generations, fraction_stability) -> dict:
        return {
            "stop_reason": reason,
            "generations": int(generations),
            "replicator_residual": float(result.replicator_residual),
            "movement": float(result.movement),
            "fraction_stability": fraction_stability,
            "partition_stable": bool(partition_stable),
            "floor_hit": reason == "min_players",
            "n_players": len(partition.groups),
        }


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
