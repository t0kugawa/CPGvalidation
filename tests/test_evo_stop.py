import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from persona.evo import EvoConfig, EvoTrainer, GenerationResult
from persona.partition import PlayerPartition


class _Matrix:
    n_segments = 3
    segments = ()
    adjacency = np.zeros((3, 3), dtype=float)
    for _left, _right in ((0, 1), (1, 2)):
        adjacency[_left, _right] = 1.0
        adjacency[_right, _left] = 1.0
    distance = shortest_path(csr_matrix(adjacency > 0.0), directed=False,
                             unweighted=True)
    edge_types = {(0, 1): frozenset({"incidence"}),
                  (1, 2): frozenset({"incidence"})}

    def validate_partition(self, partition):
        seen = set()
        for group in partition.groups:
            if not group:
                raise ValueError("empty")
            for segment in group:
                if segment in seen:
                    raise ValueError("duplicate")
                seen.add(segment)
        if seen != set(range(self.n_segments)):
            raise ValueError("missing")

    def player_link(self, segment, members):
        members = list(members)
        return (float(np.min(self.distance[segment, members])),
                float(self.adjacency[segment, members].sum()))


def _generation(partition, generation, seed, *, fractions=None,
                residual=0.0):
    n = len(partition.groups)
    values = np.ones(n) if fractions is None else np.asarray(fractions, float)
    return GenerationResult(
        partition=partition, generation=generation, seed=seed,
        fractions=values, fraction_mean=values.copy(),
        contribution=np.ones(n), transfer=np.zeros(n),
        replicator_residual=float(residual), movement=0.0,
        metrics={"loss_eval": 1.0}, wall_s=0.0)


def _trainer(matrix, config, runner):
    return EvoTrainer(matrix, config, runner)


def test_partition_stop_after_patience():
    calls = []
    matrix = _Matrix()

    def runner(partition, generation, seed):
        calls.append(partition)
        return _generation(partition, generation, seed)

    result = _trainer(matrix, EvoConfig(
        n_players=2, stop="partition", patience=2, seed=0), runner).run()
    assert result.stop_reason == "partition"
    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_nash_and_both_stop_after_one_clean_generation():
    matrix = _Matrix()

    def runner(partition, generation, seed):
        return _generation(partition, generation, seed)

    nash = _trainer(matrix, EvoConfig(
        n_players=2, stop="nash", seed=0), runner).run()
    both = _trainer(matrix, EvoConfig(
        n_players=2, stop="both", patience=1, seed=0), runner).run()
    assert nash.stop_reason == "nash"
    assert both.stop_reason == "both"
    assert len(nash.history) == 1
    assert len(both.history) == 1


def test_nash_failure_is_reported():
    matrix = _Matrix()

    def runner(partition, generation, seed):
        return _generation(partition, generation, seed, residual=0.1)

    result = _trainer(matrix, EvoConfig(
        n_players=2, stop="nash", seed=0), runner).run()
    assert result.stop_reason == "partition_stable_nash_failed"
    assert result.equilibrium["replicator_residual"] == 0.1


def test_min_players_and_all_dead_are_terminal():
    matrix = _Matrix()

    def one_alive(partition, generation, seed):
        fractions = np.ones(len(partition.groups))
        fractions[-1] = 0.0
        return _generation(partition, generation, seed, fractions=fractions)

    floor = _trainer(matrix, EvoConfig(
        n_players=2, min_players=2, stop="partition", seed=0),
        one_alive).run()
    assert floor.stop_reason == "min_players"
    assert floor.equilibrium["floor_hit"] is True

    def none_alive(partition, generation, seed):
        return _generation(partition, generation, seed,
                           fractions=np.zeros(len(partition.groups)))

    dead = _trainer(matrix, EvoConfig(
        n_players=2, min_players=1, stop="partition", seed=0),
        none_alive).run()
    assert dead.stop_reason == "all_dead"
    assert sorted(s for group in dead.final_partition.groups for s in group) == [
        0, 1, 2]


def test_max_generations_and_cycle():
    matrix = _Matrix()

    def stable(partition, generation, seed):
        return _generation(partition, generation, seed)

    capped = _trainer(matrix, EvoConfig(
        n_players=2, stop="partition", patience=10, generations=2,
        seed=0), stable).run()
    assert capped.stop_reason == "max_generations"
    assert len(capped.history) == 2

    first = PlayerPartition(((0,), (1, 2)))
    second = PlayerPartition(((0, 1), (2,)))
    sequence = [first, second, first, second]

    def alternating(partition, generation, seed):
        chosen = sequence[generation]
        return _generation(chosen, generation, seed)

    cycled = _trainer(matrix, EvoConfig(
        n_players=2, stop="partition", patience=10, generations=5,
        seed=0), alternating).run()
    assert cycled.stop_reason == "cycle"
