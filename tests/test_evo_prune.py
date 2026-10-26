import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from persona.evo import prune_and_redistribute
from persona.partition import PlayerPartition


class _Matrix:
    def __init__(self, n, edges):
        self.n_segments = n
        self.adjacency = np.zeros((n, n), dtype=float)
        for left, right in edges:
            self.adjacency[left, right] = 1.0
            self.adjacency[right, left] = 1.0
        self.distance = shortest_path(csr_matrix(self.adjacency > 0.0),
                                      directed=False, unweighted=True)
        self.edge_types = {
            (min(a, b), max(a, b)): frozenset({"incidence"})
            for a, b in edges
        }

    def validate_partition(self, partition):
        seen = set()
        for group in partition.groups:
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


def test_prune_reassigns_every_dead_segment():
    matrix = _Matrix(3, [(0, 1), (1, 2)])
    partition = PlayerPartition(((0,), (1,), (2,)))
    new, log, reason = prune_and_redistribute(
        partition, np.array([1.0, 0.0, 0.0]), matrix, seed=0,
        impact_eps=0.0, min_players=1)
    assert reason == "pruned"
    assert len(new.groups) == 1
    assert sorted(segment for group in new.groups for segment in group) == [
        0, 1, 2]
    assert {entry.segment for entry in log} == {1, 2}
    assert all(entry.to_player == 0 for entry in log)


def test_tie_is_deterministic_and_all_segments_survive():
    matrix = _Matrix(4, [(0, 1), (0, 2), (0, 3)])
    partition = PlayerPartition(((0,), (1,), (2, 3)))
    fractions = np.array([0.0, 1.0, 1.0])
    first = prune_and_redistribute(partition, fractions, matrix, seed=7,
                                   min_players=1)
    second = prune_and_redistribute(partition, fractions, matrix, seed=7,
                                    min_players=1)
    assert first[0] == second[0]
    assert first[1][0].to_player == second[1][0].to_player
    assert sorted(s for group in first[0].groups for s in group) == [0, 1, 2, 3]


def test_disconnected_segment_uses_flagged_fallback():
    matrix = _Matrix(4, [(0, 1), (1, 2)])
    partition = PlayerPartition(((0,), (1, 2), (3,)))
    new, log, reason = prune_and_redistribute(
        partition, np.array([1.0, 1.0, 0.0]), matrix, seed=0,
        impact_eps=0.0, min_players=1)
    assert reason == "pruned"
    assert log[0].segment == 3
    assert log[0].fallback is True
    assert 3 in [s for group in new.groups for s in group]


def test_min_players_floor_returns_input():
    matrix = _Matrix(3, [(0, 1), (1, 2)])
    partition = PlayerPartition(((0,), (1,), (2,)))
    new, log, reason = prune_and_redistribute(
        partition, np.array([1.0, 0.0, 0.0]), matrix, seed=0,
        impact_eps=0.0, min_players=2)
    assert reason == "min_players"
    assert new == partition
    assert log == []


def test_all_dead_keeps_one_covering_player():
    matrix = _Matrix(3, [(0, 1), (1, 2)])
    partition = PlayerPartition(((0,), (1,), (2,)))
    new, log, reason = prune_and_redistribute(
        partition, np.zeros(3), matrix, seed=0, impact_eps=0.0,
        min_players=1)
    assert reason == "all_dead"
    assert sorted(s for group in new.groups for s in group) == [0, 1, 2]
