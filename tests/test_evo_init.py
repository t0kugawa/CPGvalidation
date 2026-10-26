import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from persona.evo import random_partition, territory_partition


class _Matrix:
    n_segments = 6
    segments = ()
    adjacency = np.zeros((6, 6), dtype=float)
    for _left, _right in ((0, 1), (1, 2), (3, 4), (4, 5)):
        adjacency[_left, _right] = 1.0
        adjacency[_right, _left] = 1.0
    distance = shortest_path(csr_matrix(adjacency > 0.0), directed=False,
                             unweighted=True)

    def validate_partition(self, partition):
        expected = set(range(self.n_segments))
        seen = set()
        for group in partition.groups:
            if not group:
                raise ValueError("empty group")
            for segment in group:
                if segment in seen:
                    raise ValueError("duplicate segment")
                seen.add(segment)
        if seen != expected:
            raise ValueError("missing segment")


def test_territory_is_deterministic_and_covers_components():
    matrix = _Matrix()
    first = territory_partition(matrix, 3, seed=0)
    second = territory_partition(matrix, 3, seed=0)
    assert first == second
    first.validate(matrix)
    assert len(first.groups) == 3
    assert all(first.groups)
    # both connected components own at least one segment
    assert {segment < 3 for group in first.groups for segment in group} == {
        True, False}


def test_territory_rejects_too_many_or_too_few_players():
    matrix = _Matrix()
    with pytest.raises(ValueError):
        territory_partition(matrix, matrix.n_segments + 1, seed=0)
    with pytest.raises(ValueError):
        territory_partition(matrix, 1, seed=0)


def test_random_partition_is_deterministic_and_non_empty():
    matrix = _Matrix()
    first = random_partition(matrix, 3, seed=0)
    second = random_partition(matrix, 3, seed=0)
    assert first == second
    first.validate(matrix)
    assert len(first.groups) == 3
    assert all(first.groups)
