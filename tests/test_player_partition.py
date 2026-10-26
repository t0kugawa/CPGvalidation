import pytest

from persona.partition import PlayerPartition
from persona.segments import Segment


class _Matrix:
    n_segments = 4
    segments = (
        Segment(0, "pop_A", "population", ("pop_A_a",), 4, "exc"),
        Segment(1, "pop_B", "population", ("pop_B_b",), 4, "exc"),
        Segment(2, "con_A_B", "projection", ("conn_A_B_weight",), 3,
                "static_exc", source="A", target="B"),
        Segment(3, "con_B_A", "projection", ("conn_B_A_weight",), 2,
                "static_exc", source="B", target="A"),
    )
    segment_of_site = {
        "pop_A_a": 0, "pop_B_b": 1,
        "conn_A_B_weight": 2, "conn_B_A_weight": 3,
    }

    def validate_partition(self, partition):
        expected = set(range(self.n_segments))
        seen = set()
        for group in partition.groups:
            if not group:
                raise ValueError("empty player group")
            for segment in group:
                if segment not in expected:
                    raise ValueError(f"unknown segment {segment}")
                if segment in seen:
                    raise ValueError("segment appears in two players")
                seen.add(segment)
        if seen != expected:
            raise ValueError(f"partition misses {sorted(expected - seen)}")


def test_partition_round_trip_and_site_groups():
    matrix = _Matrix()
    partition = PlayerPartition(groups=((0, 2), (1, 3)))
    partition.validate(matrix)
    assert partition.to_json()["groups"][0]["segments"] == [0, 2]
    assert PlayerPartition.from_json(partition.to_json()) == partition
    assert partition.site_groups(matrix) == [
        ("pop_A_a", "conn_A_B_weight"),
        ("pop_B_b", "conn_B_A_weight"),
    ]


def test_partition_physical_map():
    matrix = _Matrix()
    partition = PlayerPartition(groups=((0, 2), (1, 3)))
    physical = partition.physical_map(matrix)
    assert physical.n_players == 2
    assert physical.player_of["pop_A_a"] == 0
    assert physical.player_of["pop_B_b"] == 1
    assert physical.player_of["conn_A_B_weight"] == 0
    assert physical.specs[0].kind == "segment_group"


def test_partition_rejects_overlap_and_empty():
    matrix = _Matrix()
    with pytest.raises(ValueError):
        PlayerPartition(groups=((0, 1), (1, 2))).validate(matrix)
    with pytest.raises(ValueError):
        PlayerPartition(groups=((), (0, 1, 2, 3))).validate(matrix)
