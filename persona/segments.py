"""Registry segment matrix: one registry object = one physiological segment.

A segment is a parameterized population or projection. The matrix encodes the
physical relations the game uses for territory: incidence, antagonism,
reciprocity, shared fan-out, and relay drive. It is deliberately independent
of the PERSONA update; the strategy remains site-level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from persona.homeostasis import (
    _connection_sign,
    _population_type,
    _synapse_size,
    _synapse_type,
)

EDGE_WEIGHTS = {
    "incidence": 1.0,
    "relay_drive": 1.0,
    "antagonism": 0.70,
    "reciprocity": 0.60,
    "shared_source": 0.40,
    "shared_target": 0.40,
}


@dataclass(frozen=True)
class Segment:
    """One parameterized registry object."""

    id: int
    name: str
    kind: str
    param_names: tuple[str, ...]
    size: int
    ptype: str
    source: str | None = None
    target: str | None = None
    model: str | None = None
    metadata: dict = field(default_factory=dict, compare=False)


@dataclass
class SegmentMatrix:
    """Segments, typed adjacency, and shortest-path distance."""

    segments: tuple[Segment, ...]
    segment_of_site: dict[str, int]
    site_names: tuple[str, ...]
    adjacency: np.ndarray
    edge_types: dict[tuple[int, int], frozenset[str]]
    distance: np.ndarray

    @classmethod
    def from_registry(cls, registry: Any, *,
                      include_shared: bool = True) -> "SegmentMatrix":
        segments = _extract_segments(registry)
        n = len(segments)
        adjacency = np.zeros((n, n), dtype=np.float64)
        edge_types: dict[tuple[int, int], set[str]] = {}

        def add_edge(left: int, right: int, kind: str) -> None:
            if left == right:
                return
            key = (min(left, right), max(left, right))
            edge_types.setdefault(key, set()).add(kind)
            weight = EDGE_WEIGHTS[kind]
            adjacency[left, right] = max(adjacency[left, right], weight)
            adjacency[right, left] = max(adjacency[right, left], weight)

        population_segments = {
            segment.name[4:]: segment.id
            for segment in segments
            if segment.kind == "population" and segment.name.startswith("pop_")
        }
        connection_items = [
            (segment, registry.connection_cfgs[segment.name[4:]])
            for segment in segments
            if segment.kind == "projection"
            and segment.name.startswith("con_")
            and segment.name[4:] in registry.connection_cfgs
        ]
        relay_links = {}
        relay_fn = getattr(registry, "relay_links", None)
        if callable(relay_fn):
            relay_links = dict(relay_fn())

        # 1. Incidence and relay drive.
        for segment, conn in connection_items:
            for endpoint in (str(conn.source), str(conn.target)):
                population_id = population_segments.get(endpoint)
                if population_id is not None:
                    add_edge(segment.id, population_id, "incidence")
            relay = relay_links.get(str(conn.source))
            if relay is not None:
                _pg_key, pop_name = relay
                population_id = population_segments.get(str(pop_name))
                if population_id is not None:
                    add_edge(segment.id, population_id, "relay_drive")

        # 2. Antagonism: opposite-sign incoming projections to one target.
        incoming: dict[str, list[tuple[int, int]]] = {}
        for segment, conn in connection_items:
            sign = _connection_sign(conn)
            if sign == 0:
                continue
            incoming.setdefault(str(conn.target), []).append(
                (segment.id, sign))
        for items in incoming.values():
            for (left, left_sign), (right, right_sign) in combinations(items, 2):
                if left_sign * right_sign < 0:
                    add_edge(left, right, "antagonism")

        # 3. Reciprocity: A->B and B->A projections.
        by_pair: dict[tuple[str, str], list[int]] = {}
        for segment, conn in connection_items:
            by_pair.setdefault(
                (str(conn.source), str(conn.target)), []).append(segment.id)
        for (source, target), forward in by_pair.items():
            backward = by_pair.get((target, source))
            if not backward:
                continue
            for left in forward:
                for right in backward:
                    add_edge(left, right, "reciprocity")

        if include_shared:
            # 4. Shared source and shared target fan-out.
            for conn_attr, kind in (("source", "shared_source"),
                                    ("target", "shared_target")):
                groups: dict[str, list[int]] = {}
                for segment, conn in connection_items:
                    groups.setdefault(str(getattr(conn, conn_attr)),
                                      []).append(segment.id)
                for members in groups.values():
                    for left, right in combinations(members, 2):
                        add_edge(left, right, kind)

        adjacency = 0.5 * (adjacency + adjacency.T)
        distance = _distance_matrix(adjacency)
        site_names = tuple(sorted(
            site for segment in segments for site in segment.param_names))
        segment_of_site = {
            site: segment.id
            for segment in segments
            for site in segment.param_names
        }
        if len(segment_of_site) != len(site_names):
            raise ValueError("a PERSONA site appears in more than one segment")
        return cls(
            segments=tuple(segments),
            segment_of_site=segment_of_site,
            site_names=site_names,
            adjacency=adjacency,
            edge_types={key: frozenset(value)
                        for key, value in edge_types.items()},
            distance=distance,
        )

    # --- geometry ------------------------------------------------------------

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_sites(self) -> int:
        return len(self.site_names)

    def player_link(self, segment_id: int,
                    player_segments: Iterable[int]) -> tuple[float, float]:
        """(shortest distance, summed edge weight) to one player's segments."""
        members = [int(s) for s in player_segments]
        if not members:
            return float("inf"), 0.0
        distances = self.distance[int(segment_id), members]
        weight = float(self.adjacency[int(segment_id), members].sum())
        return float(np.min(distances)), weight

    def nearest_segment(self, segment_id: int,
                        candidates: Iterable[int]) -> int:
        members = [int(s) for s in candidates]
        if not members:
            raise ValueError("nearest_segment needs at least one candidate")
        segment_id = int(segment_id)
        best = min(
            members,
            key=lambda member: (
                float(self.distance[segment_id, member]),
                -float(self.adjacency[segment_id, member]),
                int(member),
            ))
        return int(best)

    def nearest_player(self, segment_id: int,
                       player_segments: Iterable[Iterable[int]]) -> int:
        players = [list(int(s) for s in group) for group in player_segments]
        if not players:
            raise ValueError("nearest_player needs at least one player")
        scored = []
        for index, members in enumerate(players):
            distance, weight = self.player_link(segment_id, members)
            scored.append((distance, -weight, index))
        return int(min(scored)[2])

    # --- validation ----------------------------------------------------------

    def validate_partition(self, partition: Any) -> None:
        groups = getattr(partition, "groups", partition)
        groups = [tuple(int(s) for s in group) for group in groups]
        expected = set(range(self.n_segments))
        seen: set[int] = set()
        for group in groups:
            if not group:
                raise ValueError("player partition contains an empty group")
            for segment_id in group:
                if segment_id not in expected:
                    raise ValueError(
                        f"unknown segment id {segment_id} in partition")
                if segment_id in seen:
                    raise ValueError(
                        f"segment id {segment_id} appears in two players")
                seen.add(segment_id)
        missing = expected - seen
        if missing:
            raise ValueError(f"partition misses segment ids {sorted(missing)}")


def _extract_segments(registry: Any) -> list[Segment]:
    segments: list[Segment] = []
    for pop_name, pop in registry.population_cfgs.items():
        if not pop.params:
            continue
        segments.append(Segment(
            id=-1,
            name=f"pop_{pop_name}",
            kind="population",
            param_names=tuple(f"pop_{pop_name}_{name}"
                              for name in pop.params),
            size=int(pop.n),
            ptype=_population_type(pop_name, pop.model),
            model=str(pop.model),
            metadata={"population": str(pop_name)},
        ))
    for conn_key, conn in registry.connection_cfgs.items():
        if not conn.parameters:
            continue
        size = int(_synapse_size(registry, conn_key, conn))
        metadata: dict[str, Any] = {"connection": str(conn_key)}
        if size == 0:
            metadata["zero_synapse"] = True
        segments.append(Segment(
            id=-1,
            name=f"con_{conn_key}",
            kind="projection",
            param_names=tuple(f"conn_{conn_key}_{name}"
                              for name in conn.parameters),
            size=size,
            ptype=_synapse_type(conn),
            source=str(conn.source),
            target=str(conn.target),
            model=str(conn.model),
            metadata=metadata,
        ))
    segments.sort(key=lambda segment: segment.name)
    result = []
    for index, segment in enumerate(segments):
        result.append(Segment(
            id=index,
            name=segment.name,
            kind=segment.kind,
            param_names=segment.param_names,
            size=segment.size,
            ptype=segment.ptype,
            source=segment.source,
            target=segment.target,
            model=segment.model,
            metadata=segment.metadata,
        ))
    registry_sites = set(registry.get_all_optimizable_params())
    matrix_sites = {site for segment in result for site in segment.param_names}
    if registry_sites != matrix_sites:
        missing = sorted(registry_sites - matrix_sites)
        extra = sorted(matrix_sites - registry_sites)
        raise ValueError(
            f"segment matrix does not cover the registry: missing={missing} "
            f"extra={extra}")
    return result


def _distance_matrix(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    graph = csr_matrix((adjacency > 0.0).astype(np.int8))
    distance = shortest_path(graph, directed=False, unweighted=True)
    distance = np.asarray(distance, dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    return distance
