"""Player partitions: named groups of whole registry segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from persona.homeostasis import PhysicalMap, PlayerSpec


@dataclass(frozen=True)
class PlayerPartition:
    groups: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        normalized = []
        for group in self.groups:
            members = tuple(sorted(int(segment) for segment in group))
            if len(set(members)) != len(members):
                raise ValueError(f"player group has duplicate segments: {members}")
            normalized.append(members)
        normalized.sort(key=lambda group: (len(group) == 0,
                                           min(group) if group else -1))
        object.__setattr__(self, "groups", tuple(normalized))

    def validate(self, matrix: Any) -> None:
        matrix.validate_partition(self)

    def to_json(self) -> dict:
        n_segments = (max((segment for group in self.groups
                           for segment in group), default=-1) + 1)
        return {
            "version": 1,
            "n_segments": int(n_segments),
            "groups": [
                {"name": f"player_{index:03d}", "segments": list(group)}
                for index, group in enumerate(self.groups)
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "PlayerPartition":
        if int(data.get("version", 1)) != 1:
            raise ValueError(f"unsupported partition version: {data.get('version')}")
        groups = []
        for item in data.get("groups", []):
            if isinstance(item, dict):
                groups.append(tuple(item.get("segments", [])))
            else:
                groups.append(tuple(item))
        return cls(groups=tuple(groups))

    def site_groups(self, matrix: Any) -> list[tuple[str, ...]]:
        self.validate(matrix)
        out = []
        for group in self.groups:
            sites = []
            for segment_id in group:
                sites.extend(matrix.segments[segment_id].param_names)
            out.append(tuple(sites))
        return out

    def physical_map(self, matrix: Any, *,
                     name_prefix: str = "player") -> PhysicalMap:
        self.validate(matrix)
        specs = []
        for index, group in enumerate(self.groups):
            param_names = []
            size = 0
            for segment_id in group:
                segment = matrix.segments[segment_id]
                param_names.extend(segment.param_names)
                size += int(segment.size)
            specs.append(PlayerSpec(
                name=f"{name_prefix}_{index:03d}",
                kind="segment_group",
                param_names=tuple(param_names),
                size=size,
                ptype="mixed",
            ))
        return PhysicalMap(specs)
