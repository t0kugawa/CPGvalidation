"""Duck-compatible group objects exposed by the NEURON adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

from persona.neuron.cells import SectionCell, SpikeSource


def _broadcast(value, n: int, what: str) -> List[Any]:
    arr = np.asarray(value, dtype=object).ravel()
    if arr.size == 0:
        raise ValueError(f"{what} value is empty")
    if arr.size == 1:
        return [arr[0]] * n
    if arr.size != n:
        raise ValueError(
            f"{what} sequence has length {arr.size}, expected {n}")
    return list(arr)


@dataclass
class CellGroup:
    name: str
    cells: List[SectionCell]
    cell_type: Any

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[SectionCell]:
        return iter(self.cells)

    def __getitem__(self, index: int) -> SectionCell:
        return self.cells[index]

    @property
    def ids(self) -> List[int]:
        return [self.cell_type.stable_id(cell) for cell in self.cells]

    def set(self, values: Dict[str, Any]) -> None:
        for path, value in values.items():
            vals = _broadcast(value, len(self.cells), path)
            for cell, val in zip(self.cells, vals):
                self.cell_type.set_param(cell, path, float(val))

    def get(self, name: str) -> np.ndarray:
        return np.asarray(
            [self.cell_type.get_param(cell, name) for cell in self.cells],
            dtype=np.float64,
        )

    def has_param(self, name: str) -> bool:
        return all(self.cell_type.has_param(cell, name) for cell in self.cells)

    def param_paths(self) -> set[str]:
        out: set[str] = set()
        for cell in self.cells:
            out.update(self.cell_type.param_paths(cell))
        return out

    def spike_source(self, cell: SectionCell | None = None) -> Any:
        if cell is None:
            if len(self.cells) != 1:
                return [self.cell_type.spike_source(c) for c in self.cells]
            cell = self.cells[0]
        return self.cell_type.spike_source(cell)


@dataclass
class SynapseGroup:
    key: str
    source: str
    target: str
    model: str
    handles: List[Any]
    weight_recorder: Any = None
    factory: Any = None

    def __len__(self) -> int:
        return len(self.handles)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.handles)

    def __getitem__(self, index: int) -> Any:
        return self.handles[index]

    def set(self, values: Dict[str, Any]) -> None:
        for path, value in values.items():
            vals = _broadcast(value, len(self.handles), path)
            for handle, val in zip(self.handles, vals):
                self.factory.set_param(handle, path, float(val))

    def get(self, name: str) -> np.ndarray:
        return np.asarray(
            [self.factory.get_param(handle, name) for handle in self.handles],
            dtype=np.float64,
        )

    def param_paths(self) -> set[str]:
        return set(self.factory.param_paths())


@dataclass
class RateGroup:
    name: str
    stims: List[Any]
    ids: List[int] = field(default_factory=list)
    _last_number: int = 1_000_000_000

    def __post_init__(self):
        if not self.ids:
            self.ids = list(range(len(self.stims)))

    def __len__(self) -> int:
        return len(self.stims)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.stims)

    def __getitem__(self, index: int) -> Any:
        return self.stims[index]

    def _apply(self, name: str, value) -> None:
        vals = _broadcast(value, len(self.stims), name)
        for stim, val in zip(self.stims, vals):
            if name == "rate":
                hz = float(val)
                if not np.isfinite(hz) or hz < 0.0:
                    raise ValueError(f"rate must be finite and >= 0, got {hz}")
                if hz <= 0.0:
                    if int(stim.number) > 0:
                        self._last_number = int(stim.number)
                    stim.number = 0
                    continue
                if int(stim.number) == 0:
                    stim.number = int(self._last_number)
                stim.interval = 1000.0 / hz
            elif name == "interval":
                interval = float(val)
                if not np.isfinite(interval) or interval <= 0.0:
                    raise ValueError(
                        f"interval must be finite and > 0, got {interval}")
                stim.interval = interval
                if int(stim.number) == 0:
                    stim.number = int(self._last_number)
            elif name == "number":
                number = int(val)
                if number < 0:
                    raise ValueError("number must be >= 0")
                stim.number = number
                if number > 0:
                    self._last_number = number
            elif name == "start":
                stim.start = float(val)
            elif name == "noise":
                stim.noise = float(val)
            else:
                raise AttributeError(f"RateGroup has no settable {name!r}")

    def set(self, values: Dict[str, Any]) -> None:
        for name, value in values.items():
            self._apply(name, value)

    def get(self, name: str) -> np.ndarray:
        if name == "rate":
            return np.asarray([
                0.0 if int(stim.number) == 0 else 1000.0 / float(stim.interval)
                for stim in self.stims
            ], dtype=np.float64)
        if name == "interval":
            return np.asarray([float(s.interval) for s in self.stims],
                              dtype=np.float64)
        if name == "number":
            return np.asarray([int(s.number) for s in self.stims],
                              dtype=np.float64)
        if name == "start":
            return np.asarray([float(s.start) for s in self.stims],
                              dtype=np.float64)
        if name == "noise":
            return np.asarray([float(s.noise) for s in self.stims],
                              dtype=np.float64)
        raise AttributeError(f"RateGroup has no readable {name!r}")


@dataclass
class RelayGroup:
    name: str
    cells: List[Any]
    kind: str = "forward"       # "forward" | "tap"
    ids: List[int] = field(default_factory=list)
    rate_group: RateGroup | None = None

    def __post_init__(self):
        if not self.ids:
            self.ids = list(range(len(self.cells)))

    @property
    def can_source(self) -> bool:
        return self.kind == "forward"

    @property
    def can_target(self) -> bool:
        return self.kind == "tap"

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.cells)

    def __getitem__(self, index: int) -> Any:
        return self.cells[index]

    def set(self, values: Dict[str, Any]) -> None:
        # Forwarding relays are NetStims; their rate is owned by RateGroup.
        if self.rate_group is not None:
            self.rate_group.set(values)
            return
        raise AttributeError("readout relay has no settable parameters")

    def get(self, name: str) -> np.ndarray:
        if self.rate_group is not None:
            return self.rate_group.get(name)
        raise AttributeError(f"readout relay has no readable {name!r}")

    def spike_source(self, cell: Any | None = None) -> Any:
        if cell is None:
            if len(self.cells) != 1:
                return [self.spike_source(c) for c in self.cells]
            cell = self.cells[0]
        # NetStim and IntFire accept a direct object in h.NetCon.  A section
        # cell is wrapped by its factory, but relays only expose the former.
        return SpikeSource(obj=cell, threshold=-20.0)
