"""Built-in NEURON cell factories and the registry-native path resolver.

A cell factory is the only sanctioned place for raw section/mechanism
creation.  It never connects cells; ``NeuronAdapter`` owns NetCon creation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np
from neuron import h

from persona.population import Population


@dataclass(frozen=True)
class SpikeSource:
    """Everything ``h.NetCon`` needs to read one cell's output.

    ``obj`` is used for artificial cells / ``NetStim`` (where NEURON accepts
    the object directly); for biophysical sections ``ref`` + ``sec`` are used.
    """

    ref: Any = None
    sec: Any = None
    threshold: float = -20.0
    obj: Any = None


@dataclass
class SectionCell:
    """Opaque built-in cell handle exposed to the registry/groups."""

    kind: str
    sections: Dict[str, List[Any]]
    syn_pools: Dict[str, List[Any]] = field(default_factory=dict)
    syn_used: Dict[str, int] = field(default_factory=dict)
    ident: int = 0
    threshold: float = -20.0

    @property
    def all(self) -> List[Any]:
        return [sec for secs in self.sections.values() for sec in secs]

    @property
    def soma(self):
        return (self.sections.get("soma") or [None])[0]

    @property
    def dend(self) -> List[Any]:
        return list(self.sections.get("dend") or [])

    @property
    def axon(self):
        return (self.sections.get("axon") or [None])[0]

    @property
    def muscle_unit(self):
        return (self.sections.get("muscle_unit") or [None])[0]

    @property
    def node(self) -> List[Any]:
        return list(self.sections.get("node") or [])

    @property
    def synlistex(self) -> List[Any]:
        return list(self.syn_pools.get("exc") or [])

    @property
    def synlistinh(self) -> List[Any]:
        return list(self.syn_pools.get("inh") or [])

    @property
    def synlistees(self) -> List[Any]:
        return list(self.syn_pools.get("exc") or [])

    @property
    def synlistexstdp(self) -> List[Any]:
        return list(self.syn_pools.get("stdp_exc") or [])

    @property
    def synlistinhstdp(self) -> List[Any]:
        return list(self.syn_pools.get("stdp_inh") or [])


@dataclass(frozen=True)
class PathSpec:
    """Parsed ``section[index](x).attribute`` PERSONA path."""

    section: str
    attr: str
    index: int | None = None
    x: float | None = None

    @property
    def name(self) -> str:
        return self.section


_SECTION_PATH = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)"
    r"(?:\[(?P<index>\d+)\])?"
    r"(?:\((?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\))?"
    r"\.(?P<attr>[A-Za-z_]\w*)$"
)


def parse_cell_path(path: str) -> PathSpec:
    match = _SECTION_PATH.match(str(path))
    if match is None:
        raise ValueError(
            f"cell parameter path {path!r} is not a section.attribute path")
    index = match.group("index")
    x_text = match.group("x")
    return PathSpec(
        section=match.group("name"),
        attr=match.group("attr"),
        index=None if index is None else int(index),
        x=None if x_text is None else float(x_text),
    )

_STRUCTURAL = {
    "nseg", "insert", "uninsert", "connect", "disconnect",
    "L", "diam", "pt3dadd", "pt3dclear", "pt3dchange",
}

_COMMON_ATTRS = (
    "L", "diam", "Ra", "cm", "nseg", "v", "ena", "ek",
    "el_hh", "gnabar_hh", "gkbar_hh", "gl_hh",
    "g_pas", "e_pas",
    "gnabar_fastchannels", "gkbar_fastchannels",
    "gl_fastchannels", "el_fastchannels",
    "gbar_nap", "gmax_cac1", "gbar_na14a", "gkbar_kir",
    "gkmax_K_No", "gcalbar_cal",
    "tau", "e", "tau1", "tau2",
)

# memcpg-style names are optional aliases of the canonical NEURON-native
# attributes used by the built-in simplified cell mechanisms.
_CELL_ALIASES = {
    "gnabar_fastchannels": "gnabar_hh",
    "gkbar_fastchannels": "gkbar_hh",
    "gl_fastchannels": "gl_hh",
    "el_fastchannels": "el_hh",
    "gnabar_motoneuron": "gnabar_hh",
    "gkrect_motoneuron": "gkbar_hh",
    "gl_motoneuron": "gl_hh",
    # The simplified built-in cells do not ship memcpg's special Ca/K
    # mechanisms.  The aliases still resolve to a real numeric RANGE on the
    # section so model code can keep its native paths.
    "gbar_nap": "g_pas",
    "gcaL_motoneuron": "g_pas",
    "gcak_motoneuron": "g_pas",
    "gmax_cac1": "g_pas",
    "gbar_na14a": "g_pas",
    "gkbar_kir": "g_pas",
    "gkmax_K_No": "g_pas",
    "gcalbar_cal": "g_pas",
}


def _option(cfg: Population, key: str, i: int, default):
    value = cfg.neuron.get(key, default)
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) <= i:
            raise ValueError(
                f"Population.neuron[{key!r}] has length {len(value)}, "
                f"expected at least {i + 1}")
        return value[i]
    return value


def _canonical_attr(attr: str) -> str:
    return _CELL_ALIASES.get(attr, attr)


class CellFactory:
    """Protocol-like base for built-in and registered cell types."""

    name = "cell"
    threshold = -20.0
    default_n_synapses = 80

    def create(self, registry, cfg: Population) -> "CellGroup":  # pragma: no cover
        raise NotImplementedError

    # --- attributes ---------------------------------------------------------
    def _resolve(self, cell: SectionCell, path: str) -> tuple[List[Any], float | None, str]:
        match = _SECTION_PATH.match(str(path))
        if match is None:
            raise ValueError(
                f"cell parameter path {path!r} is not a section.attribute path")
        name = match.group("name")
        if name not in cell.sections:
            raise ValueError(
                f"cell has no section named {name!r}; available: "
                f"{sorted(cell.sections)}")
        secs = list(cell.sections[name])
        if not secs:
            raise ValueError(f"cell has no sections named {name!r}")
        index = match.group("index")
        if index is not None:
            i = int(index)
            if i < 0 or i >= len(secs):
                raise ValueError(
                    f"section {name}[{i}] does not exist (have {len(secs)})")
            secs = [secs[i]]
        x_text = match.group("x")
        x = None if x_text is None else float(x_text)
        return secs, x, _canonical_attr(match.group("attr"))

    def param_paths(self, cell: SectionCell | None = None) -> set[str]:
        out: set[str] = set()
        if cell is None:
            return out
        for name, secs in cell.sections.items():
            for attr in _COMMON_ATTRS:
                if attr in _STRUCTURAL:
                    continue
                out.add(f"{name}.{attr}")
                for i in range(len(secs)):
                    out.add(f"{name}[{i}].{attr}")
        return out

    @staticmethod
    def _target(sec: Any, x: float | None) -> Any:
        """Section-level properties (Ra/L/diam) win over segment properties.

        ``sec(0.5)`` does not expose every section attribute, while ``sec``
        itself also exposes range variables such as ``gnabar_hh``.  Try the
        section first and fall back to the 0.5 segment only when needed.
        """
        return sec if x is None else sec(x)

    @staticmethod
    def _has_attr(sec: Any, attr: str, x: float | None) -> bool:
        target = CellFactory._target(sec, x)
        if hasattr(target, attr):
            return True
        if x is None:
            return hasattr(sec(0.5), attr)
        return False

    def has_param(self, cell: SectionCell, path: str) -> bool:
        try:
            secs, x, attr = self._resolve(cell, path)
            if attr in _STRUCTURAL:
                return False
            for sec in secs:
                if not self._has_attr(sec, attr, x):
                    return False
            return True
        except (AttributeError, IndexError, ValueError, TypeError):
            return False

    def get_param(self, cell: SectionCell, path: str) -> float:
        secs, x, attr = self._resolve(cell, path)
        sec = secs[0]
        target = self._target(sec, x)
        if not hasattr(target, attr) and x is None:
            target = sec(0.5)
        if not hasattr(target, attr):
            raise ValueError(
                f"section {sec} has no runtime attribute {attr!r}")
        return float(getattr(target, attr))

    def set_param(self, cell: SectionCell, path: str, value: float) -> None:
        secs, x, attr = self._resolve(cell, path)
        if attr in _STRUCTURAL:
            raise ValueError(
                f"cell parameter {path!r} is structural and cannot be set "
                "at runtime")
        val = float(value)
        for sec in secs:
            target = self._target(sec, x)
            if not hasattr(target, attr) and x is None:
                target = sec(0.5)
            if not hasattr(target, attr):
                raise ValueError(
                    f"section {sec} has no runtime attribute {attr!r}")
            setattr(target, attr, val)

    # --- synapses -----------------------------------------------------------
    def _pool_key(self, kind: str) -> str:
        kind = str(kind).lower()
        if kind in {"exc", "static_exc"}:
            return "exc"
        if kind in {"inh", "static_inh"}:
            return "inh"
        if kind in {"stdp_exc", "stdp_synapse"}:
            return "stdp_exc"
        if kind == "stdp_inh":
            return "stdp_inh"
        raise ValueError(f"unknown synapse pool kind {kind!r}")

    def synapse_pool(self, cell: SectionCell, kind: str) -> list[Any]:
        key = self._pool_key(kind)
        if key not in cell.syn_pools:
            raise KeyError(f"cell has no {key!r} synapse pool")
        return cell.syn_pools[key]

    def allocate_synapse(self, cell: SectionCell, kind: str) -> Any:
        key = self._pool_key(kind)
        pool = self.synapse_pool(cell, key)
        used = int(cell.syn_used.get(key, 0))
        if used >= len(pool):
            raise RuntimeError(
                f"{self.name} synapse pool {key!r} exhausted "
                f"({len(pool)} allocated); increase Population.neuron"
                "['n_synapses']")
        cell.syn_used[key] = used + 1
        return pool[used]

    def spike_source(self, cell: SectionCell) -> SpikeSource:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def stable_id(cell: SectionCell) -> int:
        return int(cell.ident)


def _make_syn_pools(sec, n: int) -> Dict[str, List[Any]]:
    """Allocate the memcpg-equivalent point-process pools on one segment."""
    n = max(0, int(n))
    loc = sec(0.5)
    pools: Dict[str, List[Any]] = {
        "exc": [], "inh": [], "stdp_exc": [], "stdp_inh": [],
    }
    for _ in range(n):
        s = h.ExpSyn(loc)
        s.tau = 0.1
        s.e = 50.0
        pools["exc"].append(s)

        s = h.Exp2Syn(loc)
        s.tau1 = 0.6
        s.tau2 = 2.2
        s.e = -70.0
        pools["inh"].append(s)

        s = h.ExpSyn(loc)
        s.tau = 1.3
        s.e = 55.0
        pools["stdp_exc"].append(s)

        s = h.Exp2Syn(loc)
        s.tau1 = 0.5
        s.tau2 = 2.8
        s.e = -70.0
        pools["stdp_inh"].append(s)
    return pools


def _new_section(name: str):
    return h.Section(name=name)


def _add_pas(sec, g_pas: float = 0.0002, e_pas: float = -70.0):
    sec.insert("pas")
    sec.g_pas = float(g_pas)
    sec.e_pas = float(e_pas)


def _add_hh(sec, gnabar: float = 0.12, gkbar: float = 0.036,
            gl: float = 0.0003, el: float = -54.3):
    sec.insert("hh")
    sec.gnabar_hh = float(gnabar)
    sec.gkbar_hh = float(gkbar)
    sec.gl_hh = float(gl)
    sec.el_hh = float(el)


class InterneuronFactory(CellFactory):
    name = "interneuron"

    def __init__(self, bursting: bool = False):
        self.bursting = bool(bursting)
        self.name = "bursting_interneuron" if bursting else "interneuron"

    def create(self, registry, cfg: Population) -> "CellGroup":
        from persona.neuron.groups import CellGroup

        n_syn = int(_option(cfg, "n_synapses", 0, self.default_n_synapses))
        n_dend_default = int(registry.rng.integers(2, 5))
        cells: List[SectionCell] = []
        for i in range(int(cfg.n)):
            n_dend = int(_option(cfg, "n_dend", i, n_dend_default))
            bursting = bool(_option(cfg, "bursting_mode", i, self.bursting))
            serotonin = bool(_option(cfg, "serotonin_mode", i, False))
            soma = _new_section(f"{cfg.name}_{i}_soma")
            axon = _new_section(f"{cfg.name}_{i}_axon")
            dend = [_new_section(f"{cfg.name}_{i}_dend{j}")
                    for j in range(max(0, n_dend))]

            soma.L = soma.diam = float(registry.rng.uniform(5.0, 15.0))
            soma.Ra = 100.0
            soma.cm = 1.0
            _add_hh(soma)
            if bursting:
                # NEURON has no built-in NaP; keep the fast channel set and
                # expose the same canonical paths.  A model supplying NaP can
                # register its own factory.
                soma.insert("pas")
                soma.g_pas = 0.0001
                soma.e_pas = -70.0
            for sec in dend:
                sec.L = 200.0
                sec.diam = float(max(0.1, registry.rng.normal(1.0, 0.1)))
                sec.Ra = 100.0
                sec.cm = 1.0
                _add_pas(sec)
                if serotonin:
                    _add_hh(sec, gnabar=0.05, gkbar=0.01,
                            gl=0.0001, el=-70.0)
                sec.connect(soma(0.5))
            axon.L = 150.0
            axon.diam = 1.0
            axon.Ra = 50.0
            axon.cm = 1.0
            _add_hh(axon, gnabar=0.5, gkbar=0.1, gl=0.01, el=-70.0)
            axon.nseg = 5
            axon.connect(soma(1))

            pools = _make_syn_pools(soma, n_syn)
            cells.append(SectionCell(
                kind=self.name,
                sections={"soma": [soma], "dend": dend, "axon": [axon]},
                syn_pools=pools,
                ident=registry._next_cell_id(),
                threshold=-20.0,
            ))
        return CellGroup(cfg.name, cells, self)

    def spike_source(self, cell: SectionCell) -> SpikeSource:
        sec = cell.sections["axon"][0]
        return SpikeSource(ref=sec(1.0)._ref_v, sec=sec, threshold=-20.0)


class BurstingInterneuronFactory(InterneuronFactory):
    def __init__(self):
        super().__init__(bursting=True)


class MotoneuronFactory(CellFactory):
    name = "motoneuron"

    def create(self, registry, cfg: Population) -> "CellGroup":
        from persona.neuron.groups import CellGroup

        n_syn = int(_option(cfg, "n_synapses", 0, self.default_n_synapses))
        cells: List[SectionCell] = []
        for i in range(int(cfg.n)):
            diam = float(_option(cfg, "diam", i, 30.0))
            soma = _new_section(f"{cfg.name}_{i}_soma")
            dend = _new_section(f"{cfg.name}_{i}_dend")
            axon = _new_section(f"{cfg.name}_{i}_axon")
            mysa = [_new_section(f"{cfg.name}_{i}_MYSA{j}")
                    for j in range(2)]
            flut = [_new_section(f"{cfg.name}_{i}_FLUT{j}")
                    for j in range(2)]
            stin = [_new_section(f"{cfg.name}_{i}_STIN{j}")
                    for j in range(2)]
            nodes = [_new_section(f"{cfg.name}_{i}_node{j}")
                     for j in range(4)]

            soma.L = soma.diam = max(1.0, diam)
            soma.Ra = 200.0
            soma.cm = 2.0
            # The simplified built-in uses hh for the same canonical paths.
            _add_hh(soma, gnabar=0.1 if diam > 50 else 0.05,
                    gkbar=0.2 if diam > 50 else 0.1,
                    gl=0.005 if diam > 50 else 0.0003,
                    el=-70.0)
            _add_pas(soma, g_pas=0.0002, e_pas=-70.0)
            for sec in mysa + flut + stin:
                sec.L = 10.0
                sec.diam = 1.0
                sec.Ra = 50.0
                sec.cm = 1.0
                _add_pas(sec, g_pas=0.0001, e_pas=-70.0)
            for sec in nodes:
                sec.L = 5.0
                sec.diam = 1.0
                sec.Ra = 50.0
                sec.cm = 1.0
                _add_hh(sec, gnabar=0.5, gkbar=0.1, gl=0.01, el=-70.0)
            dend.L = 200.0
            dend.diam = 1.0
            dend.Ra = 100.0
            dend.cm = 1.0
            _add_pas(dend, g_pas=0.0002, e_pas=-80.0)
            dend.connect(soma(0))
            axon.L = 150.0
            axon.diam = 1.0
            axon.Ra = 50.0
            axon.cm = 1.0
            _add_hh(axon, gnabar=0.5, gkbar=0.1, gl=0.01, el=-70.0)
            axon.nseg = 5
            axon.connect(soma(1))

            cells.append(SectionCell(
                kind=self.name,
                sections={
                    "soma": [soma], "dend": [dend], "axon": [axon],
                    "MYSA": mysa, "FLUT": flut, "STIN": stin,
                    "node": nodes,
                },
                syn_pools=_make_syn_pools(soma, n_syn),
                ident=registry._next_cell_id(),
                threshold=-20.0,
            ))
        return CellGroup(cfg.name, cells, self)

    def spike_source(self, cell: SectionCell) -> SpikeSource:
        sec = cell.sections["axon"][0]
        return SpikeSource(ref=sec(1.0)._ref_v, sec=sec, threshold=-20.0)


class MuscleFactory(CellFactory):
    name = "muscle"
    threshold = 10.0

    def create(self, registry, cfg: Population) -> "CellGroup":
        from persona.neuron.groups import CellGroup

        n_syn = int(_option(cfg, "n_synapses", 0, self.default_n_synapses))
        nseg = int(_option(cfg, "nseg", 0, 3))
        cells: List[SectionCell] = []
        for i in range(int(cfg.n)):
            unit = _new_section(f"{cfg.name}_{i}_muscle_unit")
            soma = _new_section(f"{cfg.name}_{i}_soma")
            for sec in (unit, soma):
                sec.L = 3000.0
                sec.diam = 40.0
                sec.nseg = max(1, nseg)
                sec.cm = 3.6
                sec.Ra = 1.1
                _add_pas(sec, g_pas=0.004, e_pas=-70.0)
                _add_hh(sec, gnabar=0.05, gkbar=0.01,
                        gl=0.01, el=-70.0)
            unit.connect(soma(0.5))
            cells.append(SectionCell(
                kind=self.name,
                sections={"muscle_unit": [unit], "soma": [soma]},
                syn_pools=_make_syn_pools(unit, n_syn),
                ident=registry._next_cell_id(),
                threshold=10.0,
            ))
        return CellGroup(cfg.name, cells, self)

    def spike_source(self, cell: SectionCell) -> SpikeSource:
        sec = cell.sections["muscle_unit"][0]
        return SpikeSource(ref=sec(0.5)._ref_v, sec=sec, threshold=10.0)


class AfferentFactory(CellFactory):
    name = "afferent"

    def create(self, registry, cfg: Population) -> "CellGroup":
        from persona.neuron.groups import CellGroup

        n_syn = int(_option(cfg, "n_synapses", 0, self.default_n_synapses))
        n_nodes_default = int(registry.rng.integers(8, 12))
        cells: List[SectionCell] = []
        for i in range(int(cfg.n)):
            n_nodes = int(_option(cfg, "n_axon_nodes", i, n_nodes_default))
            soma = _new_section(f"{cfg.name}_{i}_soma")
            axon = _new_section(f"{cfg.name}_{i}_axon")
            nodes = [_new_section(f"{cfg.name}_{i}_node{j}")
                     for j in range(max(1, n_nodes))]
            mysa = [_new_section(f"{cfg.name}_{i}_MYSA{j}")
                    for j in range(2)]
            flut = [_new_section(f"{cfg.name}_{i}_FLUT{j}")
                    for j in range(2)]
            stin = [_new_section(f"{cfg.name}_{i}_STIN{j}")
                    for j in range(2)]

            soma.L = soma.diam = float(registry.rng.uniform(15.0, 20.0))
            soma.Ra = 150.0
            soma.cm = 1.0
            _add_hh(soma, gnabar=0.65, gkbar=0.15, gl=0.0001, el=-65.0)
            axon.L = 150.0
            axon.diam = 1.0
            axon.Ra = 50.0
            axon.cm = 1.0
            _add_hh(axon, gnabar=0.5, gkbar=0.1, gl=0.01, el=-70.0)
            previous = soma(1.0)
            for group in (mysa, flut, stin):
                for sec in group:
                    sec.L = 10.0
                    sec.diam = 1.0
                    sec.Ra = 50.0
                    sec.cm = 1.0
                    _add_pas(sec, g_pas=0.0001, e_pas=-70.0)
                    sec.connect(previous)
                    previous = sec(1.0)
            for node in nodes:
                node.L = 5.0
                node.diam = 1.0
                node.Ra = 50.0
                node.cm = 1.0
                _add_hh(node, gnabar=0.5, gkbar=0.1, gl=0.01, el=-70.0)
                node.connect(previous)
                previous = node(1.0)

            cells.append(SectionCell(
                kind=self.name,
                sections={
                    "soma": [soma], "axon": [axon], "node": nodes,
                    "MYSA": mysa, "FLUT": flut, "STIN": stin,
                },
                syn_pools=_make_syn_pools(soma, n_syn),
                ident=registry._next_cell_id(),
                threshold=-20.0,
            ))
        return CellGroup(cfg.name, cells, self)

    def spike_source(self, cell: SectionCell) -> SpikeSource:
        nodes = cell.sections.get("node") or []
        sec = nodes[-1] if nodes else cell.sections["axon"][0]
        x = 1.0 if nodes else 0.5
        return SpikeSource(ref=sec(x)._ref_v, sec=sec, threshold=-20.0)


def builtin_cell_factories() -> Dict[str, CellFactory]:
    return {
        "interneuron": InterneuronFactory(),
        "bursting_interneuron": BurstingInterneuronFactory(),
        "motoneuron": MotoneuronFactory(),
        "muscle": MuscleFactory(),
        "afferent": AfferentFactory(),
    }
