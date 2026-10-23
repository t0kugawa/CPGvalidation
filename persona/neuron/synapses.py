"""Static and plastic NEURON synapse factories.

The factory is the only place a ``NetCon`` is connected; the registry stores
the returned handle in a ``SynapseGroup``.  Signed PERSONA site values are
converted to the positive hardware conductance here, and the hard E/I
invariant is enforced again at every runtime set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

from neuron import h

from persona.neuron.cells import SpikeSource


@dataclass(frozen=True)
class SynapseContext:
    registry: Any
    conn_key: str
    source: str
    target: str
    presyn_cell: Any
    postsyn_cell: Any
    presyn_source: SpikeSource
    postsyn_pool: Any
    weight: float
    delay: float
    params: Dict[str, float] = field(default_factory=dict)
    weight_recorder: Any = None
    threshold: float = -20.0
    postsyn_source: SpikeSource | None = None
    presyn_id: int = 0
    postsyn_id: int = 0


@dataclass
class SynapseHandle:
    netcon: Any
    synapse: Any
    stdp: Any | None
    teaching_netcons: List[Any]
    presyn_id: int
    postsyn_id: int
    params: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)
    factory: Any = None
    sign: int = 1

    def set_signed_weight(self, value: float) -> None:
        self.factory.set_param(self, "weight", value)


def _apply_creation_params(factory, handle: "SynapseHandle",
                           params: Dict[str, float]) -> None:
    """Apply creation-only static values after the handle exists."""
    reserved = {"weight", "delay", "threshold", "weight_cv", "delay_cv"}
    for key, value in params.items():
        if key in reserved:
            continue
        path = str(key)
        if path in {"e", "tau", "tau1", "tau2"}:
            path = f"syn.{path}"
        factory.set_param(handle, path, float(value))


def _netcon(source: SpikeSource, target: Any, delay: float,
            weight: float, threshold: float):
    if source.obj is not None:
        nc = h.NetCon(source.obj, target)
    else:
        nc = h.NetCon(source.ref, target, sec=source.sec)
    nc.threshold = float(threshold)
    nc.delay = float(delay)
    nc.weight[0] = float(weight)
    return nc


@runtime_checkable
class SynapseFactory(Protocol):
    name: str
    kind: str
    plastic: bool
    weight_sign_mode: str

    def create(self, registry: Any, ctx: "SynapseContext") -> "SynapseHandle": ...
    def param_paths(self) -> set[str]: ...
    def set_param(self, handle: "SynapseHandle", path: str, value: float) -> None: ...
    def get_param(self, handle: "SynapseHandle", path: str) -> float: ...
    def weight(self, handle: "SynapseHandle") -> float: ...
    def set_weight(self, handle: "SynapseHandle", value: float) -> None: ...
    def delay(self, handle: "SynapseHandle") -> float: ...
    def set_delay(self, handle: "SynapseHandle", value: float) -> None: ...


class TapSynapseFactory:
    """Direct event connection into a readout relay (``IntFire4``)."""

    name = "tap_synapse"
    kind = "signed"
    pool_kind = "tap"
    plastic = False
    weight_sign_mode = "signed"

    def create(self, registry, ctx: SynapseContext) -> SynapseHandle:
        nc = _netcon(ctx.presyn_source, ctx.postsyn_cell, ctx.delay,
                     abs(float(ctx.weight)), ctx.threshold)
        return SynapseHandle(
            netcon=nc,
            synapse=None,
            stdp=None,
            teaching_netcons=[],
            presyn_id=int(ctx.presyn_id or registry._stable_id(ctx.presyn_cell)),
            postsyn_id=int(ctx.postsyn_id or registry._stable_id(ctx.postsyn_cell)),
            params=dict(ctx.params),
            factory=self,
            sign=1 if float(ctx.weight) >= 0.0 else -1,
        )

    def param_paths(self) -> set[str]:
        return {"weight", "signed_weight", "delay", "nc.threshold"}

    def set_param(self, handle: SynapseHandle, path: str, value: float) -> None:
        if path in {"weight", "signed_weight"}:
            handle.netcon.weight[0] = abs(float(value))
            handle.sign = 1 if float(value) >= 0.0 else -1
            handle.params[path] = float(value)
            return
        if path == "delay":
            handle.netcon.delay = float(value)
            handle.params[path] = float(value)
            return
        if path == "nc.threshold":
            handle.netcon.threshold = float(value)
            handle.params[path] = float(value)
            return
        raise AttributeError(f"tap synapse has no parameter {path!r}")

    def get_param(self, handle: SynapseHandle, path: str) -> float:
        if path == "weight":
            return float(handle.netcon.weight[0])
        if path == "signed_weight":
            return float(handle.sign) * float(handle.netcon.weight[0])
        if path == "delay":
            return float(handle.netcon.delay)
        if path == "nc.threshold":
            return float(handle.netcon.threshold)
        raise AttributeError(f"tap synapse has no parameter {path!r}")

    def weight(self, handle: SynapseHandle) -> float:
        return float(handle.netcon.weight[0])

    def set_weight(self, handle: SynapseHandle, value: float) -> None:
        self.set_param(handle, "weight", value)

    def delay(self, handle: SynapseHandle) -> float:
        return float(handle.netcon.delay)

    def set_delay(self, handle: SynapseHandle, value: float) -> None:
        self.set_param(handle, "delay", value)


class StaticSynapseFactory:
    """`ExpSyn` (exc) / `Exp2Syn` (inh) with signed PERSONA sites."""

    plastic = False
    weight_sign_mode = "signed"

    def __init__(self, name: str, kind: str, pool_kind: str | None = None):
        self.name = str(name)
        self.kind = "inh" if str(kind).lower() == "inh" else "exc"
        self.pool_kind = pool_kind or ("inh" if self.kind == "inh" else "exc")

    def _normalize(self, value: float) -> tuple[float, int]:
        value = float(value)
        if self.kind == "inh":
            if value > 0.0:
                value = -value
            return abs(value), -1
        if value < 0.0:
            value = abs(value)
        return value, 1

    def create(self, registry, ctx: SynapseContext) -> SynapseHandle:
        hw, sign = self._normalize(ctx.weight)
        nc = _netcon(ctx.presyn_source, ctx.postsyn_pool, ctx.delay, hw,
                     ctx.threshold)
        handle = SynapseHandle(
            netcon=nc,
            synapse=ctx.postsyn_pool,
            stdp=None,
            teaching_netcons=[],
            presyn_id=int(ctx.presyn_id or registry._stable_id(ctx.presyn_cell)),
            postsyn_id=int(ctx.postsyn_id or registry._stable_id(ctx.postsyn_cell)),
            params=dict(ctx.params),
            factory=self,
            sign=sign,
        )
        _apply_creation_params(self, handle, ctx.params)
        return handle

    def param_paths(self) -> set[str]:
        return {"weight", "delay", "signed_weight", "nc.threshold",
                "syn.tau", "syn.e", "syn.tau1", "syn.tau2"}

    def _require_syn(self, handle: SynapseHandle) -> Any:
        if handle.synapse is None:
            raise AttributeError("synapse handle has no point process")
        return handle.synapse

    def set_param(self, handle: SynapseHandle, path: str, value: float) -> None:
        path = str(path)
        if path in {"weight", "signed_weight"}:
            v = float(value)
            if self.kind == "inh":
                if v > 0.0:
                    raise ValueError(
                        "inhibitory synapse cannot accept a positive weight")
                handle.netcon.weight[0] = abs(v)
                handle.sign = -1
            else:
                if v < 0.0:
                    raise ValueError(
                        "excitatory synapse cannot accept a negative weight")
                handle.netcon.weight[0] = v
                handle.sign = 1
            handle.params[path] = v
            return
        if path == "delay":
            handle.netcon.delay = float(value)
            handle.params[path] = float(value)
            return
        if path == "nc.threshold":
            handle.netcon.threshold = float(value)
            handle.params[path] = float(value)
            return
        syn = self._require_syn(handle)
        if path.startswith("syn."):
            attr = path[4:]
            if self.kind == "inh" and attr == "tau":
                raise AttributeError(
                    "inhibitory Exp2Syn uses syn.tau1/syn.tau2, not syn.tau")
            if self.kind == "exc" and attr in {"tau1", "tau2"}:
                raise AttributeError(
                    "excitatory ExpSyn uses syn.tau, not syn.tau1/tau2")
            if attr not in {"tau", "tau1", "tau2", "e"}:
                raise AttributeError(f"unknown synapse attribute {attr!r}")
            setattr(syn, attr, float(value))
            handle.params[path] = float(value)
            return
        if path.startswith("stdp.") or path in {"tauhebb", "wmax", "Wmax"}:
            raise AttributeError("static synapse has no STDP mechanism")
        raise AttributeError(f"unknown synapse parameter {path!r}")

    def get_param(self, handle: SynapseHandle, path: str) -> float:
        path = str(path)
        if path == "weight":
            return float(handle.netcon.weight[0])
        if path == "signed_weight":
            return float(handle.sign) * float(handle.netcon.weight[0])
        if path == "delay":
            return float(handle.netcon.delay)
        if path == "nc.threshold":
            return float(handle.netcon.threshold)
        syn = self._require_syn(handle)
        if path.startswith("syn."):
            attr = path[4:]
            if not hasattr(syn, attr):
                raise AttributeError(f"synapse has no attribute {attr!r}")
            return float(getattr(syn, attr))
        if path in {"tauhebb", "wmax", "Wmax"} or path.startswith("stdp."):
            raise AttributeError("static synapse has no STDP mechanism")
        raise AttributeError(f"unknown synapse parameter {path!r}")

    def weight(self, handle: SynapseHandle) -> float:
        return float(handle.netcon.weight[0])

    def set_weight(self, handle: SynapseHandle, value: float) -> None:
        self.set_param(handle, "weight", value)

    def delay(self, handle: SynapseHandle) -> float:
        return float(handle.netcon.delay)

    def set_delay(self, handle: SynapseHandle, value: float) -> None:
        self.set_param(handle, "delay", value)


class StaticDispatcherFactory:
    """NEST-compatible ``static_synapse``: sign chooses the concrete type."""

    name = "static_synapse"
    plastic = False
    weight_sign_mode = "signed"
    kind = "signed"

    def __init__(self):
        self.exc = StaticSynapseFactory("static_exc", "exc")
        self.inh = StaticSynapseFactory("static_inh", "inh")

    @property
    def pool_kind(self) -> str:
        raise ValueError("static_synapse is resolved before pool allocation")

    def resolve(self, weight: float) -> StaticSynapseFactory:
        value = float(weight)
        if value > 0.0:
            return self.exc
        if value < 0.0:
            return self.inh
        raise ValueError(
            "static_synapse weight is zero; the excitatory/inhibitory "
            "identity is ambiguous")

    def create(self, registry, ctx: SynapseContext) -> SynapseHandle:
        return self.resolve(ctx.weight).create(registry, ctx)

    def param_paths(self) -> set[str]:
        return self.exc.param_paths() | self.inh.param_paths()

    def set_param(self, handle: SynapseHandle, path: str, value: float) -> None:
        handle.factory.set_param(handle, path, value)

    def get_param(self, handle: SynapseHandle, path: str) -> float:
        return handle.factory.get_param(handle, path)

    def weight(self, handle: SynapseHandle) -> float:
        return handle.factory.weight(handle)

    def set_weight(self, handle: SynapseHandle, value: float) -> None:
        handle.factory.set_weight(handle, value)

    def delay(self, handle: SynapseHandle) -> float:
        return handle.factory.delay(handle)

    def set_delay(self, handle: SynapseHandle, value: float) -> None:
        handle.factory.set_delay(handle, value)


class STDPSynapseFactory:
    """`ExpSyn`/`Exp2Syn` plus the memcpg ``STDP`` point process."""

    plastic = True
    weight_sign_mode = "signed"
    _ALIASES = {
        "Wmax": "wmax",
        "tau_plus": "tauhebb",
        "tau_minus": "tauanti",
        "lambda": "hebbwt",
        "alpha": "antiwt",
        "mu_plus": "hebbwt",
        "mu_minus": "antiwt",
    }

    def __init__(self, name: str, kind: str):
        self.name = str(name)
        self.kind = "inh" if str(kind).lower() == "inh" else "exc"
        self.pool_kind = "stdp_inh" if self.kind == "inh" else "stdp_exc"

    def _stdp_attr(self, handle: SynapseHandle) -> Any:
        if handle.stdp is None:
            raise RuntimeError(
                "STDP mechanism is unavailable; compile the reference "
                "stdp.mod (nrnivmodl) before using stdp_exc/stdp_inh")
        return handle.stdp

    def create(self, registry, ctx: SynapseContext) -> SynapseHandle:
        stdp_cls = getattr(h, "STDP", None)
        if stdp_cls is None:
            # Convenience: if the reference mechanisms were compiled in place
            # (the documented nrnivmodl recipe), load that library once.
            try:
                from pathlib import Path
                root = Path(__file__).resolve().parents[2]
                for candidate in (
                    root / "memcpg" / "mod_files" / "x86_64"
                    / "libnrnmech.so",
                    root / "memcpg" / "mod_files" / "x86_64"
                    / "libnrnmech.dylib",
                ):
                    if candidate.exists():
                        h.nrn_load_dll(str(candidate))
                        break
            except Exception:
                pass
            stdp_cls = getattr(h, "STDP", None)
        if stdp_cls is None:
            raise RuntimeError(
                "NEURON mechanism 'STDP' is not loaded. Compile the reference "
                "stdp.mod with nrnivmodl and load the resulting library "
                "before creating a plastic projection.")
        dummy = h.Section(name=f"stdp_dummy_{ctx.conn_key}_{len(ctx.params)}")
        stdpmech = stdp_cls(0, dummy)
        stdpmech.softthresh = 1
        stdpmech.STDPon = 1
        # The site is signed; the hardware point process only sees a positive
        # conductance.  Keep the inhibitory identity on the handle even though
        # the STDP mechanism itself works with positive weights.
        if self.kind == "inh":
            weight = abs(float(ctx.weight))
            sign = -1
        else:
            weight = abs(float(ctx.weight))
            sign = 1
        nc = _netcon(ctx.presyn_source, ctx.postsyn_pool, ctx.delay, weight,
                     ctx.threshold)
        # Apply creation-time STDP parameters.  Canonical names are bare
        # (tauhebb, wmax); NEST-style aliases are accepted.
        for key, value in ctx.params.items():
            attr = {"Wmax": "wmax", "stdp.wmax": "wmax"}.get(key, key)
            if attr.startswith("stdp."):
                attr = attr[5:]
            attr = self._ALIASES.get(attr, attr)
            if attr in {"tauhebb", "tauanti", "hebbwt", "antiwt", "wmax",
                        "softthresh", "STDPon"}:
                setattr(stdpmech, attr, float(value))
        if not hasattr(stdpmech, "wmax"):
            raise AttributeError("STDP mechanism has no wmax")
        h.setpointer(nc._ref_weight[0], "synweight", stdpmech)

        pre = _netcon(ctx.presyn_source, stdpmech, ctx.delay, 2.0,
                      ctx.threshold)
        post_source = ctx.postsyn_source
        if post_source is None:
            raise ValueError(
                "STDP factory needs a postsynaptic SpikeSource")
        post = _netcon(post_source, stdpmech, ctx.delay, -2.0,
                       post_source.threshold)
        handle = SynapseHandle(
            netcon=nc,
            synapse=ctx.postsyn_pool,
            stdp=stdpmech,
            teaching_netcons=[pre, post],
            presyn_id=int(ctx.presyn_id or registry._stable_id(ctx.presyn_cell)),
            postsyn_id=int(ctx.postsyn_id or registry._stable_id(ctx.postsyn_cell)),
            params=dict(ctx.params),
            meta={"dummy": dummy},
            factory=self,
            sign=sign,
        )
        _apply_creation_params(self, handle, ctx.params)
        return handle

    def param_paths(self) -> set[str]:
        return {"weight", "delay", "signed_weight", "nc.threshold",
                "syn.tau", "syn.e", "stdp.tauhebb", "stdp.tauanti",
                "stdp.hebbwt", "stdp.antiwt", "stdp.wmax", "stdp.softthresh",
                "tauhebb", "tauanti", "hebbwt", "antiwt", "wmax", "Wmax",
                "tau_plus", "tau_minus", "lambda", "alpha",
                "mu_plus", "mu_minus", "softthresh", "STDPon"}

    def _normalize_weight(self, value: float) -> tuple[float, int]:
        v = float(value)
        if self.kind == "inh":
            if v > 0.0:
                v = -v
            return abs(v), -1
        if v < 0.0:
            v = abs(v)
        return v, 1

    def set_param(self, handle: SynapseHandle, path: str, value: float) -> None:
        path = str(path)
        if path in {"weight", "signed_weight"}:
            hw, sign = self._normalize_weight(value)
            handle.netcon.weight[0] = hw
            handle.sign = sign
            handle.params[path] = float(value)
            return
        if path == "delay":
            handle.netcon.delay = float(value)
            handle.params[path] = float(value)
            return
        if path == "nc.threshold":
            handle.netcon.threshold = float(value)
            handle.params[path] = float(value)
            return
        syn = handle.synapse
        if path.startswith("syn."):
            attr = path[4:]
            if not hasattr(syn, attr):
                raise AttributeError(f"synapse has no attribute {attr!r}")
            setattr(syn, attr, float(value))
            handle.params[path] = float(value)
            return
        attr = path
        if attr.startswith("stdp."):
            attr = attr[5:]
        attr = {"Wmax": "wmax"}.get(attr, attr)
        attr = self._ALIASES.get(attr, attr)
        stdp = self._stdp_attr(handle)
        if not hasattr(stdp, attr):
            raise AttributeError(f"STDP mechanism has no attribute {attr!r}")
        setattr(stdp, attr, float(value))
        handle.params[path] = float(value)

    def get_param(self, handle: SynapseHandle, path: str) -> float:
        path = str(path)
        if path == "weight":
            return float(handle.netcon.weight[0])
        if path == "signed_weight":
            return float(handle.sign) * float(handle.netcon.weight[0])
        if path == "delay":
            return float(handle.netcon.delay)
        if path == "nc.threshold":
            return float(handle.netcon.threshold)
        syn = handle.synapse
        if path.startswith("syn."):
            attr = path[4:]
            if not hasattr(syn, attr):
                raise AttributeError(f"synapse has no attribute {attr!r}")
            return float(getattr(syn, attr))
        attr = path[5:] if path.startswith("stdp.") else path
        attr = {"Wmax": "wmax"}.get(attr, attr)
        attr = self._ALIASES.get(attr, attr)
        stdp = self._stdp_attr(handle)
        if not hasattr(stdp, attr):
            raise AttributeError(f"STDP mechanism has no attribute {attr!r}")
        return float(getattr(stdp, attr))

    def weight(self, handle: SynapseHandle) -> float:
        return float(handle.netcon.weight[0])

    def set_weight(self, handle: SynapseHandle, value: float) -> None:
        self.set_param(handle, "weight", value)

    def delay(self, handle: SynapseHandle) -> float:
        return float(handle.netcon.delay)

    def set_delay(self, handle: SynapseHandle, value: float) -> None:
        self.set_param(handle, "delay", value)


def builtin_synapse_factories() -> Dict[str, Any]:
    dispatcher = StaticDispatcherFactory()
    return {
        "static_exc": dispatcher.exc,
        "static_inh": dispatcher.inh,
        "static_synapse": dispatcher,
        "stdp_exc": STDPSynapseFactory("stdp_exc", "exc"),
        "stdp_synapse": STDPSynapseFactory("stdp_synapse", "exc"),
        "stdp_inh": STDPSynapseFactory("stdp_inh", "inh"),
    }
