"""Memory-backed NEURON spike and weight recorders."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from neuron import h

from persona.neuron.cells import SpikeSource


class SpikeRecorder:
    """Records NetCon events and exposes the NEST recorder subset."""

    def __init__(self):
        self._items: List[tuple[int, Any, Any]] = []
        self._pending: Dict[str, np.ndarray] | None = None

    def attach(self, source: SpikeSource, sender_id: int) -> None:
        if source.obj is not None:
            nc = h.NetCon(source.obj, None)
        else:
            nc = h.NetCon(source.ref, None, sec=source.sec)
        nc.threshold = float(source.threshold)
        vec = h.Vector()
        nc.record(vec)
        self._items.append((int(sender_id), nc, vec))
        self._pending = None

    def _read(self) -> Dict[str, np.ndarray]:
        times: List[float] = []
        senders: List[int] = []
        for sender, _nc, vec in self._items:
            n = int(vec.size())
            if n <= 0:
                continue
            arr = np.asarray(vec.as_numpy(), dtype=np.float64)
            times.extend(arr.tolist())
            senders.extend([int(sender)] * int(arr.size))
        if times:
            order = np.argsort(np.asarray(times, dtype=np.float64),
                               kind="stable")
            t = np.asarray(times, dtype=np.float64)[order]
            s = np.asarray(senders, dtype=np.int64)[order]
        else:
            t = np.empty(0, dtype=np.float64)
            s = np.empty(0, dtype=np.int64)
        return {"times": t, "senders": s}

    def get(self, key: str) -> Any:
        events = self._read()
        if key in ("events", "all"):
            return events
        if key == "n_events":
            return int(events["times"].size)
        raise KeyError(f"SpikeRecorder has no field {key!r}")

    def set(self, values: Dict[str, Any]) -> None:
        if int(values.get("n_events", -1)) == 0:
            for _sender, _nc, vec in self._items:
                vec.resize(0)
            self._pending = None
            return
        raise KeyError(f"unsupported SpikeRecorder set {values!r}")

    def __len__(self) -> int:
        return int(self.get("n_events"))


class WeightRecorder:
    """Step-samples live ``NetCon.weight[0]`` values (memcpg's 10 ms recipe)."""

    def __init__(self, interval_ms: float = 10.0):
        self.interval_ms = float(interval_ms)
        self._items: List[dict] = []
        self._pending: List[tuple[float, float, int, int]] = []

    def attach(self, handle: Any) -> None:
        vec = h.Vector()
        if getattr(handle, "stdp", None) is not None:
            try:
                ref = handle.stdp._ref_synweight
            except AttributeError:
                ref = handle.netcon._ref_weight[0]
        else:
            ref = handle.netcon._ref_weight[0]
        vec.record(ref, self.interval_ms)
        self._items.append({
            "handle": handle,
            "vec": vec,
            "idx": 0,
            "t0": float(h.t),
            "last": float(handle.netcon.weight[0]),
            "sender": int(handle.presyn_id),
            "target": int(handle.postsyn_id),
        })

    def _read_new(self) -> None:
        for item in self._items:
            vec = item["vec"]
            n = int(vec.size())
            idx = int(item["idx"])
            if n <= idx:
                continue
            arr = np.asarray(vec.as_numpy(), dtype=np.float64)
            # ``Vector.record(ref, interval)`` stores only the values; sample
            # times are implicit: t0 + i * interval.
            for i in range(idx, n):
                w = float(arr[i])
                if w != float(item["last"]):
                    t = float(item["t0"]) + i * self.interval_ms
                    self._pending.append(
                        (t, w, int(item["sender"]), int(item["target"])))
                    item["last"] = w
            item["idx"] = n

    def get(self, key: str) -> Any:
        if key not in ("events", "all", "n_events"):
            raise KeyError(f"WeightRecorder has no field {key!r}")
        self._read_new()
        if key == "n_events":
            return len(self._pending)
        if self._pending:
            times = np.asarray([e[0] for e in self._pending], dtype=np.float64)
            weights = np.asarray([e[1] for e in self._pending],
                                 dtype=np.float64)
            senders = np.asarray([e[2] for e in self._pending],
                                 dtype=np.int64)
            targets = np.asarray([e[3] for e in self._pending],
                                 dtype=np.int64)
            order = np.argsort(times, kind="stable")
            times, weights = times[order], weights[order]
            senders, targets = senders[order], targets[order]
        else:
            times = np.empty(0, dtype=np.float64)
            weights = np.empty(0, dtype=np.float64)
            senders = np.empty(0, dtype=np.int64)
            targets = np.empty(0, dtype=np.int64)
        return {"times": times, "weights": weights,
                "senders": senders, "targets": targets}

    def set(self, values: Dict[str, Any]) -> None:
        if int(values.get("n_events", -1)) != 0:
            raise KeyError(f"unsupported WeightRecorder set {values!r}")
        self._pending = []
        for item in self._items:
            item["vec"].resize(0)
            item["idx"] = 0
            item["t0"] = float(h.t)
            item["last"] = float(item["handle"].netcon.weight[0])

    def __len__(self) -> int:
        return int(self.get("n_events"))
