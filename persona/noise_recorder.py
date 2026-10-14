from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class NoiseSample:
    producer_id: str          # noise producer name (synapse / population / generator)
    t: float
    value: float              # delta or instantaneous observed fluctuation
    param: str                # "weight", "rate", "jitter", etc.


class NoiseRecorder:
    """
    Monitors and records noise producer fluctuations (e.g. STDP weight changes,
    Poisson rate fluctuations, input jitter). Exposes current step changes and
    historical statistics so Strategy / FisherMatrix can treat noise as an
    ordinary parameter affecting reward.
    """
    def __init__(self, name: str):
        self.name = name
        self.samples: List[NoiseSample] = []
        self._last_values: Dict[tuple, float] = {}
        self._step_deltas: Dict[str, float] = {}

    def observe(self, producer_id: str, t: float,
                param: str, current_value: float) -> NoiseSample | None:
        key = (producer_id, param)
        prev = self._last_values.get(key)
        self._last_values[key] = current_value
        if prev is None:
            delta = 0.0
        else:
            delta = float(current_value - prev)

        s = NoiseSample(producer_id=producer_id, t=t,
                        value=delta, param=param)
        self.samples.append(s)

        tag = f"noise_{producer_id}_{param}"
        self._step_deltas[tag] = delta
        return s

    def record_direct(self, producer_id: str, t: float, param: str, delta: float) -> NoiseSample:
        s = NoiseSample(producer_id=producer_id, t=t, value=float(delta), param=param)
        self.samples.append(s)
        tag = f"noise_{producer_id}_{param}"
        self._step_deltas[tag] = float(delta)
        return s

    def get_latest_step_deltas(self) -> Dict[str, float]:
        """
        Returns recent noise producer deltas for the current step.
        """
        return dict(self._step_deltas)

    def flush_distribution(self, window: float, now: float) -> Dict[str, dict]:
        """
        Computes mean and stddev of fluctuations within the given time window.
        """
        cutoff = now - window
        out: Dict[str, List[float]] = {}
        for s in self.samples:
            if s.t >= cutoff:
                key = f"noise_{s.producer_id}_{s.param}"
                out.setdefault(key, []).append(s.value)

        stats = {}
        for tag, vals in out.items():
            arr = np.asarray(vals, dtype=np.float64)
            if len(arr) > 0:
                stats[tag] = {
                    "mean": float(arr.mean()),
                    "sigma": float(arr.std() + 1e-8),
                    "n": len(arr),
                    "latest": float(arr[-1])
                }
            else:
                stats[tag] = {"mean": 0.0, "sigma": 1e-4, "n": 0, "latest": 0.0}
        return stats