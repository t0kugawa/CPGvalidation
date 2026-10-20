import numpy as np


class NoiseRecorder:
    """
    Records one signal stream for PERSONA:

    - a **site** recorder holds the last explored sample `(x̂, σ)` applied at a
      personalizable parameter; it is never cleared, because the applied value
      stays in force until the next refresh step;
    - a **producer** recorder holds the per-step delta of an external noise
      source (STDP weight increments, sensory Poisson rate changes) and is
      read-and-clear, so a stale delta can never leak into a later step.

    A producer also carries the **current value of its signal at registration**
    `R` (STDP: Wmax, Poisson: rate), held fixed for the run. The strategy
    scores the delta as `δ/R²`; there is no estimation, no warm-up and no
    per-step gate. Missing/zero value means the producer has no influence.
    Statistics remain available for diagnostics only and are never used for
    scoring.
    """
    def __init__(self, name: str, beta: float = 0.995,
                 scale: float | None = None):
        self.name = name
        self.beta = float(beta)
        self.scale = (float(scale) if scale is not None and float(scale) > 0.0
                      else None)
        self.n = 0
        self.t = 0.0
        self._m: float | None = None
        self._q: float | None = None
        self._latest: float = 0.0
        self._latest_sample: tuple[float, float] | None = None
        self._pending: float | None = None

    def _observe(self, t: float, value: float) -> None:
        v = float(value)
        b = self.beta
        if self._m is None:
            self._m = v
            self._q = v * v
        else:
            self._m = b * self._m + (1.0 - b) * v
            self._q = b * self._q + (1.0 - b) * v * v
        self.n += 1
        self.t = float(t)
        self._latest = v

    def record_sample(self, t: float, sample: float, sigma: float) -> None:
        """Site recorder: the sample that was just applied, with its sigma."""
        self._latest_sample = (float(sample), float(sigma))
        self._observe(t, sample)

    def record_delta(self, t: float, delta: float) -> None:
        """Producer recorder: this step's delta (read once with `take_delta`)."""
        self._pending = float(delta)
        self._observe(t, delta)

    def latest_sample(self) -> tuple[float, float] | None:
        """Last applied `(sample, sigma)`; never cleared."""
        return self._latest_sample

    def take_delta(self) -> float | None:
        """Delta recorded since the previous take, or None; read-and-clear."""
        d = self._pending
        self._pending = None
        return d

    def stats(self) -> dict:
        """β-EWMA mean/variance of the recorded stream (constant cost)."""
        m = 0.0 if self._m is None else float(self._m)
        q = 0.0 if self._q is None else float(self._q)
        var = max(q - m * m, 0.0)
        return {
            "mean": m,
            "var": var,
            "std": float(np.sqrt(var)),
            "latest": float(self._latest),
            "t": float(self.t),
            "n": int(self.n),
        }
