import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpringSpec:
    """One registry-created homeostatic spring.

    ``power`` is a constant per-step stiffness when given. When it is
    ``None`` the stiffness is dynamic: it follows the current mean learning
    rate and the site's variance ratio against its creation sigma.
    """
    name: str
    keys: tuple[str, ...]
    power: float | None
    gain: float
    max_power: float

    def power_for(self, sigma: float, sigma_ref: float, lr_t: float,
                  sigma_min: float = 0.0) -> float:
        if self.power is not None:
            return self.power
        lr = float(lr_t)
        if not math.isfinite(lr) or lr <= 0.0:
            return 0.0
        ref = float(sigma_ref)
        if not math.isfinite(ref) or ref <= 0.0:
            return 0.0
        sig = float(sigma)
        den = sig * sig if math.isfinite(sig) else 0.0
        floor = float(sigma_min)
        floor = floor * floor if math.isfinite(floor) else 0.0
        ratio = ref * ref / max(den, floor, 1e-12)
        return float(min(max(self.gain * lr * ratio, 0.0), self.max_power))
