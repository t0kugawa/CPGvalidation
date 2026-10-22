import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CosineLR:
    """Cosine learning-rate schedule over a fixed 1-based step horizon.

    ``at(1) == start`` and ``at(total_steps) == end`` exactly; later steps
    remain at ``end``. With ``warmup_steps > 0`` the first
    ``warmup_steps + 1`` steps hold ``start`` and the cosine covers the
    remaining intervals. A one-step schedule returns ``start``.
    """
    start: float
    end: float
    total_steps: int
    warmup_steps: int = 0

    def __post_init__(self):
        object.__setattr__(self, "start", float(self.start))
        object.__setattr__(self, "end", float(self.end))
        object.__setattr__(self, "total_steps", int(self.total_steps))
        object.__setattr__(self, "warmup_steps", int(self.warmup_steps))
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("start and end must be finite")
        if self.start < 0.0 or self.end < 0.0:
            raise ValueError("start and end must be non-negative")
        if self.total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")

    def at(self, step: int) -> float:
        """Learning rate for a 1-based strategy step."""
        step = int(step)
        if self.total_steps == 1:
            return self.start
        if step <= self.warmup_steps + 1:
            return self.start
        if step >= self.total_steps:
            return self.end
        p = ((step - self.warmup_steps - 1)
             / (self.total_steps - self.warmup_steps - 1))
        return self.end + 0.5 * (self.start - self.end) * (
            1.0 + math.cos(math.pi * p))


def resolve_lr_schedule(mode: str, start: float, end: float | None,
                        total_steps: int) -> CosineLR | None:
    """Build the configured schedule; ``constant`` means no schedule."""
    if mode == "constant":
        return None
    if mode == "cosine":
        if end is None:
            raise ValueError("lr_end is required when lr_mode is 'cosine'")
        return CosineLR(start=start, end=end, total_steps=total_steps)
    raise ValueError(f"unknown lr_mode {mode!r}")
