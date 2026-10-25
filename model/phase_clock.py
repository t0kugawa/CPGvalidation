"""First-class circular phase state for the CPG/plant loop.

The contact detector gives a stance/swing state. This clock turns it into a
continuous circular phase with a period estimate, a bounded phase/frequency
PLL, confidence, and a bilateral offset target. The muscle E/F gate uses the
clock's stance state, so the reciprocal drive is not tied to raw contact
chatter.
"""
from __future__ import annotations

import math

TAU = 2.0 * math.pi


def wrap_pi(x: float) -> float:
    return (float(x) + math.pi) % TAU - math.pi


class PhaseClock:
    def __init__(self, dt_ms: float = 10.0, period_s: float = 1.0,
                 kp: float = 0.45, ki: float = 0.12,
                 min_period_s: float = 0.4, max_period_s: float = 2.0):
        self.dt_s = float(dt_ms) / 1000.0
        self.period_s = float(period_s)
        self.omega = TAU / self.period_s
        self.phase = 0.0
        self.stance = True
        self.confidence = 0.0
        self.error = 0.0
        self._last_stance = True
        self._t = 0.0
        self._last_onset = None
        self.kp = float(kp)
        self.ki = float(ki)
        self.min_omega = TAU / float(max_period_s)
        self.max_omega = TAU / float(min_period_s)

    def update(self, stance: bool, *, offset: float = 0.0,
               bilateral_error: float = 0.0) -> bool:
        stance = bool(stance)
        self._t += self.dt_s
        self.phase = (self.phase + self.omega * self.dt_s) % TAU

        if stance != self._last_stance:
            self._last_stance = stance
            self.stance = stance
            # stance onset -> 0, swing onset -> pi, plus the bilateral offset.
            target = (0.0 if stance else math.pi) + float(offset)
            err = wrap_pi(target - self.phase)
            self.phase = (self.phase + self.kp * err) % TAU
            self.omega = min(self.max_omega,
                             max(self.min_omega, self.omega + self.ki * err))
            self.error = err
            self.confidence = max(0.0, 1.0 - abs(err) / math.pi)
            if stance:
                if self._last_onset is not None:
                    p = self._t - self._last_onset
                    if 0.4 <= p <= 2.0:
                        self.period_s = 0.8 * self.period_s + 0.2 * p
                        self.omega = min(self.max_omega,
                                         max(self.min_omega,
                                             TAU / self.period_s))
                self._last_onset = self._t

        if bilateral_error:
            # Small bounded phase pull; no absolute setpoint is imposed on the
            # offset, only the measured/initial offset is held.
            self.phase = (self.phase + 0.04 * wrap_pi(bilateral_error)) % TAU

        # Export the PLL phase as the stance state; raw contact only drives the
        # phase error. This is what makes the E/F gate and the phase log use the
        # first-class state rather than contact chatter.
        self.stance = (self.phase % TAU) < math.pi
        return self.stance
