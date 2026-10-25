"""Fast tests for the first-class phase clock/PLL."""
from __future__ import annotations

from model.phase_clock import PhaseClock, wrap_pi


def test_clock_locks_a_periodic_contact_pattern():
    clock = PhaseClock(dt_ms=10.0, period_s=1.0)
    period = 0.9
    for i in range(600):
        t = i * 0.01
        clock.update((t % period) < period / 2)
    assert abs(clock.period_s - period) < 0.08, clock.period_s
    assert clock.confidence >= 0.0
    assert clock.stance in (True, False)


def test_clock_phase_error_is_bounded():
    clock = PhaseClock(dt_ms=10.0)
    clock.update(True)
    clock.update(False)
    assert abs(clock.error) <= 3.141592653589793
    assert -3.141592653589793 <= wrap_pi(clock.phase) <= 3.141592653589793
