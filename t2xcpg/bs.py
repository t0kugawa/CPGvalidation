import numpy as np

from t2xcpg.utils import poisson_to_parrot, clamp


class BrainStem:
    def __init__(self, bs_regular_hz: float = 60.0, bs_rate_min_hz: float = 0.0):
        self.bs_regular_hz = bs_regular_hz
        self.bs_rate_min_hz = bs_rate_min_hz

    def bs_input(self) -> float:
        input = max(self.bs_regular_hz, self.bs_rate_min_hz)
        return input


