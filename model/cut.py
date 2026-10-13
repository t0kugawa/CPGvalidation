from math import exp
from leg.model_cfg import OBJECT_MASS
from utils import clamp

E: float = 0.02
BETA: float = 0.3
P_THETA: float = 10.0
TAU: float = 20.0


class CUTAfferent:
    def __init__(self, dt: float, r_peak_hz: float):
        self.R = 0.0
        self.r_peak_hz = r_peak_hz
        self.dt = dt

    def reset(self):
        self.R = 0.0

    def get_cut_hz(self, touch_area: float) -> float:
        P = OBJECT_MASS / (touch_area + E * touch_area)
        R_inst = self.r_peak_hz / (1 + exp(-BETA * (P - P_THETA)))

        R_new = self.R + (self.dt / TAU) * (R_inst - self.R)
        R_new = clamp(R_new, 0, self.r_peak_hz)

        self.R = R_new
        return R_new