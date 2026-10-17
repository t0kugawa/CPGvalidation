from math import exp


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class CUTAfferent:
    """
    Plantar cutaneous mechanoreceptor model.
    Converts contact pressure / touch area into firing rate (Hz)
    with first-order low-pass dynamics.

    TODO(cut-multi-site): replace this single per-leg afferent + scalar Hz with
    a multi-site plantar model (user-proposed, more physiological):
      - N Poisson generators distributed across the sole (heel / metatarsal
        pads / toe) instead of one scalar generator.
      - A site's generator joins the ACTIVE list only while that site is in
        contact with the floor; active generators drive RG-E through STDP.
      - Each site's rate derives from the site's contact AREA and the contact
        force the limb applies to that site (local force -> local Hz), so the
        aggregate CUT code is a nonlinear population code (foot roll / centre
        of pressure) rather than one sigmoid of total force.
      - Requires: a sole site layout, contact-point -> site assignment in
        model/leg.py (contacts are MuJoCo point contacts, so "area" must be a
        fixed per-site weighting), and per-site STDP synapses to RG-E.
      - The STDP postfix for these is already reserved: STDP_POSTFIX has an
        "ia_rg"/"cut_rge" family; add per-site keys when implementing.

    TODO(magic-numbers): the constants below (beta, p_theta, tau_ms, r_peak_hz,
    and the `contact * 50` pressure gain / 100 cap in get_cut_hz) are hand-tuned.
    Consider exposing sensor/actuator gains to PERSONA (see discussion) while
    keeping geometry/contact flags fixed.
    """
    def __init__(self, dt_ms: float = 2.0, r_peak_hz: float = 120.0,
                 beta: float = 0.3, p_theta: float = 10.0, tau_ms: float = 20.0):
        self.R = 0.0
        self.r_peak_hz = r_peak_hz
        self.dt = dt_ms
        self.beta = beta
        self.p_theta = p_theta
        self.tau = tau_ms

    def reset(self):
        self.R = 0.0

    def get_cut_hz(self, contact_force: float) -> float:
        if contact_force <= 1e-4:
            R_inst = 0.0
        else:
            # Pressure proportional to contact force
            P = min(100.0, contact_force * 50.0)
            R_inst = self.r_peak_hz / (1.0 + exp(-self.beta * (P - self.p_theta)))

        R_new = self.R + (self.dt / self.tau) * (R_inst - self.R)
        self.R = clamp(R_new, 0.0, self.r_peak_hz)
        return self.R


class IaAfferent:
    """
    Primary muscle spindle afferent model (group Ia).
    Sensitive to muscle length stretch (l - l_0) and positive lengthening velocity (v = dl/dt).
    Prevents excessive muscle stretching by exciting homonymous motor pool and reciprocal interneurons.
    """
    def __init__(self,
                 l_0: float = 0.035,
                 k_l: float = 800.0,
                 k_v: float = 150.0,
                 r_base_hz: float = 5.0,
                 r_max_hz: float = 250.0):
        self.l_0 = l_0
        self.k_l = k_l
        self.k_v = k_v
        self.r_base_hz = r_base_hz
        self.r_max_hz = r_max_hz

    def get_ia_hz(self, length: float, velocity: float) -> float:
        # Stretch beyond resting length
        stretch = max(0.0, length - self.l_0)
        # Lengthening velocity
        pos_vel = max(0.0, velocity)

        rate = self.r_base_hz + self.k_l * stretch + self.k_v * (pos_vel ** 0.6 if pos_vel > 0 else 0.0)
        return clamp(rate, 0.0, self.r_max_hz)


class IbAfferent:
    """
    Golgi tendon organ afferent model (group Ib).
    Sensitive to active muscle tension and tendon strain force.
    Acts defensively against excessive tension by inhibiting the homonymous motor pool.
    """
    def __init__(self,
                 k_force: float = 25.0,
                 k_act: float = 30.0,
                 r_base_hz: float = 2.0,
                 r_max_hz: float = 200.0):
        self.k_force = k_force
        self.k_act = k_act
        self.r_base_hz = r_base_hz
        self.r_max_hz = r_max_hz

    def get_ib_hz(self, force: float, activation: float) -> float:
        f = max(0.0, force)
        act = clamp(activation, 0.0, 1.0)
        rate = self.r_base_hz + self.k_force * f + self.k_act * act
        return clamp(rate, 0.0, self.r_max_hz)