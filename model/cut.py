from math import exp

import numpy as np

CUT_RATE_MAX_HZ = 120.0
CUT_ACTIVE_FORCE_N = 0.01


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class CUTAfferent:
    """
    Plantar cutaneous mechanoreceptor model.
    Converts contact pressure / touch area into firing rate (Hz)
    with first-order low-pass dynamics.

    TODO(cut-multi-site): replace this single per-leg afferent + scalar Hz with
    the user-specified multi-site plantar model.

    Why (observed in the 2-min run): the single sigmoid saturates at 120 Hz as
    soon as the foot carries ~0.4 N, so CUT is a 0/120 square wave that reports
    contact/no-contact only. It cannot encode foot roll, centre of pressure, or
    a site-specific phase for STDP. The aggregate 0/120 signal carries no
    foot-roll/CoP information and does not change with the ankle posture.

    Design (as specified):
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

    Implementation:
      - model/leg.py: assign each foot contact point to the nearest sole site in
        the foot body frame, sum the normal force per site, and emit per-site
        rates. The foot geom/contact mapping must be data-driven, not hardcoded
        site indices (T8).
      - model/adapter.py: extend CPGInput with per-site rates
        (`cut_sites_L`, `cut_sites_R`), keeping `cut_L`/`cut_R` as the aggregate
        for logging/back-compat.
      - t2xcpg/leg.py: create the per-site generators/relays and the per-site
        STDP connections; `update_sensory_feedback` takes the per-site rates.
      - Logging: per-site rates in `run_log`/`run.h5`, plus a figure of the
        heel->M1..M5->toe sequence against the ankle angle and the RG phase.

    Acceptance:
      - The aggregate CUT is not a square wave: it varies with local force/CoP
        within stance.
      - The heel site leads and the toe site lags within one stance, and the
        per-site STDP weights stay bounded.
      - tests/test_cut_sites.py covers contact->site assignment, the active
        list, and the per-site rate mapping.

    TODO(magic-numbers): the constants below (beta, p_theta, tau_ms, r_peak_hz,
    and the `contact * 50` pressure gain / 100 cap in get_cut_hz) are hand-tuned.
    Consider exposing sensor/actuator gains to PERSONA (see discussion) while
    keeping geometry/contact flags fixed.
    """
    def __init__(self, dt_ms: float = 2.0, r_peak_hz: float = CUT_RATE_MAX_HZ,
                 beta: float = 0.3, p_theta: float = 10.0, tau_ms: float = 20.0):
        self.R = 0.0
        self.r_peak_hz = r_peak_hz
        self.dt = dt_ms
        self.beta = beta
        self.p_theta = p_theta
        self.tau = tau_ms

    def reset(self):
        self.R = 0.0

    def get_cut_hz(self, contact_force: float, area: float = 1.0,
                   density: float = 1.0) -> float:
        """
        Site rate from local force, effective area, and receptor density.

        `area` is the site's tributary plantar area (relative; v1 uses 1.0 for
        every site). `density` is the receptor-density prior. Local pressure is
        force / area, and the area/density product scales the receptor count.
        The defaults reproduce the old single-site sigmoid exactly.
        """
        area = max(0.0, float(area))
        density = max(0.0, float(density))
        if contact_force <= 1e-4 or area <= 0.0 or density <= 0.0:
            R_inst = 0.0
        else:
            # Pressure from the site's local force and effective contact area.
            P = min(100.0, (float(contact_force) / area) * 50.0)
            R_inst = (self.r_peak_hz * density * area
                      / (1.0 + exp(-self.beta * (P - self.p_theta))))

        R_new = self.R + (self.dt / self.tau) * (R_inst - self.R)
        self.R = clamp(R_new, 0.0, self.r_peak_hz)
        return self.R


def aggregate_cut_hz(rates, areas, density, active) -> float:
    """
    Area x density weighted mean of the active-site rates.

    The aggregate stays on the 0-120 Hz CUT scale; inactive sites contribute
    nothing. With equal areas and densities this is the plain mean of the
    active rates.
    """
    rates = np.asarray(rates, dtype=float).ravel()
    areas = np.asarray(areas, dtype=float).ravel()
    density = np.asarray(density, dtype=float).ravel()
    active = np.asarray(active, dtype=bool).ravel()
    if not (rates.size == areas.size == density.size == active.size):
        raise ValueError("rates, areas, density, and active must have equal length")
    weights = np.clip(areas, 0.0, None) * np.clip(density, 0.0, None)
    mask = active & (weights > 0.0)
    denom = float(weights[mask].sum())
    if denom <= 0.0:
        return 0.0
    out = float(np.sum(weights[mask] * rates[mask]) / denom)
    return clamp(out, 0.0, CUT_RATE_MAX_HZ)


class PlantarCut:
    """
    Multi-site plantar CUT population for one controlled foot.

    Contact points are assigned to the nearest plantar site (world frame), the
    normal force is summed per site, and each active site's CUTAfferent emits a
    rate from its local force, effective area, and receptor density. The
    returned `active` array is the active buffer; inactive sites are reset so a
    re-contact starts from zero.
    """

    def __init__(self, site_names, dt_ms: float = 2.0, areas=None,
                 density=None, active_force_n: float = CUT_ACTIVE_FORCE_N,
                 r_peak_hz: float = CUT_RATE_MAX_HZ, beta: float = 0.3,
                 p_theta: float = 10.0, tau_ms: float = 20.0):
        self.site_names = tuple(site_names)
        self.n_sites = len(self.site_names)
        if self.n_sites == 0:
            raise ValueError("PlantarCut needs at least one site")
        self.areas = (np.ones(self.n_sites, dtype=float) if areas is None
                      else np.asarray(areas, dtype=float).ravel().copy())
        self.density = (np.ones(self.n_sites, dtype=float) if density is None
                        else np.asarray(density, dtype=float).ravel().copy())
        if self.areas.size != self.n_sites or self.density.size != self.n_sites:
            raise ValueError("areas and density must match the number of sites")
        if np.any(self.areas < 0.0) or np.any(self.density < 0.0):
            raise ValueError("areas and density must be non-negative")
        self.active_force_n = float(active_force_n)
        self.afferents = [
            CUTAfferent(dt_ms=dt_ms, r_peak_hz=r_peak_hz, beta=beta,
                        p_theta=p_theta, tau_ms=tau_ms)
            for _ in range(self.n_sites)
        ]
        self.forces = np.zeros(self.n_sites, dtype=float)
        self.rates = np.zeros(self.n_sites, dtype=float)
        self.active = np.zeros(self.n_sites, dtype=bool)
        self.aggregate_hz = 0.0

    def reset(self):
        for afferent in self.afferents:
            afferent.reset()
        self.forces[:] = 0.0
        self.rates[:] = 0.0
        self.active[:] = False
        self.aggregate_hz = 0.0

    def update(self, contact_positions, contact_forces, site_positions):
        """
        Parameters are world-frame contact positions (m, shape [n,3]), normal
        contact forces (N, shape [n]), and the site positions (m, shape [8,3]).
        Returns a dict with `forces`, `rates`, `active`, and `aggregate_hz`.
        """
        positions = np.asarray(contact_positions, dtype=float).reshape(-1, 3)
        forces = np.asarray(contact_forces, dtype=float).ravel()
        sites = np.asarray(site_positions, dtype=float).reshape(-1, 3)
        if forces.size != positions.shape[0]:
            raise ValueError("contact_positions and contact_forces must have equal length")
        if sites.shape != (self.n_sites, 3):
            raise ValueError(f"site_positions must have shape ({self.n_sites}, 3)")

        per_site = np.zeros(self.n_sites, dtype=float)
        for pos, force in zip(positions, forces):
            dist = np.linalg.norm(sites - pos, axis=1)
            j = int(np.argmin(dist))
            per_site[j] += abs(float(force))

        active = (self.areas > 0.0) & (per_site > self.active_force_n)
        rates = np.zeros(self.n_sites, dtype=float)
        for j, afferent in enumerate(self.afferents):
            if active[j]:
                rates[j] = afferent.get_cut_hz(
                    per_site[j], area=self.areas[j], density=self.density[j])
            else:
                afferent.reset()

        self.forces = per_site
        self.rates = rates
        self.active = active
        self.aggregate_hz = aggregate_cut_hz(
            rates, self.areas, self.density, active)
        return {
            "forces": per_site.copy(),
            "rates": rates.copy(),
            "active": active.copy(),
            "aggregate_hz": float(self.aggregate_hz),
        }


class ContactPhase:
    """
    Schmitt-trigger stance/swing phase with a debounce on the total contact
    force. Raw contact force crosses 5 mN many times within one stance; this
    detector requires the candidate state to persist for `debounce_ms`.
    """

    def __init__(self, on_n: float = 0.010, off_n: float = 0.005,
                 debounce_ms: float = 50.0, initial: str = "stance"):
        if initial not in ("stance", "swing"):
            raise ValueError("initial must be 'stance' or 'swing'")
        if on_n <= off_n:
            raise ValueError("on_n must be greater than off_n")
        self.on_n = float(on_n)
        self.off_n = float(off_n)
        self.debounce_ms = float(debounce_ms)
        self.initial = initial
        self.state = initial
        self._candidate = None
        self._elapsed_ms = 0.0

    def reset(self, state: str | None = None):
        self.state = self.initial if state is None else state
        if self.state not in ("stance", "swing"):
            raise ValueError("state must be 'stance' or 'swing'")
        self._candidate = None
        self._elapsed_ms = 0.0

    def update(self, force_n: float, dt_ms: float) -> str:
        force = float(force_n)
        if self.state == "swing" and force >= self.on_n:
            desired = "stance"
        elif self.state == "stance" and force <= self.off_n:
            desired = "swing"
        else:
            self._candidate = None
            self._elapsed_ms = 0.0
            return self.state

        if desired == self._candidate:
            self._elapsed_ms += float(dt_ms)
        else:
            self._candidate = desired
            self._elapsed_ms = float(dt_ms)
        if self._elapsed_ms >= self.debounce_ms:
            self.state = desired
            self._candidate = None
            self._elapsed_ms = 0.0
        return self.state


class IaAfferent:
    """
    Primary muscle spindle afferent model (group Ia).
    Sensitive to muscle length stretch (l - l_0), positive lengthening velocity
    (v = dl/dt) and tendon load (force), matching the reference spindle model
    (`ia = IA_BASE_HZ + IA_K_FORCE * force + IA_K_STRETCH * stretch`).
    The load term is what lets the spindle report weight-bearing during stance;
    the velocity term alone saturates between steps and reports swing only.
    Prevents excessive muscle stretching by exciting homonymous motor pool and reciprocal interneurons.
    """
    def __init__(self,
                 l_0: float = 0.014,
                 k_l: float = 2000.0,
                 k_v: float = 150.0,
                 k_force: float = 6.0,
                 r_base_hz: float = 10.0,
                 r_max_hz: float = 250.0):
        self.l_0 = float(l_0)
        self.k_l = k_l
        self.k_v = k_v
        self.k_force = k_force
        self.r_base_hz = r_base_hz
        self.r_max_hz = r_max_hz

    def set_reference_length(self, l_0: float):
        """Reference (resting) tendon length; set from the rig geometry."""
        self.l_0 = float(l_0)

    def get_ia_hz(self, length: float, velocity: float,
                  force: float = 0.0) -> float:
        # Stretch beyond the reference (resting) tendon length
        stretch = max(0.0, length - self.l_0)
        # Lengthening velocity
        pos_vel = max(0.0, velocity)
        # Tendon load
        load = max(0.0, force)

        rate = (self.r_base_hz + self.k_force * load + self.k_l * stretch
                + self.k_v * (pos_vel ** 0.6 if pos_vel > 0 else 0.0))
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