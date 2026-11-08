from math import exp

import numpy as np

from model.params import PARAMS

CUT_RATE_MAX_HZ = PARAMS["cut_rate_max_hz"]
CUT_ACTIVE_FORCE_N = PARAMS["cut_active_force_n"]


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class CUTAfferent:
    """
    Single plantar cutaneous mechanoreceptor (one sole site).

    Converts the site's local normal force into a firing rate (Hz) with a
    first-order low-pass. The instantaneous rate is a sigmoid of pressure
    (`force / area * pressure_gain`, capped), scaled by receptor count
    (`area * density`); zero force, area, or density gives zero.

    Multi-site architecture: `PlantarCut` owns one CUTAfferent per XML `rct_*`
    site, assigns each foot contact to the nearest site, sums the normal force
    per site, and marks a site active above `cut_active_force_n`. Inactive
    sites are reset and emit 0 Hz. It returns the per-site rates and an
    area x density weighted aggregate (`aggregate_cut_hz`) that stays on the
    0-120 Hz CUT scale.

    Data flow: `model/leg.py` discovers the sites from the MuJoCo model and
    fills `CPGInput.cut_sites_L/R` + `cut_active_L/R` (aggregate in `cut_L/R`);
    `t2xcpg` either drives one Poisson relay + STDP projection per site
    (`cut_multisite=True`) or routes the aggregate through the original single
    CUT pathway. All constants are hand-tuned priors in `model/params.py`,
    not fitted values or PERSONA-optimized parameters.
    """
    def __init__(self, dt_ms: float = PARAMS["cut_dt_ms"],
                 r_peak_hz: float = CUT_RATE_MAX_HZ,
                 beta: float = PARAMS["cut_beta"],
                 p_theta: float = PARAMS["cut_p_theta"],
                 tau_ms: float = PARAMS["cut_tau_ms"]):
        self.R = 0.0
        self.r_peak_hz = r_peak_hz
        self.dt = dt_ms
        self.beta = beta
        self.p_theta = p_theta
        self.tau = tau_ms

    def reset(self):
        self.R = 0.0

    def get_cut_hz(self, contact_force: float,
                   area: float = PARAMS["cut_site_area"],
                   density: float = PARAMS["cut_site_density"]) -> float:
        """
        Site rate from local force, effective area, and receptor density.

        `area` is the site's tributary plantar area (relative; v1 uses 1.0 for
        every site). `density` is the receptor-density prior. Local pressure is
        force / area, and the area/density product scales the receptor count.
        The defaults reproduce the old single-site sigmoid exactly.
        """
        area = max(0.0, float(area))
        density = max(0.0, float(density))
        if (contact_force <= PARAMS["cut_force_floor_n"]
                or area <= 0.0 or density <= 0.0):
            R_inst = 0.0
        else:
            # Pressure from the site's local force and effective contact area.
            P = min(PARAMS["cut_pressure_cap"],
                    (float(contact_force) / area) * PARAMS["cut_pressure_gain"])
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

    def __init__(self, site_names, dt_ms: float = PARAMS["cut_dt_ms"],
                 areas=None, density=None,
                 active_force_n: float = CUT_ACTIVE_FORCE_N,
                 r_peak_hz: float = CUT_RATE_MAX_HZ,
                 beta: float = PARAMS["cut_beta"],
                 p_theta: float = PARAMS["cut_p_theta"],
                 tau_ms: float = PARAMS["cut_tau_ms"]):
        self.site_names = tuple(site_names)
        self.n_sites = len(self.site_names)
        if self.n_sites == 0:
            raise ValueError("PlantarCut needs at least one site")
        self.areas = (np.full(self.n_sites, PARAMS["cut_site_area"], dtype=float)
                      if areas is None
                      else np.asarray(areas, dtype=float).ravel().copy())
        self.density = (np.full(self.n_sites, PARAMS["cut_site_density"],
                                dtype=float)
                        if density is None
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
    Schmitt-trigger stance/swing detector with force hysteresis only.

    There is intentionally no debounce timer: a timer is not allowed to
    control CPG stance/swing. The two force thresholds suppress ordinary
    chatter; phase dynamics and contact force decide the state.
    """

    def __init__(self, on_n: float = PARAMS["cut_contact_on_n"],
                 off_n: float = PARAMS["cut_contact_off_n"],
                 initial: str = PARAMS["cut_contact_initial"]):
        if initial not in ("stance", "swing"):
            raise ValueError("initial must be 'stance' or 'swing'")
        if on_n <= off_n:
            raise ValueError("on_n must be greater than off_n")
        self.on_n = float(on_n)
        self.off_n = float(off_n)
        self.initial = initial
        self.state = initial

    def reset(self, state: str | None = None):
        self.state = self.initial if state is None else state
        if self.state not in ("stance", "swing"):
            raise ValueError("state must be 'stance' or 'swing'")

    def update(self, force_n: float, dt_ms: float = 0.0) -> str:
        force = float(force_n)
        if self.state == "swing" and force >= self.on_n:
            self.state = "stance"
        elif self.state == "stance" and force <= self.off_n:
            self.state = "swing"
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
                 l_0: float = PARAMS["ia_default_l0_m"],
                 k_l: float = PARAMS["ia_default_k_l"],
                 k_v: float = PARAMS["ia_default_k_v"],
                 k_force: float = PARAMS["ia_default_k_force"],
                 r_base_hz: float = PARAMS["ia_default_base_hz"],
                 r_max_hz: float = PARAMS["ia_default_max_hz"]):
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
                + self.k_v * (pos_vel ** PARAMS["ia_velocity_exponent"]
                              if pos_vel > 0 else 0.0))
        return clamp(rate, 0.0, self.r_max_hz)


class IbAfferent:
    """
    Golgi tendon organ afferent model (group Ib).
    Sensitive to active muscle tension and tendon strain force.
    Acts defensively against excessive tension by inhibiting the homonymous motor pool.
    """
    def __init__(self,
                 k_force: float = PARAMS["ib_default_k_force"],
                 k_act: float = PARAMS["ib_default_k_act"],
                 r_base_hz: float = PARAMS["ib_default_base_hz"],
                 r_max_hz: float = PARAMS["ib_default_max_hz"]):
        self.k_force = k_force
        self.k_act = k_act
        self.r_base_hz = r_base_hz
        self.r_max_hz = r_max_hz

    def get_ib_hz(self, force: float, activation: float) -> float:
        f = max(0.0, force)
        act = clamp(activation, 0.0, 1.0)
        rate = self.r_base_hz + self.k_force * f + self.k_act * act
        return clamp(rate, 0.0, self.r_max_hz)