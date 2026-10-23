"""Relational homeostatic regulator for PERSONA's natural-gradient strategy.

The physical map is the architectural half of the regulator: it turns the
registry's populations and connections into strategy-bearing players, builds
the player graph from the physical connectivity, and exposes the zero-row-sum
combinatorial graph operator used by H.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from persona.connection import Connection

if TYPE_CHECKING:
    from persona.registry import Registry


_INHIBITORY_PREFIXES = ("in_", "in1", "r_", "ia_", "v0d_")
_STDP_PARAMETER_NAMES = {
    "tau_plus", "tau_minus", "lambda", "alpha",
    "mu_plus", "mu_minus", "Wmax",
}


@dataclass(frozen=True)
class PlayerSpec:
    """One strategy-bearing unit in the physical game."""

    name: str
    kind: str                       # population | synapse | unknown
    param_names: tuple[str, ...]
    size: int = 1
    ptype: str = "exc"              # exc | inh | poisson | static_exc | static_inh | stdp
    source: str | None = None
    target: str | None = None
    metadata: dict = field(default_factory=dict, compare=False)

    @property
    def m_count(self) -> int:
        """Number of optimizable mean coordinates (``M_i``)."""
        return len(self.param_names)


class PhysicalMap:
    """Players and the degree-scaled graph operator built from the registry.

    The graph has one node per player. Incidence, antagonism, and reciprocity
    propose edges; the adjacency is simple (a pair is either connected or not),
    so duplicate proposals do not accumulate weight. The explicit ``eps_H``
    term in ``H`` is the weak global regularizer; the adjacency itself is not
    completed.
    """

    def __init__(self, specs: list[PlayerSpec], edges=(), *,
                 eps_H: float = 1e-3,
                 skipped_endpoints=(),
                 connection_keys: dict[str, str] | None = None):
        self.specs = list(specs)
        self.eps_H = float(eps_H)
        self._skipped_endpoints = sorted({str(v) for v in skipped_endpoints})
        self.connection_keys = dict(connection_keys or {})
        self.player_of: dict[str, int] = {}
        for i, spec in enumerate(self.specs):
            for site in spec.param_names:
                if site in self.player_of:
                    raise ValueError(f"site {site!r} belongs to more than one player")
                self.player_of[site] = i

        n = len(self.specs)
        self.adjacency = np.zeros((n, n), dtype=np.float64)
        name_to_idx = {spec.name: i for i, spec in enumerate(self.specs)}
        if len(name_to_idx) != n:
            raise ValueError("player names must be unique")
        for edge in edges:
            a, b = edge
            i = name_to_idx.get(a, a) if isinstance(a, str) else int(a)
            j = name_to_idx.get(b, b) if isinstance(b, str) else int(b)
            if isinstance(i, str) or isinstance(j, str):
                raise ValueError(f"edge endpoints not in map: {edge!r}")
            self._add_edge(int(i), int(j))
        self._H = None

    # --- construction --------------------------------------------------------

    @classmethod
    def from_registry(cls, registry: Registry, *,
                      eps_H: float = 1e-3) -> "PhysicalMap":
        specs: list[PlayerSpec] = []
        pop_player: dict[str, int] = {}

        for pop_name, pop in registry.population_cfgs.items():
            if not pop.params:
                continue
            spec = PlayerSpec(
                name=f"pop_{pop_name}",
                kind="population",
                param_names=tuple(f"pop_{pop_name}_{p}" for p in pop.params),
                size=int(pop.n),
                ptype=_population_type(pop_name, pop.model),
            )
            pop_player[pop_name] = len(specs)
            specs.append(spec)

        conn_player: dict[str, int] = {}
        conn_sign: dict[str, int] = {}
        connection_keys: dict[str, str] = {}
        for conn_key, conn in registry.connection_cfgs.items():
            if not conn.parameters:
                continue
            player_key = f"con_{conn_key}"
            spec = PlayerSpec(
                name=player_key,
                kind="synapse",
                param_names=tuple(f"conn_{conn_key}_{p}" for p in conn.parameters),
                size=_synapse_size(registry, conn_key, conn),
                ptype=_synapse_type(conn),
                source=str(conn.source),
                target=str(conn.target),
            )
            conn_player[conn_key] = len(specs)
            conn_sign[conn_key] = _connection_sign(conn)
            connection_keys[conn_key] = player_key
            specs.append(spec)

        pm = cls(specs, eps_H=eps_H, connection_keys=connection_keys)
        skipped: set[str] = set()

        # 1. Incidence.
        for conn_key, conn in registry.connection_cfgs.items():
            si = conn_player.get(conn_key)
            if si is None:
                continue
            for endpoint, target_side in ((conn.source, False), (conn.target, True)):
                pi = pop_player.get(endpoint)
                if pi is None:
                    skipped.add(str(endpoint))
                    continue
                pm._add_edge(pi, si)

        # 2. Antagonism: opposite-sign incoming projections to one target.
        incoming: dict[str, list[tuple[str, str]]] = {}
        for conn_key, conn in registry.connection_cfgs.items():
            if conn_key not in conn_player or conn_sign.get(conn_key, 0) == 0:
                continue
            incoming.setdefault(str(conn.target), []).append(
                (conn_key, str(conn.source)))
        for target, items in incoming.items():
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    ka, sa = items[a]
                    kb, sb = items[b]
                    if conn_sign[ka] * conn_sign[kb] >= 0:
                        continue
                    ia, ib = conn_player.get(ka), conn_player.get(kb)
                    if ia is not None and ib is not None:
                        pm._add_edge(ia, ib)
                    pa, pb = pop_player.get(sa), pop_player.get(sb)
                    if pa is not None and pb is not None:
                        pm._add_edge(pa, pb)

        # 3. Reciprocity between synapse players of opposite projections.
        by_pair: dict[tuple[str, str], list[str]] = {}
        for conn_key, conn in registry.connection_cfgs.items():
            if conn_key in conn_player:
                by_pair.setdefault(
                    (str(conn.source), str(conn.target)), []).append(conn_key)
        for (src, tgt), forward in by_pair.items():
            backward = by_pair.get((tgt, src))
            if not backward:
                continue
            for ka in forward:
                for kb in backward:
                    ia, ib = conn_player.get(ka), conn_player.get(kb)
                    if ia is not None and ib is not None:
                        pm._add_edge(ia, ib)

        pm._skipped_endpoints = sorted(skipped)
        return pm

    @classmethod
    def from_specs(cls, player_specs, edges=(), *,
                   eps_H: float = 1e-3) -> "PhysicalMap":
        """Registry-free constructor for unit tests and synthetic games."""
        return cls(list(player_specs), edges, eps_H=eps_H)

    # --- graph ---------------------------------------------------------------

    def _add_edge(self, i: int, j: int):
        if i == j:
            return
        self.adjacency[i, j] = 1.0
        self.adjacency[j, i] = 1.0
        self._H = None

    @property
    def n_players(self) -> int:
        return len(self.specs)

    @property
    def skipped_endpoints(self) -> list[str]:
        return list(self._skipped_endpoints)

    @property
    def degree(self) -> np.ndarray:
        return self.adjacency.sum(axis=1)

    @property
    def laplacian(self) -> np.ndarray:
        return np.diag(self.degree) - self.adjacency

    @property
    def H(self) -> np.ndarray:
        if self._H is None:
            n = len(self.specs)
            if n == 0:
                self._H = np.zeros((0, 0), dtype=np.float64)
            else:
                dmax = max(float(self.degree.max()), 1.0)
                L_sym = self.laplacian / dmax
                reg = self.eps_H * (np.eye(n) - np.ones((n, n)) / n)
                H = L_sym + reg
                self._H = 0.5 * (H + H.T)
        return self._H

    def ensure_names(self, site_names) -> None:
        """Add a singleton unknown player for each unmapped site.

        Existing players and their edges are preserved; H is rebuilt lazily.
        """
        added = False
        for site in site_names:
            if site in self.player_of:
                continue
            spec = PlayerSpec(
                name=f"unknown:{site}",
                kind="unknown",
                param_names=(str(site),),
                size=1,
                ptype="unknown",
            )
            warnings.warn(
                f"PERSONA homeostasis: site {site!r} has no physical player; "
                "using an isolated singleton player", RuntimeWarning,
                stacklevel=2)
            self.player_of[site] = len(self.specs)
            self.specs.append(spec)
            added = True
        if added:
            n_old = self.adjacency.shape[0]
            n_new = len(self.specs)
            A = np.zeros((n_new, n_new), dtype=np.float64)
            A[:n_old, :n_old] = self.adjacency
            self.adjacency = A
            self._H = None

    # --- credit / variance helpers ------------------------------------------

    def player_credit(self, g_mu, site_names, g_ref) -> np.ndarray:
        """Standardize and average site credit into one scalar per player."""
        g = np.asarray(g_mu, dtype=np.float64).ravel()
        names = list(site_names)
        if g.size != len(names):
            raise ValueError("g_mu and site_names must have the same length")
        sums = np.zeros(self.n_players, dtype=np.float64)
        counts = np.zeros(self.n_players, dtype=np.float64)
        for value, site in zip(g, names):
            p = self.player_of.get(site)
            if p is None or not np.isfinite(value):
                continue
            sums[p] += float(value) / max(float(g_ref[site]), 1e-12)
            counts[p] += 1.0
        out = np.zeros(self.n_players, dtype=np.float64)
        nz = counts > 0
        out[nz] = sums[nz] / counts[nz]
        return out

    def player_variance(self, sigmas, site_names, sigma_ref) -> np.ndarray:
        """Mean squared sigma ratio per player (dimensionless ``v_i``)."""
        sig = np.asarray(sigmas, dtype=np.float64).ravel()
        names = list(site_names)
        if sig.size != len(names):
            raise ValueError("sigmas and site_names must have the same length")
        sums = np.zeros(self.n_players, dtype=np.float64)
        counts = np.zeros(self.n_players, dtype=np.float64)
        for value, site in zip(sig, names):
            p = self.player_of.get(site)
            if p is None or not np.isfinite(value):
                continue
            ref = max(float(sigma_ref[site]), 1e-12)
            sums[p] += (max(float(value), 1e-12) / ref) ** 2
            counts[p] += 1.0
        out = np.ones(self.n_players, dtype=np.float64)
        nz = counts > 0
        out[nz] = sums[nz] / counts[nz]
        return out


def _population_type(name: str, model: str) -> str:
    if str(model).lower() == "poisson_generator":
        return "poisson"
    lname = str(name).lower()
    if any(lname.startswith(prefix) for prefix in _INHIBITORY_PREFIXES):
        return "inh"
    return "exc"


def _is_stdp(conn: Connection) -> bool:
    if conn.weight_recorder is not None:
        return True
    if "stdp" in str(conn.model).lower():
        return True
    return bool(set(conn.parameters) & _STDP_PARAMETER_NAMES)


def _connection_sign(conn: Connection) -> int:
    if _is_stdp(conn):
        raw = conn.extra_syn_spec.get("weight", None)
    else:
        weight = conn.parameters.get("weight")
        raw = weight.mean if weight is not None else conn.extra_syn_spec.get("weight", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _synapse_type(conn: Connection) -> str:
    if _is_stdp(conn):
        return "stdp"
    sign = _connection_sign(conn)
    return "static_inh" if sign < 0 else "static_exc"


def _endpoint_size(registry: Registry, name: str) -> int:
    pop = registry.population_cfgs.get(name)
    if pop is not None:
        return int(pop.n)
    nodes = getattr(registry, "population_nodes", {}).get(name)
    try:
        return int(len(nodes))
    except (TypeError, AttributeError):
        return 0


def _synapse_size(registry: Registry, conn_key: str, conn: Connection) -> int:
    counts = getattr(registry, "_syn_counts", {})
    if conn_key in counts:
        return int(counts[conn_key])
    n_src = _endpoint_size(registry, conn.source)
    n_tgt = _endpoint_size(registry, conn.target)
    rule = str(conn.rule)
    if rule == "pairwise_bernoulli":
        n = n_src * n_tgt * float(np.clip(conn.get_effective_p(), 0.0, 1.0))
    elif rule == "all_to_all":
        n = n_src * n_tgt
    elif rule == "one_to_one":
        n = min(n_src, n_tgt)
    else:
        n = 0
    return int(round(n))


_MODES = ("advantage", "budget", "fisher", "band")
_LIFT_MODES = ("player", "site_centered")


@dataclass
class HomeostaticConfig:
    """Configuration for the opt-in homeostatic regulator."""
    enabled: bool = False
    lambda_H: float = 1.0
    tau_H_s: float = 5.0
    tau_m_s: float = 2.0
    tau_S_s: float = 2.0
    tau_A_s: float = 2.0
    tau_V_s: float = 5.0
    h_burnin_steps: int = 100
    global_variance_mode: str = "advantage"
    lift_mode: str = "player"
    h_delay_steps: int = 0
    T_min: float = 0.1
    T_max: float = 10.0
    band_lo: float = 0.5
    band_hi: float = 2.0
    v_min: float = 1e-8
    v_max: float = 1e6
    S_min: float = 1e-12
    g_min: float = 1e-12
    eps_H: float = 1e-3
    eps_credit: float = 1e-9

    @classmethod
    def current_best(cls) -> "HomeostaticConfig":
        """Best-known experimental preset; not an acceptance pass.

        This is the centered site-zero-sum lift with a 500-step delay, run
        with the Fisher global-variance mode and the calibrated H gain. It
        remains opt-in and is not a production default.
        """
        return cls(
            enabled=True,
            global_variance_mode="fisher",
            lift_mode="site_centered",
            h_delay_steps=500,
        )

    def __post_init__(self):
        mode = str(self.global_variance_mode)
        if mode not in _MODES:
            raise ValueError(
                f"global_variance_mode must be one of {_MODES}, got {mode!r}")
        self.global_variance_mode = mode
        lift = str(self.lift_mode)
        if lift not in _LIFT_MODES:
            raise ValueError(
                f"lift_mode must be one of {_LIFT_MODES}, got {lift!r}")
        self.lift_mode = lift
        for name in ("lambda_H", "tau_H_s", "tau_m_s", "tau_S_s",
                     "tau_A_s", "tau_V_s", "T_min", "T_max", "band_lo",
                     "band_hi", "v_min", "v_max", "S_min", "g_min",
                     "eps_H", "eps_credit"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.lambda_H < 0.0:
            raise ValueError("lambda_H must be non-negative")
        for name in ("tau_H_s", "tau_m_s", "tau_S_s", "tau_A_s", "tau_V_s",
                     "T_min", "v_min", "v_max", "S_min", "g_min"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.T_max < self.T_min:
            raise ValueError("T_max must be >= T_min")
        if self.band_hi < self.band_lo:
            raise ValueError("band_hi must be >= band_lo")
        if self.v_max < self.v_min:
            raise ValueError("v_max must be >= v_min")
        if int(self.h_burnin_steps) < 0:
            raise ValueError("h_burnin_steps must be non-negative")
        if int(self.h_delay_steps) < 0:
            raise ValueError("h_delay_steps must be non-negative")
        if self.eps_H < 0.0 or self.eps_credit < 0.0:
            raise ValueError("eps_H and eps_credit must be non-negative")
        self.h_burnin_steps = int(self.h_burnin_steps)
        self.h_delay_steps = int(self.h_delay_steps)


def gaussian_log_score(z, mu, variance):
    """Proper Gaussian log score for standardized credit."""
    z = np.asarray(z, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    v = np.maximum(np.asarray(variance, dtype=np.float64), 1e-300)
    return -0.5 * (np.log(2.0 * np.pi * v) + (z - mu) ** 2 / v)


def _softmax(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / max(float(ex.sum()), 1e-300)


class GlobalVarianceRegulator:
    """The dimensionless global-variance unit of account.

    It is one small state machine for the four modes. It is not a player and
    no player can set it directly; its scale is estimated from the task's
    advantage, the Fisher trace, or a slow target band.
    """

    def __init__(self, config: HomeostaticConfig, dt_s: float = 0.01):
        self.config = config
        self.dt_s = float(dt_s)
        self.mode = config.global_variance_mode
        self.beta_A = float(np.exp(-self.dt_s / config.tau_A_s))
        self.beta_V = float(np.exp(-self.dt_s / config.tau_V_s))
        self.V_A0 = 1.0
        self.V_F0 = 1.0
        self.mean_A = 0.0
        self.var_A = 1.0
        self.V_obs = 1.0
        self.V_F_ewma = 1.0
        self.R_ewma = 1.0
        self.T = 1.0

    def initialize(self, V_A0: float, V_F0: float, mean_A0: float):
        self.V_A0 = _positive_or_one(V_A0)
        self.V_F0 = _positive_or_one(V_F0)
        self.mean_A = float(mean_A0) if np.isfinite(mean_A0) else 0.0
        self.var_A = self.V_A0
        self.V_obs = self.V_A0
        self.V_F_ewma = self.V_F0
        self.R_ewma = 1.0
        self.T = 1.0

    def update(self, A: float, V_F: float) -> float:
        A = float(A)
        if not np.isfinite(A):
            A = 0.0
        VF = float(V_F)
        if not np.isfinite(VF):
            VF = 0.0

        delta = A - self.mean_A
        self.mean_A = self.mean_A + (1.0 - self.beta_A) * delta
        self.var_A = self.beta_A * (
            self.var_A + (1.0 - self.beta_A) * delta * delta)
        self.V_obs = max(float(self.var_A), 0.0)
        self.V_F_ewma = (self.beta_A * self.V_F_ewma
                         + (1.0 - self.beta_A) * VF)

        if self.mode == "budget":
            self.T = 1.0
        elif self.mode == "fisher":
            ratio = self.V_F_ewma / max(self.V_F0, 1e-300)
            self.T = float(np.clip(ratio, self.config.T_min,
                                   self.config.T_max))
        elif self.mode == "band":
            ratio = self.V_obs / max(self.V_A0, 1e-300)
            self.R_ewma = (self.beta_V * self.R_ewma
                           + (1.0 - self.beta_V) * ratio)
            self.T = float(np.clip(self.R_ewma, self.config.band_lo,
                                   self.config.band_hi))
        else:  # advantage
            ratio = self.V_obs / max(self.V_A0, 1e-300)
            self.T = float(np.clip(ratio, self.config.T_min,
                                   self.config.T_max))
        return self.T


def _positive_or_one(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 1.0
    return out if np.isfinite(out) and out > 0.0 else 1.0


class HomeostaticRegulator:
    """Homeostatic credit redistribution on the physical player graph.

    ``adjust_credit`` is the strategy-facing API: it standardizes site credit
    and lifts the player correction back to sites. ``step_players`` is the
    registry-free core used by the synthetic benchmark.
    """

    def __init__(self, physical_map: PhysicalMap,
                 config: HomeostaticConfig | None = None, *,
                 dt_s: float = 0.01, beta: float = 0.995,
                 site_names=None):
        self.physical_map = physical_map
        self.config = config if config is not None else HomeostaticConfig(enabled=True)
        self.dt_s = float(dt_s)
        self.beta = float(beta)
        if site_names is None:
            site_names = [site for spec in physical_map.specs
                          for site in spec.param_names]
        self._site_names = list(site_names)
        self.physical_map.ensure_names(self._site_names)

        self.g_ref: dict[str, float] = {}
        self.sigma_ref: dict[str, float] = {}
        self._player_sigma_ref: list[float | None] = [
            None] * physical_map.n_players
        c = self.config
        n = physical_map.n_players
        self.S_c = max(c.S_min, 1e-12)
        self.mu = np.zeros(n, dtype=np.float64)
        self.e_bar = np.zeros(n, dtype=np.float64)
        self.stage = "burnin"
        self._initialized = False
        self._burnin_g: list[np.ndarray] = []
        self._burnin_c: list[np.ndarray] = []
        self._burnin_A: list[float] = []
        self._burnin_VF: list[float] = []
        self.global_variance = GlobalVarianceRegulator(c, dt_s=self.dt_s)
        self.last: dict = {}
        self._set_last("burnin", steps=0)

    # --- lifecycle -----------------------------------------------------------

    def ensure_params(self, site_names) -> None:
        """Preserve known sites/players and append new sites as singletons."""
        known = set(self._site_names)
        changed = False
        for site in site_names:
            if site not in known:
                self._site_names.append(str(site))
                known.add(site)
                changed = True
        if self.physical_map.n_players != len(self._player_sigma_ref):
            self._player_sigma_ref.extend(
                [None] * (self.physical_map.n_players
                          - len(self._player_sigma_ref)))
        if changed:
            self.physical_map.ensure_names(self._site_names)
            self._resize_player_state()
            if self.physical_map.n_players != len(self._player_sigma_ref):
                self._player_sigma_ref.extend(
                    [None] * (self.physical_map.n_players
                              - len(self._player_sigma_ref)))

    def _resize_player_state(self):
        n = self.physical_map.n_players
        if self.mu.size != n:
            mu = np.zeros(n, dtype=np.float64)
            mu[:min(n, self.mu.size)] = self.mu[:min(n, self.mu.size)]
            self.mu = mu
        if self.e_bar.size != n:
            eb = np.zeros(n, dtype=np.float64)
            eb[:min(n, self.e_bar.size)] = self.e_bar[:min(n, self.e_bar.size)]
            self.e_bar = eb
        if len(self._player_sigma_ref) < n:
            self._player_sigma_ref.extend([None] * (n - len(self._player_sigma_ref)))

    @property
    def site_names(self) -> list[str]:
        return list(self._site_names)

    def diagnostics(self) -> dict:
        out = {}
        for key, value in self.last.items():
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            elif isinstance(value, list):
                out[key] = list(value)
            else:
                out[key] = value
        return out

    # --- strategy-facing API -------------------------------------------------

    def adjust_credit(self, g_mu, sigmas, advantage_scalar, F, steps: int, *,
                      dt_s: float | None = None) -> np.ndarray:
        if dt_s is not None:
            self._set_dt(dt_s)
        g = np.asarray(g_mu, dtype=np.float64).ravel()
        sig = np.asarray(sigmas, dtype=np.float64).ravel()
        if g.size != len(self._site_names) or sig.size != len(self._site_names):
            raise ValueError(
                "g_mu/sigmas length does not match the regulator site order; "
                "call ensure_params first")
        if not np.all(np.isfinite(g)):
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(sig)):
            sig = np.nan_to_num(sig, nan=1e-12, posinf=1e-12, neginf=1e-12)
        sig = np.maximum(sig, 1e-12)
        V_F = self._fisher_trace(F, len(sig))

        for site, s in zip(self._site_names, sig):
            self.sigma_ref.setdefault(str(site), max(float(s), 1e-12))

        if self.stage == "burnin":
            if self.config.h_burnin_steps == 0:
                self._init_single_site(g, sig, advantage_scalar, V_F)
                self._set_last(
                    "active", steps=steps, c=self._site_credit(g),
                    z=self.last.get("z", np.zeros(self.physical_map.n_players)),
                    u_H=np.zeros(self.physical_map.n_players),
                    gate=True, V_F=V_F)
                return g.copy()
            self._burnin_g.append(g.copy())
            self._burnin_A.append(float(advantage_scalar)
                                  if np.isfinite(advantage_scalar) else 0.0)
            self._burnin_VF.append(V_F)
            if len(self._burnin_g) >= self.config.h_burnin_steps:
                self._finalize_site_burnin()
                self.stage = "active"
            self._set_last("burnin", steps=steps,
                           u_H=np.zeros(self.physical_map.n_players),
                           gate=True, V_F=V_F,
                           burnin_step=len(self._burnin_g))
            return g.copy()

        # Active. New sites get their unit on first sight.
        for site, value in zip(self._site_names, g):
            self.g_ref.setdefault(
                str(site), max(abs(float(value)), self.config.g_min))
        c = self._site_credit(g)
        v = self._site_variance(sig)
        u, T = self._active_core(c, v, advantage_scalar, V_F, steps)
        g_h = self._lift_correction(u)
        self.last["g_H"] = g_h.copy()
        return g + g_h

    def step_players(self, c, sigmas, advantage_scalar, F, steps: int, *,
                     dt_s: float | None = None) -> np.ndarray:
        if dt_s is not None:
            self._set_dt(dt_s)
        c = np.asarray(c, dtype=np.float64).ravel()
        sig = np.asarray(sigmas, dtype=np.float64).ravel()
        n = self.physical_map.n_players
        if c.size != n:
            raise ValueError("c length must equal the number of players")
        if not np.all(np.isfinite(c)):
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        V_F = self._fisher_trace(F, max(1, len(self._site_names)))

        if sig.size == n:
            for i, value in enumerate(sig):
                if self._player_sigma_ref[i] is None:
                    self._player_sigma_ref[i] = max(
                        float(value) if np.isfinite(value) else 1e-12, 1e-12)
            v = np.array([
                (max(float(value), 1e-12)
                 / max(float(self._player_sigma_ref[i]), 1e-12)) ** 2
                for i, value in enumerate(sig)], dtype=np.float64)
        elif sig.size == len(self._site_names):
            for site, value in zip(self._site_names, sig):
                self.sigma_ref.setdefault(
                    str(site), max(float(value), 1e-12))
            v = self._site_variance(sig)
        else:
            raise ValueError("sigmas must be per-player or per-site")
        v = np.clip(v, self.config.v_min, self.config.v_max)

        if self.stage == "burnin":
            if self.config.h_burnin_steps == 0:
                self._init_single_player(c, v, advantage_scalar, V_F)
                self._set_last("active", steps=steps, c=c,
                               z=self.last.get(
                                   "z", np.zeros(n)),
                               u_H=np.zeros(n), gate=True, V_F=V_F)
                return np.zeros(n, dtype=np.float64)
            self._burnin_c.append(c.copy())
            self._burnin_A.append(float(advantage_scalar)
                                  if np.isfinite(advantage_scalar) else 0.0)
            self._burnin_VF.append(V_F)
            if len(self._burnin_c) >= self.config.h_burnin_steps:
                self._finalize_player_burnin()
                self.stage = "active"
            self._set_last("burnin", steps=steps, u_H=np.zeros(n),
                           gate=True, V_F=V_F,
                           burnin_step=len(self._burnin_c))
            return np.zeros(n, dtype=np.float64)

        u, _ = self._active_core(c, v, advantage_scalar, V_F, steps)
        return u

    # --- state initialization ------------------------------------------------

    def _init_single_site(self, g, sig, advantage, V_F):
        self.g_ref = {
            str(site): max(abs(float(value)), self.config.g_min)
            for site, value in zip(self._site_names, g)}
        c = self._site_credit(g)
        self._set_initial_state(c, advantage, V_F, n_samples=1)
        self.stage = "active"
        self._initialized = True

    def _init_single_player(self, c, v, advantage, V_F):
        self._set_initial_state(c, advantage, V_F, n_samples=1)
        self.stage = "active"
        self._initialized = True

    def _set_initial_state(self, c, advantage, V_F, n_samples):
        n = self.physical_map.n_players
        c = np.asarray(c, dtype=np.float64).ravel()
        if c.size != n:
            c = np.resize(c, n) if n else np.zeros(0)
        S_c0 = max(float(np.std(c)) if c.size else 0.0, self.config.S_min)
        z = ((c - np.mean(c)) / S_c0) if c.size else np.zeros(0)
        self.S_c = S_c0
        self.mu = z.copy()
        self.e_bar = np.zeros(n, dtype=np.float64)
        self.global_variance.initialize(
            V_A0=1.0 if n_samples <= 1 else _positive_or_one(
                float(np.var([advantage], ddof=1)) if n_samples > 1 else 0.0),
            V_F0=_positive_or_one(
                V_F / max(1.0 - self.beta ** n_samples, 0.1)),
            mean_A0=float(advantage) if np.isfinite(advantage) else 0.0)

    def _finalize_site_burnin(self):
        T = len(self._burnin_g)
        n_sites = len(self._site_names)
        garr = np.full((T, n_sites), np.nan, dtype=np.float64)
        for t, row in enumerate(self._burnin_g):
            garr[t, :min(row.size, n_sites)] = row[:n_sites]
        for j, site in enumerate(self._site_names):
            vals = garr[:, j]
            vals = vals[np.isfinite(vals)]
            rms = float(np.sqrt(np.mean(vals ** 2))) if vals.size else 0.0
            self.g_ref[str(site)] = max(rms, self.config.g_min)
        c_rows = np.array([
            self._site_credit(row) for row in garr], dtype=np.float64)
        self._init_from_batch(c_rows, self._burnin_A, self._burnin_VF)
        self._burnin_g = []

    def _finalize_player_burnin(self):
        n = self.physical_map.n_players
        T = len(self._burnin_c)
        c_rows = np.full((T, n), np.nan, dtype=np.float64)
        for t, row in enumerate(self._burnin_c):
            c_rows[t, :min(row.size, n)] = row[:n]
        self._init_from_batch(c_rows, self._burnin_A, self._burnin_VF)
        self._burnin_c = []

    def _init_from_batch(self, c_rows, a_values, vf_values):
        c_rows = np.asarray(c_rows, dtype=np.float64)
        if c_rows.ndim == 1:
            c_rows = c_rows[:, None]
        n = self.physical_map.n_players
        if c_rows.shape[1] != n:
            fixed = np.zeros((c_rows.shape[0], n), dtype=np.float64)
            cols = min(n, c_rows.shape[1])
            fixed[:, :cols] = c_rows[:, :cols]
            c_rows = fixed
        c_rows = np.nan_to_num(c_rows, nan=0.0, posinf=0.0, neginf=0.0)
        S_c0 = max(float(np.mean(np.std(c_rows, axis=1)))
                   if c_rows.size else 0.0, self.config.S_min)
        z_rows = (c_rows - np.mean(c_rows, axis=1, keepdims=True)) / S_c0
        mu0 = np.mean(z_rows, axis=0) if z_rows.size else np.zeros(n)

        a = np.asarray(a_values, dtype=np.float64)
        a = a[np.isfinite(a)]
        V_A0 = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
        V_A0 = _positive_or_one(V_A0)
        mean_A0 = float(np.mean(a)) if a.size else 0.0
        vf_end = float(vf_values[-1]) if vf_values else 0.0
        denom = max(1.0 - self.beta ** max(len(a_values), 1), 0.1)
        V_F0 = _positive_or_one(vf_end / denom)

        self.S_c = S_c0
        self.mu = mu0.astype(np.float64, copy=True)
        self.e_bar = np.zeros(n, dtype=np.float64)
        self.global_variance.initialize(V_A0, V_F0, mean_A0)
        self._initialized = True

    # --- active core ---------------------------------------------------------

    def _active_core(self, c, v, advantage_scalar, V_F, steps):
        n = self.physical_map.n_players
        c = np.asarray(c, dtype=np.float64).ravel()
        if c.size != n:
            raise ValueError("player credit length changed after initialization")
        self._resize_player_state()
        S_c_prev = max(float(self.S_c), self.config.S_min)
        cbar = float(np.mean(c)) if n else 0.0
        z = ((c - cbar) / S_c_prev) if n else np.zeros(0)
        S_c_raw = float(np.std(c)) if n else 0.0
        T = self.global_variance.update(advantage_scalar, V_F)
        T = max(float(T), 1e-12)
        v = np.clip(np.asarray(v, dtype=np.float64).ravel(),
                    self.config.v_min, self.config.v_max)
        if v.size != n:
            raise ValueError("player variance length changed")
        ell = gaussian_log_score(z, self.mu, v)
        a = _softmax(ell / T)
        pi = _softmax(z / T)
        e = a - pi

        alpha_H = 1.0 - np.exp(-self.dt_s / self.config.tau_H_s)
        e_bar_new = (1.0 - alpha_H) * self.e_bar + alpha_H * e
        gate = bool((not np.isfinite(S_c_raw))
                    or S_c_raw < self.config.eps_credit)
        delayed = bool(steps < self.config.h_delay_steps)
        if gate or delayed or self.config.lambda_H == 0.0 or n == 0:
            u = np.zeros(n, dtype=np.float64)
        else:
            u = (self.config.lambda_H * S_c_prev
                 * (self.physical_map.H @ e_bar_new))
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        projected = ((c + u - cbar) / S_c_prev) if n else np.zeros(0)
        projected_pi = _softmax(projected / T)

        alpha_S = 1.0 - np.exp(-self.dt_s / self.config.tau_S_s)
        alpha_m = 1.0 - np.exp(-self.dt_s / self.config.tau_m_s)
        S_c_next = ((1.0 - alpha_S) * S_c_prev
                    + alpha_S * max(S_c_raw, self.config.S_min))
        mu_prev = self.mu.copy()
        mu_next = (1.0 - alpha_m) * mu_prev + alpha_m * z
        if n and not np.all(np.isfinite(mu_next)):
            mu_next = np.nan_to_num(mu_next, nan=0.0, posinf=0.0,
                                    neginf=0.0)

        self.e_bar = e_bar_new
        self.S_c = float(S_c_next)
        self.mu = mu_next
        self._set_last(
            "active", steps=steps, T=T, V_F=V_F, S_c=self.S_c,
            S_c_raw=S_c_raw, S_c_prev=S_c_prev, c=c, z=z,
            mu=self.mu, mu_prev=mu_prev, v=v, ell=ell, a=a, pi=pi, e=e,
            e_bar=self.e_bar, projected_pi=projected_pi, u_H=u,
            gate=gate, h_delayed=delayed,
            correction_norm=float(np.linalg.norm(u)), burnin_step=0)
        return u, T

    # --- diagnostics / helpers ----------------------------------------------

    def _set_dt(self, dt_s):
        self.dt_s = float(dt_s)
        self.global_variance.dt_s = self.dt_s
        self.global_variance.beta_A = float(
            np.exp(-self.dt_s / self.config.tau_A_s))
        self.global_variance.beta_V = float(
            np.exp(-self.dt_s / self.config.tau_V_s))

    def _fisher_trace(self, F, n_sites):
        if F is None or n_sites <= 0:
            return 0.0
        arr = np.asarray(F, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return 0.0
        k = min(n_sites, arr.shape[0], arr.shape[1])
        if k <= 0:
            return 0.0
        return float(np.trace(arr[:k, :k]) / float(k))

    def _site_credit(self, g):
        if not self.g_ref:
            if self.stage == "burnin":
                return np.zeros(self.physical_map.n_players,
                                dtype=np.float64)
        return self.physical_map.player_credit(
            g, self._site_names, self.g_ref)

    def _site_variance(self, sig):
        for site, value in zip(self._site_names, sig):
            self.sigma_ref.setdefault(str(site), max(float(value), 1e-12))
        v = self.physical_map.player_variance(
            sig, self._site_names, self.sigma_ref)
        return np.clip(v, self.config.v_min, self.config.v_max)

    def _lift_correction(self, u):
        """Lift player corrections back to sites.

        ``player`` is the original adjoint lift: the player's credit changes
        by exactly ``u_i``. ``site_centered`` subtracts the player's mean
        ``g_ref`` before lifting, so every player's site correction sums to
        zero. It is the experimental current-best lift, not the adjoint.
        """
        g_h = np.zeros(len(self._site_names), dtype=np.float64)
        by_player: list[list[int]] = [[] for _ in range(self.physical_map.n_players)]
        for i, site in enumerate(self._site_names):
            p = self.physical_map.player_of.get(site)
            if p is not None:
                by_player[p].append(i)

        if self.config.lift_mode == "site_centered":
            for p, indices in enumerate(by_player):
                if len(indices) < 2:
                    continue
                refs = np.asarray([
                    self.g_ref.get(self._site_names[i], self.config.g_min)
                    for i in indices], dtype=np.float64)
                centered = refs - float(np.mean(refs))
                for i, ref in zip(indices, centered):
                    g_h[i] = u[p] * float(ref)
            return g_h

        for p, indices in enumerate(by_player):
            for i in indices:
                site = self._site_names[i]
                g_h[i] = u[p] * self.g_ref.get(site, self.config.g_min)
        return g_h

    def _set_last(self, stage, *, steps, **kwargs):
        n = self.physical_map.n_players
        out = {
            "stage": stage,
            "mode": self.config.global_variance_mode,
            "lift_mode": self.config.lift_mode,
            "h_delay_steps": int(self.config.h_delay_steps),
            "h_delayed": False,
            "T": 1.0,
            "V_obs": float(self.global_variance.V_obs),
            "V_A0": float(self.global_variance.V_A0),
            "V_F": 0.0,
            "V_F0": float(self.global_variance.V_F0),
            "S_c": float(self.S_c),
            "S_c_prev": float(self.S_c),
            "S_c_raw": 0.0,
            "c": np.zeros(n, dtype=np.float64),
            "z": np.zeros(n, dtype=np.float64),
            "mu": self.mu.copy(),
            "mu_prev": self.mu.copy(),
            "v": np.ones(n, dtype=np.float64),
            "ell": np.zeros(n, dtype=np.float64),
            "a": np.full(n, 1.0 / n) if n else np.zeros(0),
            "pi": np.full(n, 1.0 / n) if n else np.zeros(0),
            "e": np.zeros(n, dtype=np.float64),
            "e_bar": self.e_bar.copy(),
            "projected_pi": np.full(n, 1.0 / n) if n else np.zeros(0),
            "u_H": np.zeros(n, dtype=np.float64),
            "lambda_H": float(self.config.lambda_H),
            "tau_H": float(self.config.tau_H_s),
            "correction_norm": 0.0,
            "gate": True,
            "skipped_endpoints": list(self.physical_map.skipped_endpoints),
            "player_names": [spec.name for spec in self.physical_map.specs],
            "step": int(steps),
            "burnin_step": 0,
        }
        out.update(kwargs)
        self.last = out

