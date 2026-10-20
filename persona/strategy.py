from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class PhaseBaselines:
    """
    Per-output-component, per-phase moving-average baselines. Each output
    component keeps its own baseline for "stance" and for "swing", so the
    advantage of the left ankle during stance is measured against the left
    ankle's previous stance, independently of the right leg's phase.
    """
    alpha: float = 0.95
    _baselines: Dict[tuple, float] = field(default_factory=dict)

    def get_baseline(self, component: int, phase: str) -> float:
        return float(self._baselines.get((int(component), str(phase)), 0.0))

    def update(self, component: int, phase: str, loss: float):
        key = (int(component), str(phase))
        b = self._baselines.get(key)
        if b is None:
            self._baselines[key] = float(loss)
        else:
            self._baselines[key] = self.alpha * b + (1.0 - self.alpha) * float(loss)


class FisherMatrix:
    """
    Persisted per-step strategy (spec §4): one forgetting factor
    β = 1 − dt/τ_mem over the whole run, no windows. State:

    - ``F    = βF    + (1−β) s sᵀ`` — expected Fisher of the policy scores;
    - ``Xbar = βXbar + (1−β) a``    — EWMA trace (what actually moved, per channel);
    - ``G    = βG    + (1−β) Xbar Aᵀ`` — delayed credit: the current trace paired
      with the advantage A(t), so a site whose *trace* (its recent applied moves)
      predicts the advantage is credited even when the coupling lags.

    Channel layout (``n = 2·|S| + |P|``): site μ rows, site σ rows, then noise
    producer rows. Noise producers get Fisher rows (they shape the step length
    through F) but **no trace rows** — ``a`` is zero for them. Their score is
    ``s^noise = δ/R²`` with the producer's current value ``R`` at registration
    (no estimation).

    For each site ``s`` (sampling-time snapshot):

        s^μ_s = Δθ_s / σ_s²            score of the μ coordinate
        s^σ_s = z_s² − 1, z = Δθ_s/σ_s       log-σ score
        a^μ_s = Δθ_s / σ_s²            trace row, μ part — score units, so the
                                       natural-gradient direction is unit
                                       invariant (a raw-Δθ trace would give
                                       every site a spurious σ_s² gain)
        a^σ_s = Δlogσ_s(t)             σ change actually applied this step

    Update step: ``g = G·1₂`` (noise rows excluded), ``Δμ = lr_mu · P g``,
    where ``P`` starts at ``(1/λ)I`` and is updated by Sherman–Morrison
    between refreshes: ``P_t = (F_t + λβᵏI)⁻¹``. Every ``refresh_every``
    steps, and at a phase shift, ``P`` is recomputed exactly as
    ``(F + λI)⁻¹`` by Cholesky, restoring the constant ridge. The σ
    coordinates are updated by the Agent's log-σ rule; ``lr_sigma`` is kept
    here only for interface compatibility.

    Warm-up: F/Xbar/G accumulate from step 1, Δμ is applied from step
    ``warmup``+1. Advantage baselines are per-component, per-phase
    (α = 0.90, unchanged).

    TODO(perf): the recursive inverse removes the O(n³) solve; F is still
    O(n²) per step and O(n²) memory.
    """
    def __init__(self,
                 param_names: List[str],
                 noise_names: List[str] | None = None,
                 lr_mu: float = 0.02,
                 lr_sigma: float = 0.005,
                 lambda_reg: float = 1e-3,
                 beta: float = 0.995,
                 warmup: int = 3,
                 baseline_alpha: float = 0.90,
                 refresh_every: int | None = None):
        self.param_names = list(param_names)
        self.noise_names = list(noise_names or [])
        self.noise_scales: Dict[str, float] = {}
        self.lr_mu = float(lr_mu)
        self.lr_sigma = float(lr_sigma)
        self.lambda_reg = float(lambda_reg)
        self.beta = float(beta)
        self.warmup = int(warmup)
        self.steps = 0

        self.baselines = PhaseBaselines(alpha=baseline_alpha)

        n = self.n_total
        self.F = np.zeros((n, n), dtype=np.float64)
        self.Xbar = np.zeros(n, dtype=np.float64)
        self.G = np.zeros((n, 2), dtype=np.float64)

        if not np.isfinite(self.lambda_reg) or self.lambda_reg <= 0.0:
            raise ValueError(
                "lambda_reg must be finite and positive for the recursive inverse")
        if not np.isfinite(self.beta) or self.beta < 0.0 or self.beta > 1.0:
            raise ValueError("beta must be finite and in [0, 1]")
        self.refresh_every = self._resolve_refresh_every(refresh_every, self.beta)
        self._steps_since_refresh = 0
        self._P_valid = True
        self.P = np.eye(n, dtype=np.float64) / self.lambda_reg

    @property
    def P(self):
        """Inverse of ``F + lambda*beta**k I``; rebuilt lazily when stale."""
        if not self._P_valid:
            self._refresh_inverse()
        return self._P

    @P.setter
    def P(self, value):
        self._P = value
        self._P_valid = True

    @staticmethod
    def _resolve_refresh_every(value, beta):
        # beta == 0 has no finite recursive ridge: use the exact solve every step.
        if beta <= 0.0:
            return 1
        if value is not None:
            return max(1, int(value))
        if beta >= 1.0:
            return 10**9
        return max(1, int(np.floor(np.log(0.95) / np.log(beta))))

    def _solve_direct(self, g):
        M = self.F + self.lambda_reg * np.eye(self.n_total)
        try:
            return np.linalg.solve(M, g)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M) @ g

    def _rank1_update_inverse(self, s):
        if self.beta <= 0.0:
            self._refresh_inverse()
            return
        u = self.P @ s
        v = float(s @ u)
        denom = self.beta + (1.0 - self.beta) * v
        if denom <= 0.0 or not np.isfinite(denom):
            self._refresh_inverse()
            return
        self.P = (self.P - ((1.0 - self.beta) / denom) * np.outer(u, u)) / self.beta
        self.P = 0.5 * (self.P + self.P.T)

    def _refresh_inverse(self):
        n = self.n_total
        if n == 0:
            self.P = np.zeros((0, 0), dtype=np.float64)
            self._steps_since_refresh = 0
            return
        M = self.F + self.lambda_reg * np.eye(n)
        try:
            cf = cho_factor(M, lower=True, check_finite=False)
            P = cho_solve(cf, np.eye(n), check_finite=False)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(M)
        self.P = 0.5 * (P + P.T)
        self._steps_since_refresh = 0

    # --- layout helpers ------------------------------------------------------

    @property
    def n_sites(self) -> int:
        return len(self.param_names)

    @property
    def n_noise(self) -> int:
        return len(self.noise_names)

    @property
    def n_total(self) -> int:
        return 2 * self.n_sites + self.n_noise

    @property
    def all_names(self) -> List[str]:
        """Blocked channel labels of the F/X̄/G rows: [μ; σ; noise]."""
        return ([f"{k} μ" for k in self.param_names]
                + [f"{k} σ" for k in self.param_names]
                + [f"noise {k}" for k in self.noise_names])

    def ensure_names(self, param_names, noise_names=None) -> None:
        """
        Grows the state to cover new sites/producers, zero-padding F/Xbar/G and
        preserving the order (and history) of the existing channels. New sites
        append to the μ block; their σ rows append to the σ block; new noise
        producers append to the noise block.
        """
        new_params = [k for k in (param_names or []) if k not in self.param_names]
        new_noise = [k for k in (noise_names or []) if k not in self.noise_names]
        if not new_params and not new_noise:
            return
        m_old, k_old = self.n_sites, self.n_noise
        self.param_names += new_params
        self.noise_names += new_noise
        m_new, k_new = self.n_sites, self.n_noise

        F = np.zeros((2 * m_new + k_new, 2 * m_new + k_new), dtype=np.float64)
        X = np.zeros(2 * m_new + k_new, dtype=np.float64)
        G = np.zeros((2 * m_new + k_new, 2), dtype=np.float64)
        old_idx = ([i for i in range(m_old)]
                   + [m_old + i for i in range(m_old)]
                   + [2 * m_old + j for j in range(k_old)])
        new_idx = ([i for i in range(m_old)]                  # μ rows keep indices
                   + [m_new + i for i in range(m_old)]        # σ block shifts
                   + [2 * m_new + j for j in range(k_old)])   # noise block shifts
        F[np.ix_(new_idx, new_idx)] = self.F[np.ix_(old_idx, old_idx)]
        X[new_idx] = self.Xbar[old_idx]
        G[new_idx, :] = self.G[old_idx, :]
        self.F, self.Xbar, self.G = F, X, G
        self._refresh_inverse()

    # --- advantage / updates -------------------------------------------------

    @staticmethod
    def _per_component_phases(phase: object, n: int) -> list:
        if isinstance(phase, str):
            return [phase] * n
        seq = [str(p) for p in phase]
        if len(seq) == n:
            return seq
        return [seq[0]] * n if seq else ["stance"] * n

    def compute_advantage(self, loss_vector: np.ndarray, phase: object) -> np.ndarray:
        """
        Per-component advantage A_i = baseline_i(phase_i) − loss_i, each
        component baselined against its own previous same-phase loss.
        """
        loss = np.asarray(loss_vector, dtype=np.float64)
        phases = self._per_component_phases(phase, len(loss))
        advantage = np.empty_like(loss)
        for i in range(len(loss)):
            advantage[i] = self.baselines.get_baseline(i, phases[i]) - loss[i]
            self.baselines.update(i, phases[i], loss[i])
        return advantage

    def step(self,
             sites: Dict[str, tuple],
             noise: Dict[str, tuple],
             advantage: np.ndarray,
             *,
             phase_shift: bool = False) -> Dict[str, float]:
        """
        One strategy step.

        ``sites[key] = (Δθ, σ_sampled, Δlogσ)`` for every site (sampling-time
        values); ``noise[name] = (δ, R)`` for every producer this step, where
        `R` is the producer's current value at registration (missing producers
        contribute δ=0); ``advantage`` is the per-component advantage vector
        A(t) (length 2, one per limb).

        Updates F/Xbar/G, refreshes the recursive inverse on ``phase_shift``
        or at ``refresh_every``, counts the step, and returns the per-site Δμ
        (all zero during warm-up).
        """
        self.ensure_names(list(sites.keys()), list(noise.keys()))
        m, k = self.n_sites, self.n_noise
        n = self.n_total

        s = np.zeros(n, dtype=np.float64)
        a = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(self.param_names):
            dth, sigma, dlog = sites[name]
            sig = max(float(sigma), 1e-12)
            z2 = (float(dth) / sig) ** 2
            s[i] = float(dth) / sig ** 2
            s[m + i] = z2 - 1.0
            a[i] = float(dth) / sig ** 2
            a[m + i] = float(dlog)
        for j, name in enumerate(self.noise_names):
            delta, scale = noise.get(name, (0.0, 0.0))
            # Producer's current value at registration: s = δ/R², with no
            # estimation, no warm-up and no per-step gate. R ≤ 0 (or absent)
            # means no influence.
            r = float(scale)
            if r > 0.0:
                s[2 * m + j] = float(delta) / (r * r)
                self.noise_scales[name] = r

        A = np.asarray(advantage, dtype=np.float64).ravel()
        if A.size < 2:
            A = np.pad(A, (0, 2 - A.size))
        A = A[:2]

        b = self.beta
        self.F = b * self.F + (1.0 - b) * np.outer(s, s)
        self.Xbar = b * self.Xbar + (1.0 - b) * a
        self.G = b * self.G + (1.0 - b) * np.outer(self.Xbar, A)
        self.steps += 1

        self._steps_since_refresh += 1
        if self.refresh_every == 1:
            # Every step has the constant ridge. Solve directly instead of
            # building the full inverse, and invalidate the lazy P cache.
            self._steps_since_refresh = 0
            self._P_valid = False
        elif phase_shift or self._steps_since_refresh >= self.refresh_every:
            self._refresh_inverse()
        else:
            self._rank1_update_inverse(s)

        deltas = {name: 0.0 for name in self.param_names}
        if self.steps <= self.warmup:
            return deltas

        g = self.G.sum(axis=1)
        g[2 * m:] = 0.0                      # noise rows never receive credit
        nat = self._solve_direct(g) if self.refresh_every == 1 else self.P @ g
        if not np.all(np.isfinite(nat)):
            self._refresh_inverse()
            nat = self.P @ g
        if not np.all(np.isfinite(nat)):
            nat = self._solve_direct(g)
        for i, name in enumerate(self.param_names):
            deltas[name] = float(self.lr_mu * nat[i])
        return deltas

    # --- introspection -------------------------------------------------------

    def feature_scales(self) -> np.ndarray:
        """sqrt(diag(F)) — per-channel activity scale (kept for h5/figures)."""
        return np.sqrt(np.maximum(np.diag(self.F), 0.0))

    def influence(self) -> np.ndarray:
        """Normalised |G| row norms over the 2·|S| site rows (μ and σ of each
        site), the per-channel impact on the tracked advantage."""
        row = np.linalg.norm(self.G[:2 * self.n_sites, :], axis=1)
        return row / (row.max() + 1e-12)
