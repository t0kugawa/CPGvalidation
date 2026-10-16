from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class PhaseBaselines:
    alpha: float = 0.95
    _baselines: Dict[object, np.ndarray] = field(default_factory=dict)

    def get_baseline(self, phase_key: object, dim: int) -> np.ndarray:
        b = self._baselines.get(phase_key)
        if b is None or len(b) != dim:
            b = np.zeros(dim, dtype=np.float64)
            self._baselines[phase_key] = b
        return b

    def update(self, phase_key: object, loss_vector: np.ndarray):
        v = np.asarray(loss_vector, dtype=np.float64)
        b = self._baselines.get(phase_key)
        if b is None or len(b) != len(v):
            self._baselines[phase_key] = v.copy()
        else:
            self._baselines[phase_key] = self.alpha * b + (1.0 - self.alpha) * v


class FisherMatrix:
    """
    Fisher / Natural Gradient Strategy module.
    Maintains empirical covariance/correlation matrix of all parameters plus noise producer deltas.
    Calculates component-wise advantage separated for stance and swing phases,
    and derives natural gradient update steps Delta mu = eta_mu * F^{-1} * grad_log_pi.
    """
    def __init__(self,
                 param_names: List[str],
                 noise_names: List[str] | None = None,
                 lr_mu: float = 0.02,
                 lr_sigma: float = 0.005,
                 window_size: int = 50,
                 lambda_reg: float = 1e-3,
                 baseline_alpha: float = 0.90):
        self.param_names = list(param_names)
        self.noise_names = list(noise_names or [])
        self.all_names = self.param_names + self.noise_names
        self.n_params = len(self.param_names)
        self.n_total = len(self.all_names)

        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.window_size = window_size
        self.lambda_reg = lambda_reg

        self.baselines = PhaseBaselines(alpha=baseline_alpha)

        # Rolling histories
        self.perturbation_history: List[np.ndarray] = []
        self.loss_history: List[np.ndarray] = []
        self.phase_history: List[str] = []

        # Perturbation–advantage samples (natural gradient): (delta_theta, A)
        self.response_dtheta: List[np.ndarray] = []
        self.response_da: List[np.ndarray] = []
        self.lambda_F: float = float(lambda_reg)
        self.ridge_J: float = 1e-3  # kept for backwards-compatible imports

        self._scales: np.ndarray | None = None
        self._G: np.ndarray | None = None

    def update_feature_names(self, noise_names: List[str]):
        """
        Dynamically updates noise producer tags if new ones are observed.
        """
        for n in noise_names:
            if n not in self.noise_names:
                self.noise_names.append(n)
        self.all_names = self.param_names + self.noise_names
        self.n_total = len(self.all_names)

    def compute_advantage(self, loss_vector: np.ndarray, phase: object) -> np.ndarray:
        loss = np.asarray(loss_vector, dtype=np.float64)
        base = self.baselines.get_baseline(phase, len(loss))
        # TODO(advantage): component-wise vector advantage (see spec §5.1) —
        # currently baseline − loss for the scalarized loss.
        advantage = base - loss
        self.baselines.update(phase, loss)
        return advantage

    def record_step(self,
                    perturbation_dict: Dict[str, float],
                    loss_vector: np.ndarray,
                    phase: object):
        # Ensure all features exist in feature list
        noise_keys = [k for k in perturbation_dict.keys() if k.startswith("noise_")]
        self.update_feature_names(noise_keys)

        vec = np.zeros(self.n_total, dtype=np.float64)
        for i, name in enumerate(self.all_names):
            vec[i] = perturbation_dict.get(name, 0.0)

        self.perturbation_history.append(vec)
        self.loss_history.append(np.asarray(loss_vector, dtype=np.float64))
        self.phase_history.append(phase)

        if len(self.perturbation_history) > self.window_size:
            self.perturbation_history = self.perturbation_history[-self.window_size:]
            self.loss_history = self.loss_history[-self.window_size:]
            self.phase_history = self.phase_history[-self.window_size:]

    def compute_fisher_matrix(self) -> np.ndarray:
        """
        Backwards-compatible accessor: returns the metric matrix.
        """
        return self.compute_metric()[0]

    def feature_scales(self) -> np.ndarray:
        if self._scales is None:
            self.compute_metric()
        return np.asarray(self._scales, dtype=np.float64)

    # --- Perturbation–advantage covariance (natural gradient) ---------------

    def record_response(self, delta_theta: np.ndarray, advantage: np.ndarray):
        """
        Appends one exploration sample: delta_theta = action - mean (n_total,)
        paired with the reward advantage A (n_y,).
        """
        dth = np.asarray(delta_theta, dtype=np.float64).ravel()
        if dth.size < self.n_total:
            dth = np.concatenate([dth, np.zeros(self.n_total - dth.size)])
        elif dth.size > self.n_total:
            dth = dth[:self.n_total]
        self.response_dtheta.append(dth.copy())
        self.response_da.append(np.asarray(advantage, dtype=np.float64).ravel().copy())
        if len(self.response_dtheta) > self.window_size:
            self.response_dtheta = self.response_dtheta[-self.window_size:]
            self.response_da = self.response_da[-self.window_size:]

    def _stack(self) -> tuple[np.ndarray, np.ndarray]:
        W = len(self.response_dtheta)
        n_y = max((a.size for a in self.response_da), default=2)
        X = np.zeros((W, self.n_total), dtype=np.float64)
        Y = np.zeros((W, n_y), dtype=np.float64)
        for i in range(W):
            d = self.response_dtheta[i]
            X[i, :min(d.size, self.n_total)] = d[:self.n_total]
            a = self.response_da[i]
            Y[i, :min(a.size, n_y)] = a[:n_y]
        return X, Y

    def _covariances(self) -> tuple[np.ndarray, np.ndarray]:
        """
        F = Cov(Δθ) + λ_F I          (n_total x n_total)  — metric / preconditioner
        G = Cov(Δθ, A)               (n_total x n_y)      — per-parameter impact
        Column means removed over the rolling window.
        """
        W = len(self.response_dtheta)
        if W < 3:
            n_y = max((a.size for a in self.response_da), default=2)
            return np.eye(self.n_total, dtype=np.float64), np.zeros((self.n_total, n_y))
        X, Y = self._stack()
        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        F = (Xc.T @ Xc) / W + self.lambda_F * np.eye(self.n_total)
        G = (Xc.T @ Yc) / W
        self._G = G
        return F, G

    def compute_metric(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (F, scales); F = Cov(Δθ) + λ_F I, scales = sqrt(diag(F))."""
        F, _ = self._covariances()
        self._scales = np.sqrt(np.maximum(np.diag(F), 1e-12))
        return F, self._scales

    def compute_gradient(self) -> np.ndarray:
        """g = Cov(Δθ, ΣA) — the natural-gradient direction (n_total,)."""
        _, G = self._covariances()
        return G.sum(axis=1)

    def influence(self) -> np.ndarray:
        """Per-parameter impact |Cov(Δθ_i, A)|, normalised to [0, 1]."""
        _, G = self._covariances()
        row = np.linalg.norm(G, axis=1)
        return row / (row.max() + 1e-12)

    def compute_correlation_matrix(self) -> np.ndarray:
        """Deprecated alias for the metric matrix (kept for importers)."""
        return self.compute_metric()[0]

    def compute_natural_gradient_step(self,
                                      action_dict: Dict[str, float],
                                      mean_dict: Dict[str, float],
                                      sigma_ctrl_dict: Dict[str, float],
                                      advantage_vector: np.ndarray) -> Dict[str, float]:
        """
        Natural-gradient ascent on the reward advantage:
            Delta mu = eta_mu * F^-1 * g,   F = Cov(Δθ) + lambda I,  g = Cov(Δθ, ΣA).
        Equivalently the ridge regression of the advantage on the perturbations.
        E[Delta mu_i] = -eta * d E[ΣE] / d theta_i  (ascent on reward = descent on error).
        TODO(advantage): the baselined advantage is the reward signal; unify the
        per-leg loss/advantage weighting with the score term.
        """
        F, G = self._covariances()
        g = G.sum(axis=1)
        try:
            nat_grad = np.linalg.solve(F, g)
        except np.linalg.LinAlgError:
            nat_grad = np.linalg.pinv(F) @ g

        delta_vec = self.lr_mu * nat_grad

        deltas = {}
        for i, name in enumerate(self.param_names):
            deltas[name] = float(delta_vec[i])
        return deltas
