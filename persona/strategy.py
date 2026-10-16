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

        # Output-sensitivity responses (Variant B): (delta_theta, delta_y)
        self.response_dtheta: List[np.ndarray] = []
        self.response_dy: List[np.ndarray] = []
        self.ridge_J: float = 1e-3
        self.lambda_F: float = 1e-3

        self._scales: np.ndarray | None = None

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
        return np.asarray(self._scales, dtype=np.float64)

    # --- Variant B: output-sensitivity metric -------------------------------

    def record_response(self, delta_theta: np.ndarray, delta_y: np.ndarray):
        """
        Appends one exploratory perturbation and the output response it produced:
        delta_theta = action - mean (n_total,), delta_y = output - output_mean (2,).
        """
        dth = np.asarray(delta_theta, dtype=np.float64).ravel()
        if dth.size < self.n_total:
            dth = np.concatenate([dth, np.zeros(self.n_total - dth.size)])
        elif dth.size > self.n_total:
            dth = dth[:self.n_total]
        self.response_dtheta.append(dth.copy())
        self.response_dy.append(np.asarray(delta_y, dtype=np.float64).ravel().copy())
        if len(self.response_dtheta) > self.window_size:
            self.response_dtheta = self.response_dtheta[-self.window_size:]
            self.response_dy = self.response_dy[-self.window_size:]

    def _sensitivity(self) -> tuple[np.ndarray, np.ndarray]:
        """Ridge least-squares estimate of J = dy/dtheta (n_total x n_y)."""
        W = len(self.response_dtheta)
        if W < 3:
            n_y = len(self.response_dy[0]) if self.response_dy else 2
            return np.zeros((self.n_total, n_y)), np.ones(self.n_total)
        X = np.array(self.response_dtheta, dtype=np.float64)
        Y = np.array(self.response_dy, dtype=np.float64)
        std = np.maximum(X.std(axis=0), 1e-8)
        Xs = X / std
        A = Xs.T @ Xs + self.ridge_J * np.eye(self.n_total)
        Jstd = np.linalg.solve(A, Xs.T @ Y)
        return Jstd, std

    def compute_metric(self) -> tuple[np.ndarray, np.ndarray]:
        """
        F = J JT (standardised columns) + lambda I, plus the per-column scales.
        Diagonal = per-parameter impact on the tracked output; off-diagonal =
        coupling. Returns (F_units, scales).
        """
        Jstd, scales = self._sensitivity()
        self._scales = scales
        F = Jstd @ Jstd.T + self.lambda_F * np.eye(self.n_total)
        return F, scales

    def influence(self) -> np.ndarray:
        """Per-parameter impact on the output, normalised to [0, 1]."""
        Jstd, _ = self._sensitivity()
        row = np.linalg.norm(Jstd, axis=1)
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
        Variant B natural-gradient step:
            Delta mu = eta_mu * S^-1 * F_u^-1 * (J_std . A)
        where A is the (vector) advantage, J_std the standardised output
        sensitivity, and S the per-column scales (lossless unit round-trip).
        TODO(advantage): unify the vector loss/advantage with the score term.
        """
        Jstd, scales = self._sensitivity()
        self._scales = scales
        J = Jstd / scales[:, None]                      # physical dy/dtheta
        F = J @ J.T + self.lambda_F * np.eye(self.n_total)

        adv = np.asarray(advantage_vector, dtype=np.float64).ravel()
        n_y = J.shape[1]
        if adv.size == 1:
            adv = np.full(n_y, float(adv[0]))
        elif adv.size != n_y:
            adv = np.full(n_y, float(np.mean(adv)))
        g = J @ adv                                     # d(loss)/d(theta) direction

        try:
            nat_grad = np.linalg.solve(F, g)
        except np.linalg.LinAlgError:
            nat_grad = np.linalg.pinv(F) @ g

        delta_vec = self.lr_mu * nat_grad

        deltas = {}
        for i, name in enumerate(self.param_names):
            deltas[name] = float(delta_vec[i])
        return deltas
