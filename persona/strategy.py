from dataclasses import dataclass, field
from typing import Dict, List, Tuple
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
        self.advantage_history: List[np.ndarray] = []
        self.reward_history: List[np.ndarray] = []
        self.phase_history: List[str] = []

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
        self.advantage_history.append(np.asarray(loss_vector, dtype=np.float64))
        self.reward_history.append(np.asarray(loss_vector, dtype=np.float64))
        self.phase_history.append(phase)

        if len(self.perturbation_history) > self.window_size:
            self.perturbation_history = self.perturbation_history[-self.window_size:]
            self.advantage_history = self.advantage_history[-self.window_size:]
            self.reward_history = self.reward_history[-self.window_size:]
            self.phase_history = self.phase_history[-self.window_size:]

    def compute_fisher_matrix(self) -> np.ndarray:
        """
        Calculates the regularized empirical correlation matrix of all features
        (parameters + noise). Columns are standardized to unit variance so
        heterogeneous units (Hz, pA, probabilities, rad) enter on equal footing.
        """
        return self.compute_correlation_matrix()

    def feature_scales(self) -> np.ndarray:
        return np.asarray(self._scales, dtype=np.float64)

    def compute_correlation_matrix(self) -> np.ndarray:
        """
        F = (1 - lambda) * Corr + lambda * I, where Corr is the empirical
        correlation of the feature-perturbation history. Unit diagonal and
        strictly positive definite.
        """
        if len(self.perturbation_history) < 3:
            self._scales = np.ones(self.n_total, dtype=np.float64)
            return np.eye(self.n_total, dtype=np.float64)

        X = np.array(self.perturbation_history, dtype=np.float64)
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        self._scales = np.maximum(std.ravel(), 1e-12)
        Xs = (X - mean) / self._scales
        corr = (Xs.T @ Xs) / max(1, len(X))
        F = (1.0 - self.lambda_reg) * corr + self.lambda_reg * np.eye(self.n_total)
        return F

    def compute_natural_gradient_step(self,
                                      action_dict: Dict[str, float],
                                      mean_dict: Dict[str, float],
                                      sigma_ctrl_dict: Dict[str, float],
                                      advantage_vector: np.ndarray) -> Dict[str, float]:
        """
        Computes Delta mu = eta_mu * F^{-1} * score.
        Score = (action - mean) / (sigma_ctrl^2) * advantage_scalar.
        """
        adv_scalar = float(np.mean(advantage_vector))

        grad_log_pi = np.zeros(self.n_total, dtype=np.float64)
        for i, name in enumerate(self.param_names):
            if name in action_dict and name in mean_dict and name in sigma_ctrl_dict:
                diff = action_dict[name] - mean_dict[name]
                s_ctrl = max(1e-6, sigma_ctrl_dict[name])
                grad_log_pi[i] = (diff / (s_ctrl ** 2)) * adv_scalar

        # TODO(fisher): recursive / Sherman-Morrison inverse or a pruned
        # feature subspace (see spec §5.1 well-posedness).
        F = self.compute_correlation_matrix()
        try:
            nat_grad = np.linalg.solve(F, grad_log_pi)
        except np.linalg.LinAlgError:
            nat_grad = np.linalg.pinv(F) @ grad_log_pi

        # Apply learning rate
        delta_vec = self.lr_mu * nat_grad

        # Return dict only for controllable parameters
        deltas = {}
        for i, name in enumerate(self.param_names):
            deltas[name] = float(delta_vec[i])

        return deltas
