from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class PhaseBaselines:
    alpha: float = 0.95
    stance_baseline: np.ndarray | None = None
    swing_baseline: np.ndarray | None = None

    def get_baseline(self, phase: str, dim: int) -> np.ndarray:
        if phase == "swing":
            if self.swing_baseline is None or len(self.swing_baseline) != dim:
                self.swing_baseline = np.zeros(dim, dtype=np.float64)
            return self.swing_baseline
        else:
            if self.stance_baseline is None or len(self.stance_baseline) != dim:
                self.stance_baseline = np.zeros(dim, dtype=np.float64)
            return self.stance_baseline

    def update(self, phase: str, reward_vector: np.ndarray):
        r = np.asarray(reward_vector, dtype=np.float64)
        if phase == "swing":
            if self.swing_baseline is None or len(self.swing_baseline) != len(r):
                self.swing_baseline = r.copy()
            else:
                self.swing_baseline = self.alpha * self.swing_baseline + (1.0 - self.alpha) * r
        else:
            if self.stance_baseline is None or len(self.stance_baseline) != len(r):
                self.stance_baseline = r.copy()
            else:
                self.stance_baseline = self.alpha * self.stance_baseline + (1.0 - self.alpha) * r


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

    def update_feature_names(self, noise_names: List[str]):
        """
        Dynamically updates noise producer tags if new ones are observed.
        """
        for n in noise_names:
            if n not in self.noise_names:
                self.noise_names.append(n)
        self.all_names = self.param_names + self.noise_names
        self.n_total = len(self.all_names)

    def compute_advantage(self, reward_vector: np.ndarray, phase: str) -> np.ndarray:
        r = np.asarray(reward_vector, dtype=np.float64)
        base = self.baselines.get_baseline(phase, len(r))
        advantage = r - base
        self.baselines.update(phase, r)
        return advantage

    def record_step(self,
                    perturbation_dict: Dict[str, float],
                    advantage_vector: np.ndarray,
                    reward_vector: np.ndarray,
                    phase: str):
        # Ensure all features exist in feature list
        noise_keys = [k for k in perturbation_dict.keys() if k.startswith("noise_")]
        self.update_feature_names(noise_keys)

        vec = np.zeros(self.n_total, dtype=np.float64)
        for i, name in enumerate(self.all_names):
            vec[i] = perturbation_dict.get(name, 0.0)

        self.perturbation_history.append(vec)
        self.advantage_history.append(np.asarray(advantage_vector, dtype=np.float64))
        self.reward_history.append(np.asarray(reward_vector, dtype=np.float64))
        self.phase_history.append(phase)

        if len(self.perturbation_history) > self.window_size:
            self.perturbation_history = self.perturbation_history[-self.window_size:]
            self.advantage_history = self.advantage_history[-self.window_size:]
            self.reward_history = self.reward_history[-self.window_size:]
            self.phase_history = self.phase_history[-self.window_size:]

    def compute_fisher_matrix(self) -> np.ndarray:
        """
        Calculates the regularized empirical Fisher / covariance matrix:
        F = (X^T * X) / N + lambda * I
        """
        if len(self.perturbation_history) < 3:
            return np.eye(self.n_total, dtype=np.float64)

        X = np.array(self.perturbation_history, dtype=np.float64)
        # Center around column mean to obtain proper sample covariance / correlation
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        cov = (X_centered.T @ X_centered) / max(1, len(X) - 1)
        F = cov + self.lambda_reg * np.eye(self.n_total, dtype=np.float64)
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

        F = self.compute_fisher_matrix()
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
