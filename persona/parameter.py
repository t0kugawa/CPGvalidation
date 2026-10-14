from dataclasses import dataclass, field
import numpy as np


@dataclass
class PersonaParam:
    """
    Biological parameter with two distinct standard deviations:
    - sigma_param: parametric dispersion within the biological neuronal/synaptic population.
    - sigma_ctrl: controlling standard deviation managing exploration vs exploitation.
    """
    mean: float
    sigma_param: float = 0.05
    sigma_ctrl: float = 0.1
    min: float = -1e6
    max: float = 1e6
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @property
    def sigma(self) -> float:
        return self.sigma_param

    @sigma.setter
    def sigma(self, val: float):
        self.sigma_param = float(val)

    def sample(self, n: int) -> np.ndarray:
        """
        Generates biological population values according to sigma_param.
        """
        return self.sample_population(n)

    def sample_population(self, n: int, center: float | None = None) -> np.ndarray:
        """
        Samples a population of n biological entities around the current or explored mean.
        Uses sigma_param for biological heterogeneity.
        """
        mu = self.mean if center is None else center
        x = self.rng.normal(mu, max(1e-8, self.sigma_param), size=n)
        return np.clip(x, self.min, self.max)

    def sample_exploration(self) -> float:
        """
        Samples an exploratory mean for the current step using sigma_ctrl.
        """
        sample = float(self.rng.normal(self.mean, max(1e-6, self.sigma_ctrl)))
        return float(np.clip(sample, self.min, self.max))

    def update_controlling_variance(self, action: float, advantage: float,
                                    lr_sigma: float = 0.005,
                                    sigma_min: float = 1e-3,
                                    sigma_max: float = 2.0):
        """
        Updates controlling exploration variance using the policy gradient derivation:
        grad = ((action - mean)^2 - sigma_ctrl^2) / sigma_ctrl^3 * advantage
        """
        denom = max(1e-8, self.sigma_ctrl ** 3)
        grad = ((action - self.mean) ** 2 - self.sigma_ctrl ** 2) / denom * advantage
        self.sigma_ctrl += lr_sigma * grad
        self.sigma_ctrl = float(np.clip(self.sigma_ctrl, sigma_min, sigma_max))

    def update_mean(self, delta: float):
        self.mean = float(np.clip(self.mean + delta, self.min, self.max))

    def update_parametric_variance(self, delta: float, sigma_min: float = 1e-4, sigma_max: float = 2.0):
        self.sigma_param = float(np.clip(self.sigma_param + delta, sigma_min, sigma_max))
