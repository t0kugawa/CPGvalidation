from dataclasses import dataclass, field
import itertools
import numpy as np

_RNG_STREAM = itertools.count(1)
_RNG_BASE_SEED = 0


def reset_rng_stream(base_seed: int = 0) -> None:
    """Resets the shared spawn stream so a run is reproducible."""
    global _RNG_STREAM, _RNG_BASE_SEED
    _RNG_STREAM = itertools.count(1)
    _RNG_BASE_SEED = int(base_seed)


def _new_rng() -> np.random.Generator:
    spawn = next(_RNG_STREAM)
    return np.random.default_rng(np.random.SeedSequence(_RNG_BASE_SEED, spawn_key=(spawn,)))


@dataclass
class PersonaParam:
    """
    Biological parameter with two distinct standard deviations:
    - sigma_param: parametric dispersion within the biological neuronal/synaptic population.
    - sigma_ctrl: controlling standard deviation managing exploration vs exploitation.

    Each instance owns an independent RNG stream so different parameters explore
    independent directions (a shared seed would make every parameter's k-th draw
    identical and collapse the Fisher to rank 1).
    """
    mean: float
    sigma_param: float = 0.05
    sigma_ctrl: float = 0.1
    min: float = -1e6
    max: float = 1e6
    rng: np.random.Generator = field(default_factory=_new_rng)

    @property
    def sigma(self) -> float:
        return self.sigma_param

    @sigma.setter
    def sigma(self, val: float):
        self.sigma_param = float(val)

    @property
    def log_sigma_param(self) -> float:
        """Natural-log of the parametric dispersion (a feature in theta)."""
        return float(np.log(max(self.sigma_param, 1e-8)))

    def set_log_sigma_param(self, value: float):
        """Sets sigma_param = exp(value), clipped to [1e-4, 2.0]."""
        self.sigma_param = float(np.clip(np.exp(value), 1e-4, 2.0))

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
