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
    Personalizable parameter with a single standard deviation `sigma` that
    serves both roles at once:
    - exploration/exploitation: the sampled site value is mean + sigma * eps;
    - biological dispersion: every realized instance is value + sigma * xi.

    The recorded sample is therefore exactly what is applied, and the strategy
    scores every site with the same sigma it was sampled with.

    Each instance owns an independent RNG stream so different parameters explore
    independent directions (a shared seed would make every parameter's k-th draw
    identical and collapse the Fisher to rank 1).
    """
    mean: float
    sigma: float = 0.1
    min: float = -1e6
    max: float = 1e6
    sigma_min: float = 1e-4
    sigma_max: float | None = None
    rng: np.random.Generator = field(default_factory=_new_rng)

    def __post_init__(self):
        if self.sigma_max is None:
            # A site must never start out of bounds; large-scale sites (Izh `c`,
            # `Wmax`, drive rates) keep their initial sigma and can only shrink.
            self.sigma_max = max(2.0, float(self.sigma))

    def sample(self, n: int) -> np.ndarray:
        """Generates biological population values according to sigma."""
        return self.sample_population(n)

    def sample_population(self, n: int, center: float | None = None) -> np.ndarray:
        """
        Samples a population of n biological entities around the current or
        explored center, using the single sigma for biological dispersion.
        """
        mu = self.mean if center is None else center
        x = self.rng.normal(mu, max(1e-8, self.sigma), size=n)
        return np.clip(x, self.min, self.max)

    def sample_exploration(self) -> float:
        """Samples an exploratory value (the applied site sample) using sigma."""
        sample = float(self.rng.normal(self.mean, max(1e-6, self.sigma)))
        return float(np.clip(sample, self.min, self.max))

    def sample_explorations(self, n: int) -> np.ndarray:
        """Draw ``n`` different exploratory values in one vectorised call.

        The multi-replica path uses this to give every replica its own sample
        without paying one Python call per site and replica.
        """
        count = int(n)
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        if count == 1:
            return np.asarray([self.sample_exploration()], dtype=np.float64)
        values = self.rng.normal(
            self.mean, max(1e-6, self.sigma), size=count)
        return np.clip(values, self.min, self.max)

    def sigma_step(self, sample: float, mu_at_sample: float, sigma_at_sample: float,
                   advantage: float, lr_sigma: float = 0.005) -> float:
        """
        Log-variance policy-gradient step for the single sigma (variant A):
        d log(sigma) = lr * ((sample - mu_at_sample)^2 / sigma_at_sample^2 - 1) * advantage,
        clipped to +/-0.25 per step so sigma can never jump.

        Uses the sampling-time (mu, sigma) snapshot, so the update is
        independent of any mean change that happened later in the same step.

        TODO(sigma-ewma): variant B pairs (z^2 - 1) with an EWMA advantage
        Abar_t = beta*Abar_{t-1} + (1-beta)*A_t instead of the current A_t.
        Keep this marker so the two variants can be A/B-tested on the same run.
        """
        sig = max(float(sigma_at_sample), 1e-12)
        z2 = ((float(sample) - float(mu_at_sample)) / sig) ** 2
        return float(np.clip(lr_sigma * (z2 - 1.0) * float(advantage), -0.25, 0.25))

    def update_sigma(self, sample: float, mu_at_sample: float, sigma_at_sample: float,
                     advantage: float, lr_sigma: float = 0.005) -> float:
        """Applies `sigma_step` to the sampling-time sigma; sets and returns it."""
        new = float(np.clip(
            float(sigma_at_sample) * np.exp(
                self.sigma_step(sample, mu_at_sample, sigma_at_sample, advantage, lr_sigma)),
            self.sigma_min, self.sigma_max))
        self.sigma = new
        return new

    def update_sigma_batch(self, samples, mu_at_sample: float,
                           sigma_at_sample: float, advantages,
                           lr_sigma: float = 0.005,
                           weights=None) -> float:
        """Weighted mean of several sampling-time log-σ steps.

        Used by the multi-replica merge: every replica contributes its own
        ``sigma_step`` and the policy sigma receives the weighted mean once.
        """
        samples = list(samples)
        advantages = list(advantages)
        if len(samples) != len(advantages):
            raise ValueError("samples and advantages must have equal length")
        if not samples:
            return float(self.sigma)
        steps = [self.sigma_step(s, mu_at_sample, sigma_at_sample, a, lr_sigma)
                 for s, a in zip(samples, advantages)]
        if weights is None:
            dlog = float(np.mean(steps))
        else:
            w = np.asarray(weights, dtype=np.float64)
            total = float(w.sum())
            dlog = (float(np.dot(w, steps)) / total
                    if np.isfinite(total) and total > 0.0
                    else float(np.mean(steps)))
        new = float(np.clip(
            float(sigma_at_sample) * np.exp(dlog),
            self.sigma_min, self.sigma_max))
        self.sigma = new
        return new

    def sigma_step_applied(self, sample: float, mu_at_sample: float, sigma_at_sample: float,
                           advantage: float, lr_sigma: float = 0.005) -> float:
        """
        The Δlogσ that `update_sigma` will actually apply: the clipped log-step
        pushed through the σ bounds. The trace's σ row records this value, so
        "traced = applied" holds exactly (spec §4, b_s) even at the bounds.
        """
        new = float(np.clip(
            float(sigma_at_sample) * np.exp(
                self.sigma_step(sample, mu_at_sample, sigma_at_sample, advantage, lr_sigma)),
            self.sigma_min, self.sigma_max))
        sig = max(float(sigma_at_sample), 1e-300)
        return float(np.log(new / sig))

    def update_mean(self, delta: float):
        self.mean = float(np.clip(self.mean + delta, self.min, self.max))
