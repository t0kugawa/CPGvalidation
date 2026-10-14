import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any
from persona.parameter import PersonaParam


@dataclass
class Population:
    model: str
    n: int
    params: Dict[str, PersonaParam]
    name: str
    corr: np.ndarray | None = None
    node_collection: Any = None

    def sample_matrix(self, explored_means: Dict[str, float] | None = None) -> Dict[str, list]:
        """
        Samples neuron parameters for all n neurons in the population.
        If explored_means is provided, uses those as the centers; otherwise uses param.mean.
        The biological dispersion across neurons is governed by param.sigma_param.
        """
        keys = list(self.params.keys())
        n = self.n
        if not keys or n <= 0:
            return {}

        means = np.array([
            explored_means[k] if (explored_means is not None and k in explored_means) else self.params[k].mean
            for k in keys
        ], dtype=np.float64)

        sigmas = np.array([self.params[k].sigma_param for k in keys], dtype=np.float64)

        if self.corr is None:
            sampled_dict = {}
            for i, k in enumerate(keys):
                p = self.params[k]
                vals = p.sample_population(n, center=means[i])
                sampled_dict[k] = vals.tolist()
            return sampled_dict

        cov = self.corr * np.outer(sigmas, sigmas)
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(keys)))
        rng = self.params[keys[0]].rng
        z = rng.standard_normal((n, len(keys)))
        X = means + z @ L.T

        sampled_dict = {}
        for i, k in enumerate(keys):
            p = self.params[k]
            clamped = np.clip(X[:, i], p.min, p.max)
            sampled_dict[k] = clamped.tolist()
        return sampled_dict
