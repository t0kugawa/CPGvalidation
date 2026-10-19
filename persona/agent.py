from typing import Dict, List, Any
import numpy as np

from persona.registry import Registry
from persona.strategy import FisherMatrix
from persona.parameter import PersonaParam


class Agent:
    """
    PERSONA agent: samples each site's exploration (recorded = applied through
    the site recorders), scores the step from its sampling-time snapshot, and
    runs the persisted strategy (F/Xbar/G with β = 1 − dt/τ_mem) every dt.

    Host state is re-read every step (`refresh_params`), so sites that appear
    later are picked up automatically (zero history); a site that disappears is
    an error, not a silent drop.
    """
    def __init__(self,
                 registry: Registry,
                 strategy: FisherMatrix | None = None,
                 lr_mu: float = 0.02,
                 lr_sigma: float = 0.005,
                 tau_mem_s: float = 2.0,
                 dt_ms: float = 10.0,
                 warmup: int = 3,
                 smooth_kernel_size: int = 5):
        self.registry = registry
        self.lr_mu = float(lr_mu)
        self.lr_sigma = float(lr_sigma)
        self.dt_ms = float(dt_ms)
        self.warmup = int(warmup)
        self.smooth_kernel_size = int(smooth_kernel_size)
        beta = 1.0 - (self.dt_ms / 1000.0) / max(float(tau_mem_s), 1e-6)
        self.beta = float(np.clip(beta, 0.0, 0.999999))

        self.current_actions: Dict[str, float] = {}
        self.current_step = 0
        self._snap: Dict[str, tuple[float, float, float]] = {}
        self.output_buffer: List[np.ndarray] = []
        self.target_buffer: List[np.ndarray] = []
        self.history: Dict[str, Any] = {
            "error_vector": [],
            "loss": [],
            "advantage_vector": [],
            "phase": [],
            "influence": [],
            "means": {},
            "sigmas": {},
        }

        self.params: Dict[str, PersonaParam] = {}
        self.refresh_params()

        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = FisherMatrix(
                param_names=list(self.params.keys()),
                noise_names=sorted(self.registry.producer_recorders()),
                lr_mu=self.lr_mu,
                lr_sigma=self.lr_sigma,
                beta=self.beta,
                warmup=self.warmup,
            )

        # Noise statistics share the strategy's forgetting factor.
        for rec in self.registry.noise_recorders.values():
            rec.beta = self.beta

    # --- host state ---------------------------------------------------------

    def refresh_params(self) -> Dict[str, PersonaParam]:
        """
        Re-reads the registry's personalizable parameters (every step): new
        sites are picked up with zero history, removed sites raise.
        """
        params = self.registry.get_all_optimizable_params()
        known = set(self.params)
        if known:
            missing = known - set(params)
            if missing:
                raise RuntimeError(
                    f"PERSONA sites disappeared from the registry: {sorted(missing)}")
        for name in params:
            self.history["means"].setdefault(name, [])
            self.history["sigmas"].setdefault(name, [])
        self.params = params
        return params

    # --- per-step API -------------------------------------------------------

    def sample_actions(self, t_ms: float | None = None) -> Dict[str, float]:
        """
        Samples each site's exploratory value. The sample is written into the
        site recorder together with its sampling-time sigma (recorded = applied
        by construction), and the (μ, σ, x̂) snapshot is kept for the step.
        """
        self.refresh_params()
        t = 0.0 if t_ms is None else float(t_ms)
        self.current_actions = {}
        self._snap = {}
        for name, param in self.params.items():
            mu = float(param.mean)
            sig = float(param.sigma)
            xhat = param.sample_exploration()
            self.current_actions[name] = xhat
            self._snap[name] = (mu, sig, xhat)
            self.registry.site_recorder(name).record_sample(t, xhat, sig)
        return dict(self.current_actions)

    def _smooth(self, signal: np.ndarray, buffer: List[np.ndarray]) -> np.ndarray:
        buf_arr = np.asarray(signal, dtype=np.float64)
        buffer.append(buf_arr)
        if len(buffer) > self.smooth_kernel_size:
            buffer.pop(0)
        return np.mean(np.array(buffer), axis=0)

    def preprocess_signal(self, raw_signal: np.ndarray, zero_shift: float = 0.0) -> np.ndarray:
        """
        Preprocesses the model output signal:
        1. Zero-point shift
        2. Moving average over the last `smooth_kernel_size` steps
        """
        sig = np.asarray(raw_signal, dtype=np.float64) - zero_shift
        return self._smooth(sig, self.output_buffer)

    def preprocess_target(self, raw_target: np.ndarray) -> np.ndarray:
        """
        Applies the same moving average to the target so output and target
        share one filter (no differential lag between the two signals).
        """
        return self._smooth(raw_target, self.target_buffer)

    def compute_vector_error(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Vector error: E_v = |target - output|
        """
        out = np.asarray(output, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        return np.abs(tgt - out)

    def step(self,
             output: np.ndarray,
             target: np.ndarray,
             phase: str = "stance",
             zero_shift: float = 0.0,
             t_ms: float | None = None) -> Dict[str, Any]:
        """
        Runs one optimization step:
        1. Preprocesses output/target with the same moving-average filter.
        2. Vector error and component-wise advantage (phase baselines).
        3. Per-site sampling-time snapshot channels; producer deltas are read
           fresh (read-and-clear, zero if the step recorded nothing).
        4. Strategy step: F/Xbar/G recurrences + natural-gradient Δμ.
        5. From step warmup+1 on, applies Δμ and the log-σ update; during
           warm-up both are 0.0 — and the trace records exactly that, so
           "traced = applied" holds by construction.

        TODO(advantage): the component-wise vector error/advantage (spec §5.1)
        is baselined per limb; unify the per-leg weighting with the score term.
        """
        self.current_step += 1
        self.refresh_params()
        smoothed_out = self.preprocess_signal(output, zero_shift=zero_shift)
        smoothed_tgt = self.preprocess_target(target)

        # Vector error and scalarized loss
        error_vec = self.compute_vector_error(smoothed_out, smoothed_tgt)
        loss = float(np.mean(error_vec))

        # Normalize phase: str broadcasts to one key per output component;
        # a sequence is keyed per component (per-leg phase).
        if isinstance(phase, str):
            phase_key = phase
        else:
            phase_key = tuple(phase)

        advantage_vec = self.strategy.compute_advantage(error_vec, phase=phase_key)
        adv_scalar = float(np.mean(advantage_vec))

        apply_updates = (self.strategy.steps + 1) > self.strategy.warmup

        # Sampling-time snapshot channels (spec §4): score units for the trace.
        sites: Dict[str, tuple] = {}
        for name, param in self.params.items():
            mu_at, sig_at, xhat = self._snap.get(
                name, (float(param.mean), float(param.sigma),
                       float(self.current_actions.get(name, param.mean))))
            d_log_sigma = 0.0
            if apply_updates:
                d_log_sigma = param.sigma_step_applied(
                    sample=xhat, mu_at_sample=mu_at, sigma_at_sample=sig_at,
                    advantage=adv_scalar, lr_sigma=self.lr_sigma)
            sites[name] = (float(xhat) - float(mu_at), float(sig_at), float(d_log_sigma))

        # Producer deltas: fresh since the previous step; a step without new
        # records contributes 0.0, never a stale value.
        noise: Dict[str, tuple] = {}
        for rec_name, rec in self.registry.producer_recorders().items():
            delta = rec.take_delta()
            tau = max(float(rec.stats()["std"]), 1e-6)
            noise[rec_name] = (0.0 if delta is None else float(delta), tau)

        delta_means = self.strategy.step(sites, noise, advantage_vec)

        # Apply to the host (from warmup+1): means first, then sigma — both from
        # the sampling-time snapshot, so neither depends on the other's change.
        for name, param in self.params.items():
            mu_at, sig_at, xhat = self._snap.get(
                name, (float(param.mean), float(param.sigma),
                       float(self.current_actions.get(name, param.mean))))
            if apply_updates and name in delta_means:
                param.update_mean(delta_means[name])
            if apply_updates:
                param.update_sigma(sample=xhat, mu_at_sample=mu_at,
                                   sigma_at_sample=sig_at, advantage=adv_scalar,
                                   lr_sigma=self.lr_sigma)

        # Logging
        self.history["error_vector"].append(np.asarray(error_vec, dtype=np.float64))
        self.history["loss"].append(loss)
        self.history["advantage_vector"].append(np.asarray(advantage_vec, dtype=np.float64))
        self.history["phase"].append(phase_key)
        self.history["influence"].append(self.strategy.influence())
        for name, param in self.params.items():
            self.history["means"].setdefault(name, []).append(float(param.mean))
            self.history["sigmas"].setdefault(name, []).append(float(param.sigma))

        return {
            "step": self.current_step,
            "error_vector": np.asarray(error_vec, dtype=np.float64),
            "loss": loss,
            "advantage_vector": np.asarray(advantage_vec, dtype=np.float64),
            "phase": phase_key,
            "influence": self.strategy.influence(),
            "delta_means": dict(delta_means),
        }
