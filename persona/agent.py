from typing import Dict, List, Any
import numpy as np

from persona.registry import Registry
from persona.strategy import FisherMatrix
from persona.parameter import PersonaParam
from persona.sampling import ExplorationSampler
from persona.game import GameConfig, StackGame
from persona.homeostasis import (
    HomeostaticConfig,
    HomeostaticRegulator,
    PhysicalMap,
)
from persona.schedule import CosineLR


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
                 smooth_kernel_size: int = 5,
                 homeostasis: HomeostaticConfig | None = None,
                 game: GameConfig | None = None,
                 lr_schedule: CosineLR | None = None,
                 sampling_threads: int = 1):
        self.registry = registry
        self.lr_mu = float(lr_mu)
        self.lr_sigma = float(lr_sigma)
        self.lr_schedule = lr_schedule
        self.sampler = ExplorationSampler(sampling_threads)
        self.dt_ms = float(dt_ms)
        self.warmup = int(warmup)
        self.smooth_kernel_size = int(smooth_kernel_size)
        beta = 1.0 - (self.dt_ms / 1000.0) / max(float(tau_mem_s), 1e-6)
        self.beta = float(np.clip(beta, 0.0, 0.999999))

        self.current_actions: Dict[str, float] = {}
        self.current_step = 0
        self._prev_phase = None
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
        if self.lr_schedule is not None:
            self.history["lr_mu"] = []

        self.params: Dict[str, PersonaParam] = {}
        self.homeostasis = None
        self.refresh_params()

        if homeostasis is not None and homeostasis.enabled:
            physical_map = PhysicalMap.from_registry(
                registry, eps_H=homeostasis.eps_H)
            self.homeostasis = HomeostaticRegulator(
                physical_map, homeostasis, dt_s=self.dt_ms / 1000.0,
                beta=self.beta, site_names=list(self.params.keys()))
            self.history["homeostasis"] = []

        self.game = None
        if game is not None and game.enabled:
            game_map = PhysicalMap.from_registry(registry)
            self.game = StackGame(
                game_map, game, dt_s=self.dt_ms / 1000.0,
                beta=self.beta, site_names=list(self.params.keys()))
            self.history["game"] = []

        if strategy is not None:
            self.strategy = strategy
            if self.homeostasis is not None:
                self.strategy.homeostasis = self.homeostasis
            if self.game is not None:
                self.strategy.game = self.game
            if (self.homeostasis is not None
                    or self.game is not None):
                self.strategy.dt_s = self.dt_ms / 1000.0
                if hasattr(self.strategy, "ensure_names"):
                    self.strategy.ensure_names(self.strategy.param_names,
                                               self.strategy.noise_names)
        else:
            self.strategy = FisherMatrix(
                param_names=list(self.params.keys()),
                noise_names=sorted(self.registry.producer_recorders()),
                lr_mu=self.lr_mu,
                lr_sigma=self.lr_sigma,
                beta=self.beta,
                warmup=self.warmup,
                homeostasis=self.homeostasis,
                game=self.game,
                dt_s=self.dt_ms / 1000.0,
            )

    # --- host state ---------------------------------------------------------

    @property
    def current_lr_mu(self) -> float:
        """Learning rate for the current 1-based step (scheduled or fixed)."""
        if self.lr_schedule is None:
            return self.lr_mu
        return self.lr_schedule.at(self.current_step)

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
        # Noise statistics share the strategy's forgetting factor — also for
        # recorders created after this agent was constructed (spec §5.3).
        for rec in self.registry.noise_recorders.values():
            if rec.beta != self.beta:
                rec.beta = self.beta
        self.params = params
        if getattr(self, "homeostasis", None) is not None:
            self.homeostasis.ensure_params(params.keys())
        if getattr(self, "game", None) is not None:
            self.game.ensure_params(params.keys())
        return params

    # --- per-step API -------------------------------------------------------

    def sample_actions(self, t_ms: float | None = None) -> Dict[str, float]:
        """
        Samples each site's exploratory value and keeps the (μ, σ, x̂) snapshot
        for the step. The site recorder is written by the registry's realize
        step, so recorded = applied with one record per realized step
        (the realize step's own clock supplies the recording timestamp).
        """
        self.refresh_params()
        # Snapshot (mu, sigma) before sampling: they are constant for the
        # whole call, and threaded sampling must not change them.
        snapshot = {name: (float(param.mean), float(param.sigma))
                    for name, param in self.params.items()}
        self.current_actions = self.sampler.sample(self.params)
        self._snap = {
            name: (snapshot[name][0], snapshot[name][1],
                   float(self.current_actions[name]))
            for name in self.params
        }
        return dict(self.current_actions)

    def close(self) -> None:
        """Release the sampling thread pool (if any)."""
        self.sampler.close()

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
             t_ms: float | None = None,
             *,
             advantage_phase=None) -> Dict[str, Any]:
        """
        Runs one optimization step:
        1. Preprocesses output/target with the same moving-average filter.
        2. Vector error and component-wise advantage (phase baselines).
        3. Per-site sampling-time snapshot channels; producer deltas are read
           fresh (read-and-clear, zero if the step recorded nothing).
        4. Strategy step: F/Xbar/G recurrences + natural-gradient Δμ; the
           recursive inverse refreshes on a phase shift or at K_max.
        5. From step warmup+1 on, applies Δμ and the log-σ update; during
           warm-up both are 0.0 — and the trace records exactly that, so
           "traced = applied" holds by construction.

        TODO(advantage): the component-wise vector error/advantage (spec §5.1)
        is baselined per limb; unify the per-leg weighting with the score term.
        """
        self.current_step += 1
        lr_t = self.current_lr_mu
        self.refresh_params()
        smoothed_out = self.preprocess_signal(output, zero_shift=zero_shift)
        smoothed_tgt = self.preprocess_target(target)

        # Vector error and scalarized loss
        error_vec = self.compute_vector_error(smoothed_out, smoothed_tgt)
        loss = float(np.mean(error_vec))

        # Normalize phase to one key per output component so a string and an
        # equivalent per-leg tuple compare equal ("stance" == ("stance",)*n).
        phase_key = tuple(
            self.strategy._per_component_phases(phase, len(error_vec)))
        phase_record = phase if isinstance(phase, str) else phase_key
        # The advantage baseline must be conditioned on the *reference* gait
        # phase when one is available: the model phase is what the policy is
        # changing, so using it makes the baseline endogenous. `phase` stays
        # the model phase for logging and the Fisher refresh.
        adv_source = phase if advantage_phase is None else advantage_phase
        adv_key = tuple(
            self.strategy._per_component_phases(adv_source, len(error_vec)))

        phase_shift = (self._prev_phase is not None
                       and phase_key != self._prev_phase)
        self._prev_phase = phase_key

        advantage_vec = self.strategy.compute_advantage(error_vec, phase=adv_key)
        adv_scalar = float(np.mean(advantage_vec))

        apply_updates = (self.strategy.steps + 1) > self.strategy.warmup
        frozen_attrs = getattr(self.registry, "frozen_attrs", None)
        frozen = {name: (frozen_attrs(name) if frozen_attrs is not None
                         else frozenset())
                  for name in self.params}

        springs_fn = getattr(self.registry, "springs", None)
        has_springs = springs_fn is not None and bool(springs_fn())
        spring = {}
        if has_springs and apply_updates:
            means = {
                name: float(self._snap.get(
                    name, (param.mean, param.sigma, param.mean))[0])
                for name, param in self.params.items()
            }
            sigmas = {
                name: float(self._snap.get(
                    name, (param.mean, param.sigma, param.mean))[1])
                for name, param in self.params.items()
            }
            spring = self.registry.spring_state(means, sigmas, lr_t)

        # Sampling-time snapshot channels (spec §4): score units for the trace.
        sites: Dict[str, tuple] = {}
        for name, param in self.params.items():
            mu_at, sig_at, xhat = self._snap.get(
                name, (float(param.mean), float(param.sigma),
                       float(self.current_actions.get(name, param.mean))))
            d_log_sigma = 0.0
            if apply_updates and "sigma" not in frozen[name]:
                d_log_sigma = param.sigma_step_applied(
                    sample=xhat, mu_at_sample=mu_at, sigma_at_sample=sig_at,
                    advantage=adv_scalar, lr_sigma=self.lr_sigma)
            sites[name] = (float(xhat) - float(mu_at), float(sig_at), float(d_log_sigma))

        # Producer deltas: fresh since the previous step; a step without new
        # records contributes 0.0, never a stale value. The second element is
        # the producer's current value at registration (0.0 = no influence);
        # no estimation and no warm-up gate.
        noise: Dict[str, tuple] = {}
        for rec_name, rec in self.registry.producer_recorders().items():
            delta = rec.take_delta()
            noise[rec_name] = (0.0 if delta is None else float(delta),
                               0.0 if rec.scale is None else float(rec.scale))

        self.strategy.lr_mu = lr_t
        delta_means = self.strategy.step(
            sites, noise, advantage_vec, phase_shift=phase_shift)
        if self.lr_schedule is not None:
            self.history["lr_mu"].append(lr_t)
        if self.homeostasis is not None:
            self.history["homeostasis"].append(self.homeostasis.diagnostics())
        if self.game is not None:
            self.history["game"].append(self.game.diagnostics())

        # Apply to the host (from warmup+1): means first, then sigma — both from
        # the sampling-time snapshot, so neither depends on the other's change.
        for name, param in self.params.items():
            mu_at, sig_at, xhat = self._snap.get(
                name, (float(param.mean), float(param.sigma),
                       float(self.current_actions.get(name, param.mean))))
            if (apply_updates and name in delta_means
                    and "mean" not in frozen[name]):
                applied = (delta_means[name]
                           + spring.get(name, {}).get("force", 0.0))
                param.update_mean(applied)
                delta_means[name] = applied
            if apply_updates and "sigma" not in frozen[name]:
                param.update_sigma(sample=xhat, mu_at_sample=mu_at,
                                   sigma_at_sample=sig_at, advantage=adv_scalar,
                                   lr_sigma=self.lr_sigma)

        # A frozen mean applies no displacement this step.
        for name in delta_means:
            if "mean" in frozen.get(name, ()):
                delta_means[name] = 0.0

        if has_springs:
            self.history.setdefault("spring", []).append(spring)

        # Logging
        self.history["error_vector"].append(np.asarray(error_vec, dtype=np.float64))
        self.history["loss"].append(loss)
        self.history["advantage_vector"].append(np.asarray(advantage_vec, dtype=np.float64))
        self.history["phase"].append(phase_record)
        self.history["influence"].append(self.strategy.influence())
        for name, param in self.params.items():
            self.history["means"].setdefault(name, []).append(float(param.mean))
            self.history["sigmas"].setdefault(name, []).append(float(param.sigma))

        return {
            "step": self.current_step,
            "error_vector": np.asarray(error_vec, dtype=np.float64),
            "loss": loss,
            "advantage_vector": np.asarray(advantage_vec, dtype=np.float64),
            "phase": phase_record,
            "influence": self.strategy.influence(),
            "delta_means": dict(delta_means),
        }
