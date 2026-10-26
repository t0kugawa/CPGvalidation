from typing import Dict, List, Any
import numpy as np

from persona.registry import Registry
from persona.strategy import FisherMatrix
from persona.parameter import PersonaParam
from persona.noise_recorder import select_producer_names
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
                 sampling_threads: int = 1,
                 noise_config: str = "all",
                 trace_strategy: bool = False,
                 trace_float32: bool = False,
                 snapshot_strategy: int = 0,
                 player_map: Any = None):
        self.registry = registry
        self.noise_config = str(noise_config)
        self.trace_strategy = bool(trace_strategy)
        self.trace_float32 = bool(trace_float32)
        self.snapshot_strategy = max(0, int(snapshot_strategy))
        self.player_map = player_map
        self._trace_site_names: tuple[str, ...] | None = None
        if (player_map is not None and homeostasis is not None
                and homeostasis.enabled):
            raise ValueError("Evo and H are mutually exclusive")
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
            if player_map is not None:
                from persona.segments import SegmentMatrix
                matrix = SegmentMatrix.from_registry(registry)
                game_map = player_map.physical_map(matrix)
            else:
                game_map = PhysicalMap.from_registry(registry)
            self.game = StackGame(
                game_map, game, dt_s=self.dt_ms / 1000.0,
                beta=self.beta, site_names=list(self.params.keys()))
            self.history["game"] = []

        if strategy is not None:
            if self.noise_config != "all":
                raise ValueError(
                    "noise_config cannot be applied to a supplied strategy")
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
            producer_names = select_producer_names(
                sorted(self.registry.producer_recorders()), self.noise_config)
            self.strategy = FisherMatrix(
                param_names=list(self.params.keys()),
                noise_names=producer_names,
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
        if self.trace_strategy:
            names = tuple(params)
            if self._trace_site_names is None:
                self._trace_site_names = names
            elif names != self._trace_site_names:
                added = sorted(set(names) - set(self._trace_site_names))
                removed = sorted(set(self._trace_site_names) - set(names))
                raise RuntimeError(
                    "strategy trace requires a fixed site set; changed sites: "
                    f"added={added} removed={removed}")
        strategy = getattr(self, "strategy", None)
        if strategy is not None:
            selected = select_producer_names(
                sorted(self.registry.producer_recorders()), self.noise_config)
            strategy.ensure_names(list(self.params.keys()), selected)
        if getattr(self, "homeostasis", None) is not None:
            self.homeostasis.ensure_params(params.keys())
        if getattr(self, "game", None) is not None:
            self.game.ensure_params(params.keys())
        return params

    def _sync_trace_noise_names(self) -> None:
        """Extend earlier trace rows when a producer appears mid-run."""
        trace = self.history.get("strategy_trace")
        if trace is None:
            return
        current = list(self.strategy.noise_names)
        known = list(trace["noise_names"])
        if current == known:
            return
        if len(current) < len(known) or current[:len(known)] != known:
            raise RuntimeError(
                "strategy trace producer set changed by removal or reorder: "
                f"known={known} current={current}")
        pad = len(current) - len(known)
        for key in ("noise_delta", "noise_scale"):
            for i, row in enumerate(trace[key]):
                trace[key][i] = np.concatenate(
                    [row, np.zeros(pad, dtype=row.dtype)])
        trace["noise_names"] = current

    def _record_strategy_trace(self, sites: Dict[str, tuple],
                               noise: Dict[str, tuple],
                               advantage: np.ndarray) -> None:
        if not self.trace_strategy:
            return
        trace = self.history.get("strategy_trace")
        dtype = "float32" if self.trace_float32 else "float64"
        if trace is None:
            trace = {
                "dtheta": [], "sigma": [], "dlogsigma": [],
                "advantage": [], "noise_delta": [], "noise_scale": [],
                "param_names": list(self.strategy.param_names),
                "noise_names": list(self.strategy.noise_names),
                "beta": float(self.strategy.beta),
                "lambda_reg": float(self.strategy.lambda_reg),
                "warmup": int(self.strategy.warmup),
                "refresh_every": int(self.strategy.refresh_every),
                "lr_mu": float(self.strategy.lr_mu),
                "lr_sigma": float(self.lr_sigma),
                "dt_s": float(self.strategy.dt_s),
                "dtype": dtype,
            }
            self.history["strategy_trace"] = trace
        self._sync_trace_noise_names()
        names = list(trace["param_names"])
        noise_names = list(trace["noise_names"])
        trace["dtheta"].append(np.asarray(
            [sites[name][0] for name in names], dtype=dtype))
        trace["sigma"].append(np.asarray(
            [sites[name][1] for name in names], dtype=dtype))
        trace["dlogsigma"].append(np.asarray(
            [sites[name][2] for name in names], dtype=dtype))
        trace["advantage"].append(np.asarray(advantage, dtype=dtype))
        trace["noise_delta"].append(np.asarray(
            [noise.get(name, (0.0, 0.0))[0] for name in noise_names],
            dtype=dtype))
        trace["noise_scale"].append(np.asarray(
            [noise.get(name, (0.0, 0.0))[1] for name in noise_names],
            dtype=dtype))
        trace["lr_mu"] = float(self.strategy.lr_mu)

    def strategy_trace_arrays(self) -> Dict[str, Any] | None:
        """Return the trace with list rows converted to arrays."""
        trace = self.history.get("strategy_trace")
        if trace is None:
            return None
        from persona.strategy_trace import trace_arrays
        return trace_arrays(trace)

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
        allowed = set(self.strategy.noise_names)
        for rec_name, rec in self.registry.producer_recorders().items():
            if rec_name not in allowed:
                continue
            delta = rec.take_delta()
            noise[rec_name] = (0.0 if delta is None else float(delta),
                               0.0 if rec.scale is None else float(rec.scale))

        self._record_strategy_trace(sites, noise, advantage_vec)
        self.strategy.lr_mu = lr_t
        delta_means = self.strategy.step(
            sites, noise, advantage_vec, phase_shift=phase_shift)
        if (self.snapshot_strategy > 0
                and self.current_step % self.snapshot_strategy == 0):
            self.history.setdefault("strategy_checkpoints", []).append(
                self.strategy.snapshot())
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
