"""Multi-replica PERSONA agent.

One shared policy (mean/sigma) drives N independent replicas.  Each replica
owns its own strategy (F/X̄/G) and accumulates from its own sample every dt.
At reference phase boundaries the strategies are merged with
advantage-dependent weights, one Δμ/Δlogσ is applied to the shared policy,
and the merged state is copied back into every replica.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from persona.game import GameConfig, StackGame
from persona.homeostasis import HomeostaticConfig, HomeostaticRegulator, PhysicalMap
from persona.parameter import PersonaParam
from persona.sampling import ReplicaSampler
from persona.schedule import CosineLR
from persona.strategy import FisherMatrix


class MultiAgent:
    def __init__(self,
                 registries,
                 strategy: FisherMatrix | None = None,
                 lr_mu: float = 0.02,
                 lr_sigma: float = 0.005,
                 tau_mem_s: float = 2.0,
                 dt_ms: float = 10.0,
                 warmup: int = 3,
                 smooth_kernel_size: int = 5,
                 homeostasis: HomeostaticConfig | None = None,
                 game: GameConfig | None = None,
                 lr_schedule: CosineLR | None = None):
        self.registries = list(registries)
        if not self.registries:
            raise ValueError("MultiAgent needs at least one registry")
        self.policy = self.registries[0]
        self.n_replicas = len(self.registries)
        self.lr_mu = float(lr_mu)
        self.lr_sigma = float(lr_sigma)
        self.lr_schedule = lr_schedule
        self.dt_ms = float(dt_ms)
        self.warmup = int(warmup)
        self.smooth_kernel_size = int(smooth_kernel_size)
        self.beta = float(np.clip(
            1.0 - (self.dt_ms / 1000.0) / max(float(tau_mem_s), 1e-6),
            0.0, 0.999999))

        self.params: dict[str, PersonaParam] = {}
        self.history: dict[str, Any] = {
            "error_vector": [],
            "loss": [],
            "advantage_vector": [],
            "phase": [],
            "influence": [],
            "means": {},
            "sigmas": {},
        }
        self.current_actions: list[dict[str, float]] = []
        self.current_step = 0
        self._snapshots: list[dict[str, tuple]] = []
        self._last_sites: list[dict[str, tuple]] = []
        self._last_advantages: list[np.ndarray] = []
        self._prev_ref_phase = None
        self.replica_adv = np.zeros(self.n_replicas, dtype=np.float64)
        self._phase_adv_sum = np.zeros(self.n_replicas, dtype=np.float64)
        self._phase_adv_count = 0
        self._last_merge_weights = np.full(
            self.n_replicas, 1.0 / self.n_replicas)
        self.sampler = ReplicaSampler(self.n_replicas)
        self.output_buffers: list[list[np.ndarray]] = [
            [] for _ in range(self.n_replicas)]
        self.target_buffers: list[list[np.ndarray]] = [
            [] for _ in range(self.n_replicas)]

        self.params = self.policy.get_all_optimizable_params()
        if lr_schedule is not None:
            self.history["lr_mu"] = []

        self.homeostasis = None
        if homeostasis is not None and homeostasis.enabled:
            physical_map = PhysicalMap.from_registry(
                self.policy, eps_H=homeostasis.eps_H)
            self.homeostasis = HomeostaticRegulator(
                physical_map, homeostasis, dt_s=self.dt_ms / 1000.0,
                beta=self.beta, site_names=list(self.params.keys()))
            self.history["homeostasis"] = []

        self.game = None
        if game is not None and game.enabled:
            game_map = PhysicalMap.from_registry(self.policy)
            self.game = StackGame(
                game_map, game, dt_s=self.dt_ms / 1000.0,
                beta=self.beta, site_names=list(self.params.keys()))
            self.history["game"] = []

        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = FisherMatrix(
                param_names=list(self.params.keys()),
                noise_names=sorted(self.policy.producer_recorders()),
                lr_mu=self.lr_mu,
                lr_sigma=self.lr_sigma,
                beta=self.beta,
                warmup=self.warmup,
                homeostasis=self.homeostasis,
                game=self.game,
                dt_s=self.dt_ms / 1000.0,
            )
        self.strategy.homeostasis = self.homeostasis
        self.strategy.game = self.game
        self.strategy.dt_s = self.dt_ms / 1000.0

        # Replica-local strategy state. These never apply a policy update;
        # they only accumulate F/X̄/G until the next phase merge.
        self.replica_strategies = [
            FisherMatrix(
                param_names=list(self.params.keys()),
                noise_names=sorted(self.policy.producer_recorders()),
                lr_mu=self.lr_mu,
                lr_sigma=self.lr_sigma,
                beta=self.beta,
                warmup=self.warmup,
                dt_s=self.dt_ms / 1000.0,
            )
            for _ in range(self.n_replicas)
        ]

    # --- policy sharing -----------------------------------------------------

    @property
    def current_lr_mu(self) -> float:
        if self.lr_schedule is None:
            return self.lr_mu
        return self.lr_schedule.at(self.current_step)

    def refresh_params(self) -> dict[str, PersonaParam]:
        params = self.policy.get_all_optimizable_params()
        known = set(self.params)
        if known:
            missing = known - set(params)
            if missing:
                raise RuntimeError(
                    f"PERSONA sites disappeared from the policy registry: "
                    f"{sorted(missing)}")
        for name in params:
            self.history["means"].setdefault(name, [])
        self.params = params
        return params

    def sync_policy(self) -> None:
        """Copy the policy's mean/sigma into every replica's parameters."""
        params = self.refresh_params()
        for registry in self.registries[1:]:
            replica = registry.get_all_optimizable_params()
            missing = set(params) - set(replica)
            if missing:
                raise RuntimeError(
                    f"replica is missing PERSONA sites: {sorted(missing)}")
            for name, param in params.items():
                target = replica[name]
                target.mean = float(param.mean)
                target.sigma = float(param.sigma)

    # --- sampling -----------------------------------------------------------

    def sample_actions(self, t_ms: float | None = None):
        self.sync_policy()
        sampled = self.sampler.sample(self.refresh_params())
        self.current_actions = [actions for actions, _ in sampled]
        self._snapshots = [snapshot for _, snapshot in sampled]
        return self.current_actions

    def apply_actions(self, actions) -> None:
        for registry, replica_actions in zip(self.registries, actions):
            registry.apply_explored_parameters(replica_actions)

    # --- per-step update ----------------------------------------------------

    def _smooth(self, signal: np.ndarray, buffer: list[np.ndarray]) -> np.ndarray:
        arr = np.asarray(signal, dtype=np.float64)
        buffer.append(arr)
        if len(buffer) > self.smooth_kernel_size:
            buffer.pop(0)
        return np.mean(np.array(buffer), axis=0)

    def _replica_losses(self, outputs, targets):
        errors = []
        for i, (output, target) in enumerate(zip(outputs, targets)):
            smoothed_out = self._smooth(output, self.output_buffers[i])
            smoothed_tgt = self._smooth(target, self.target_buffers[i])
            errors.append(np.abs(np.asarray(smoothed_tgt)
                                 - np.asarray(smoothed_out)))
        return errors

    def _shared_advantages(self, errors, ref_phase):
        phases = self.strategy._per_component_phases(ref_phase, len(errors[0]))
        baseline = np.asarray([
            self.strategy.baselines.get_baseline(i, phases[i])
            for i in range(len(phases))
        ], dtype=np.float64)
        advantages = [baseline - np.asarray(err) for err in errors]
        mean_error = np.mean(np.asarray(errors), axis=0)
        for i in range(len(phases)):
            self.strategy.baselines.update(i, phases[i], float(mean_error[i]))
        return advantages

    def _merge_weights(self, advantage=None) -> np.ndarray:
        advantage = (np.asarray(self.replica_adv, dtype=np.float64)
                     if advantage is None else
                     np.asarray(advantage, dtype=np.float64))
        spread = max(float(np.std(advantage)), 1e-6)
        z = (advantage - float(np.max(advantage))) / spread
        weights = np.exp(z)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full(self.n_replicas, 1.0 / self.n_replicas)
        return weights / total

    def _merge(self, sites_ref, advantages, weights_adv, lr_t: float,
               frozen: dict[str, frozenset], apply_updates: bool
               ) -> dict[str, float]:
        weights = self._merge_weights(weights_adv)
        self._last_merge_weights = weights
        self.refresh_params()
        self.strategy.ensure_names(
            list(self.params.keys()),
            sorted(self.policy.producer_recorders()),
        )
        self.strategy.merge(self.replica_strategies, weights)
        weighted_A = np.average(np.asarray(advantages), axis=0, weights=weights)
        delta_means = self.strategy.deltas(sites_ref, weighted_A)

        params = self.refresh_params()
        means = {name: float(param.mean) for name, param in params.items()}
        sigmas = {name: float(param.sigma) for name, param in params.items()}
        spring = {}
        if apply_updates and self.policy.springs():
            spring = self.policy.spring_state(means, sigmas, lr_t)

        for name, param in params.items():
            if apply_updates and name in delta_means \
                    and "mean" not in frozen.get(name, frozenset()):
                applied = delta_means[name] + spring.get(
                    name, {}).get("force", 0.0)
                param.update_mean(applied)
                delta_means[name] = applied
            if apply_updates and "sigma" not in frozen.get(name, frozenset()):
                snapshots = [self._snapshots[i][name] for i in range(self.n_replicas)]
                mu_at = snapshots[0][0]
                sigma_at = snapshots[0][1]
                samples = [snap[2] for snap in snapshots]
                advantages_scalar = np.asarray(weights_adv).tolist()
                param.update_sigma_batch(
                    samples, mu_at, sigma_at, advantages_scalar,
                    lr_sigma=self.lr_sigma, weights=weights)

        for name in list(delta_means):
            if "mean" in frozen.get(name, frozenset()):
                delta_means[name] = 0.0

        # The next phase starts from the merged state.
        for replica in self.replica_strategies:
            replica.F = self.strategy.F.copy()
            replica.Xbar = self.strategy.Xbar.copy()
            replica.G = self.strategy.G.copy()
            replica.steps = self.strategy.steps
            replica.noise_scales = dict(self.strategy.noise_scales)
            replica._P_valid = False

        self.sync_policy()
        return delta_means

    def step(self, outputs, targets, ref_phases, model_phases=None,
             t_ms: float | None = None) -> dict[str, Any]:
        """Advance the shared policy from all replica observations.

        ``outputs`` and ``targets`` are per-replica arrays; ``ref_phases`` is
        the shared reference phase sequence (one entry per replica, normally
        identical). ``model_phases`` is only used for logging.
        """
        if model_phases is None:
            model_phases = ref_phases
        outputs = list(outputs)
        targets = list(targets)
        if len(outputs) != self.n_replicas or len(targets) != self.n_replicas:
            raise ValueError("outputs/targets must have one entry per replica")

        self.current_step += 1
        lr_t = self.current_lr_mu
        self.refresh_params()
        apply_updates = (self.strategy.steps + 1) > self.strategy.warmup
        frozen = {
            name: self.policy.frozen_attrs(name)
            for name in self.params
        }

        errors = self._replica_losses(outputs, targets)
        losses = [float(np.mean(err)) for err in errors]
        ref_phase = ref_phases[0]
        advantages = self._shared_advantages(errors, ref_phase)
        current_adv = np.asarray(
            [float(np.mean(A)) for A in advantages], dtype=np.float64)
        # Weight the merge by the advantage accumulated *during the phase*
        # that is ending, not by the first step of the new phase.
        if self._phase_adv_count > 0:
            weights_adv = self._phase_adv_sum / self._phase_adv_count
        else:
            weights_adv = current_adv
        self._phase_adv_sum += current_adv
        self._phase_adv_count += 1
        self.replica_adv = weights_adv.copy()

        sites_list = []
        noise_list = []
        for i, registry in enumerate(self.registries):
            sites = {}
            for name in self.params:
                mu, sigma, xhat = self._snapshots[i][name]
                dlog = 0.0
                if apply_updates and "sigma" not in frozen.get(name, frozenset()):
                    dlog = self.params[name].sigma_step(
                        xhat, mu, sigma, float(np.mean(advantages[i])),
                        self.lr_sigma)
                sites[name] = (float(xhat) - float(mu), float(sigma), float(dlog))
            sites_list.append(sites)

            noise = {}
            for rec_name, rec in registry.producer_recorders().items():
                delta = rec.take_delta()
                noise[rec_name] = (
                    0.0 if delta is None else float(delta),
                    0.0 if rec.scale is None else float(rec.scale),
                )
            noise_list.append(noise)

            self.replica_strategies[i].step(
                sites, noise, advantages[i], accumulate_only=True)

        self._last_sites = sites_list
        self._last_advantages = advantages

        ref_key = (tuple(ref_phase) if not isinstance(ref_phase, str)
                   else (ref_phase,))
        merge = self._prev_ref_phase is not None and ref_key != self._prev_ref_phase
        self._prev_ref_phase = ref_key
        if merge:
            delta_means = self._merge(
                sites_list[0], advantages, weights_adv, lr_t, frozen,
                apply_updates)
            # The current step is the first sample of the new phase.
            self._phase_adv_sum = current_adv.copy()
            self._phase_adv_count = 1
        else:
            delta_means = {name: 0.0 for name in self.params}

        mean_error = np.mean(np.asarray(errors), axis=0)
        mean_advantage = np.mean(np.asarray(advantages), axis=0)
        self.history["error_vector"].append(mean_error)
        self.history["loss"].append(float(np.mean(losses)))
        self.history["advantage_vector"].append(mean_advantage)
        self.history["phase"].append(model_phases[0])
        self.history["influence"].append(self.strategy.influence())
        if self.lr_schedule is not None:
            self.history["lr_mu"].append(lr_t)
        if self.homeostasis is not None:
            self.history["homeostasis"].append(self.homeostasis.diagnostics())
        if self.game is not None:
            self.history["game"].append(self.game.diagnostics())
        for name, param in self.params.items():
            self.history["means"][name].append(float(param.mean))
            self.history["sigmas"].setdefault(name, []).append(float(param.sigma))

        return {
            "step": self.current_step,
            "error_vector": mean_error,
            "loss": float(np.mean(losses)),
            "advantage_vector": mean_advantage,
            "phase": model_phases[0],
            "influence": self.strategy.influence(),
            "delta_means": delta_means,
            "losses": losses,
            "advantages": [A.tolist() for A in advantages],
            "merged": bool(merge),
            "merge_weights": self._last_merge_weights.tolist(),
        }

    def close(self) -> None:
        self.sampler.close()
