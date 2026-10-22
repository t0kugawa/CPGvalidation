"""Conserved-stack game for PERSONA's natural-gradient strategy.

The stack ``s_i`` is a player's share of the total advantage (``Σ s_i = 1``).
The game throttles each player's natural-gradient proposal from its stack and
transfers share between players from the first-order contribution ``g^T Δ``.
The transfer is zero-sum by construction; the optional common pot debits a
Fisher-priced movement fee and redistributes it. No setpoints, target bands, or
stability clamps are involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from persona.homeostasis import PhysicalMap


_THROTTLE_MODES = ("proportional", "inverse")
_POT_MODES = ("contribution", "equal")
_BASE_SEATS = ("none", "uniform", "uncertainty")


@dataclass
class GameConfig:
    """Configuration for the opt-in conserved-stack game."""

    enabled: bool = False
    throttle_mode: str = "inverse"
    transfer_rate: float = 0.05
    movement_fee: float = 0.0
    pot_mode: str = "contribution"
    base_seat: str = "none"
    liveness_fraction: float = 0.0
    throttle_exponent: float = 1.0
    burnin_steps: int = 100
    share_floor: float = 1e-12
    eps: float = 1e-12

    def __post_init__(self):
        mode = str(self.throttle_mode)
        if mode not in _THROTTLE_MODES:
            raise ValueError(
                f"throttle_mode must be one of {_THROTTLE_MODES}, got {mode!r}")
        self.throttle_mode = mode
        pot = str(self.pot_mode)
        if pot not in _POT_MODES:
            raise ValueError(
                f"pot_mode must be one of {_POT_MODES}, got {pot!r}")
        self.pot_mode = pot
        seat = str(self.base_seat)
        if seat not in _BASE_SEATS:
            raise ValueError(
                f"base_seat must be one of {_BASE_SEATS}, got {seat!r}")
        self.base_seat = seat
        self.liveness_fraction = float(self.liveness_fraction)
        if (not np.isfinite(self.liveness_fraction)
                or self.liveness_fraction < 0.0
                or self.liveness_fraction > 1.0):
            raise ValueError("liveness_fraction must be in [0, 1]")
        self.throttle_exponent = float(self.throttle_exponent)
        if (not np.isfinite(self.throttle_exponent)
                or self.throttle_exponent <= 0.0):
            raise ValueError("throttle_exponent must be finite and positive")
        for name in ("transfer_rate", "movement_fee", "share_floor", "eps"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.transfer_rate < 0.0:
            raise ValueError("transfer_rate must be non-negative")
        if self.movement_fee < 0.0:
            raise ValueError("movement_fee must be non-negative")
        if self.share_floor <= 0.0:
            raise ValueError("share_floor must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        self.burnin_steps = int(self.burnin_steps)
        if self.burnin_steps < 0:
            raise ValueError("burnin_steps must be non-negative")

    @classmethod
    def no_consume(cls, throttle_mode: str = "inverse", **kwargs) -> "GameConfig":
        """The stack moves only by outcome transfer; movement is free."""
        return cls(enabled=True, throttle_mode=throttle_mode,
                   movement_fee=0.0, **kwargs)

    @classmethod
    def pot(cls, throttle_mode: str = "inverse",
            movement_fee: float = 0.05, **kwargs) -> "GameConfig":
        """Movement pays a Fisher-priced fee that is redistributed."""
        return cls(enabled=True, throttle_mode=throttle_mode,
                   movement_fee=float(movement_fee), **kwargs)

    @classmethod
    def current_best(cls) -> "GameConfig":
        """Selected experimental configuration (2026-10-22).

        Uncertainty base seat, proportional ranking, liveness 0.05, curvature
        2.5, transfer 0.05, no pot. This is the a25 arm from the tuning grid;
        it is opt-in and not a production default.
        """
        return cls(
            enabled=True,
            throttle_mode="proportional",
            base_seat="uncertainty",
            liveness_fraction=0.05,
            throttle_exponent=2.5,
            transfer_rate=0.05,
            movement_fee=0.0,
            burnin_steps=100,
        )


class StackGame:
    """Player-level stack allocation over a :class:`PhysicalMap`.

    ``apply`` is the strategy-facing API: it throttles the per-site proposal,
    updates the stacks from the first-order contribution, and returns the
    throttled proposal. ``throttle`` and ``diagnostics`` expose the state.
    """

    def __init__(self, physical_map: PhysicalMap,
                 config: GameConfig | None = None, *,
                 dt_s: float = 0.01, beta: float = 0.995,
                 site_names=None):
        self.physical_map = physical_map
        self.config = config if config is not None else GameConfig(enabled=True)
        self.dt_s = float(dt_s)
        self.beta = float(beta)
        if site_names is None:
            site_names = [site for spec in physical_map.specs
                          for site in spec.param_names]
        self._site_names = [str(site) for site in site_names]
        self.physical_map.ensure_names(self._site_names)
        self.steps = 0
        self.stacks = self._uniform(self.physical_map.n_players)
        self.sigma_ref: dict[str, float] = {}
        self._last_sigmas: np.ndarray | None = None
        self.last: dict = {}
        self._player_sites: list[list[int]] = []
        self._rebuild_site_map()
        self._set_last("burnin")

    # --- lifecycle -----------------------------------------------------------

    @staticmethod
    def _uniform(n: int) -> np.ndarray:
        return (np.full(int(n), 1.0 / float(n), dtype=np.float64)
                if n > 0 else np.zeros(0, dtype=np.float64))

    def _rebuild_site_map(self) -> None:
        n = self.physical_map.n_players
        self._player_sites = [[] for _ in range(n)]
        for i, site in enumerate(self._site_names):
            p = self.physical_map.player_of.get(site)
            if p is not None:
                self._player_sites[int(p)].append(i)

    def ensure_params(self, site_names) -> None:
        """Append new sites as singleton players without losing stack history."""
        known = set(self._site_names)
        added = False
        for site in site_names:
            name = str(site)
            if name not in known:
                self._site_names.append(name)
                known.add(name)
                added = True
        if not added and self.stacks.size == self.physical_map.n_players:
            return
        n_old = self.physical_map.n_players
        self.physical_map.ensure_names(self._site_names)
        n_new = self.physical_map.n_players
        if n_new != self.stacks.size:
            stacks = self._uniform(n_new)
            if n_old > 0 and n_new > n_old:
                stacks[:n_old] = (self.stacks[:n_old]
                                  * (float(n_old) / float(n_new)))
            self.stacks = stacks
        self._rebuild_site_map()

    # --- game state ----------------------------------------------------------

    @property
    def site_names(self) -> list[str]:
        return list(self._site_names)

    @property
    def n_players(self) -> int:
        return self.physical_map.n_players

    def throttle(self) -> np.ndarray:
        """Per-player movement entitlement, relative and in ``[0, 1]``.

        The contestable part is neutral (all ones) at uniform stacks.
        ``proportional`` gives it to the richest player; ``inverse`` gives it
        to the poorest. ``liveness_fraction`` blends in a physical base seat so
        that a zero-stack player keeps a right to move. The entitlement is not
        a fee: unused movement is not transferred.
        """
        n = self.stacks.size
        if n <= 1:
            return np.ones(n, dtype=np.float64)
        s = np.asarray(self.stacks, dtype=np.float64)
        if self.config.throttle_mode == "proportional":
            raw = s
        else:
            raw = 1.0 - s
        denom = float(np.max(raw))
        if not np.isfinite(denom) or denom <= self.config.eps:
            contest = np.ones(n, dtype=np.float64)
        else:
            contest = np.clip(raw / denom, 0.0, 1.0)
        contest = contest ** float(self.config.throttle_exponent)
        eps = float(self.config.liveness_fraction)
        if eps <= 0.0 or self.config.base_seat == "none":
            return contest
        base = self._base_seat()
        return np.clip((1.0 - eps) * contest + eps * base, 0.0, 1.0)

    def _base_seat(self) -> np.ndarray:
        """Physical movement right, normalized so the largest seat is 1.

        ``uniform`` gives every player the same seat. ``uncertainty`` uses the
        player's RMS sigma ratio to its first observed sigma: an uncertain
        player keeps more base movement, a confident one keeps less. No sigma
        history yet means a uniform seat.
        """
        n = self.stacks.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        base = np.ones(n, dtype=np.float64)
        if self.config.base_seat == "uncertainty":
            sig = self._last_sigmas
            if sig is not None and sig.size == len(self._site_names):
                base = np.zeros(n, dtype=np.float64)
                for p, idx in enumerate(self._player_sites):
                    if not idx:
                        continue
                    ratios = [
                        max(float(sig[i]), self.config.eps)
                        / max(float(self.sigma_ref.get(
                            self._site_names[i], self.config.eps)),
                              self.config.eps)
                        for i in idx
                    ]
                    base[p] = float(np.sqrt(np.mean(np.square(ratios))))
        denom = float(np.max(base))
        if not np.isfinite(denom) or denom <= self.config.eps:
            return np.ones(n, dtype=np.float64)
        return np.clip(base / denom, 0.0, 1.0)

    def apply(self, g_mu, delta_mu, steps: int, *, F=None,
              sigmas=None) -> np.ndarray:
        """Throttle the proposal, transfer share, and return the applied move.

        ``g_mu`` and ``delta_mu`` are per-site (mean-coordinate) vectors in the
        same order as ``site_names``. ``F`` is the strategy Fisher matrix; only
        each player's diagonal block is used, for the pot arm's Fisher length.
        ``sigmas`` are the per-site sampling widths, used by the uncertainty
        base seat; the first value seen per site becomes its reference.
        """
        g = np.asarray(g_mu, dtype=np.float64).ravel()
        d = np.asarray(delta_mu, dtype=np.float64).ravel()
        m = len(self._site_names)
        if g.size != m or d.size != m:
            raise ValueError(
                "g_mu and delta_mu must match the game's site order")
        if not np.all(np.isfinite(g)):
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(d)):
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        if sigmas is not None:
            sig = np.asarray(sigmas, dtype=np.float64).ravel()
            if sig.size != m:
                raise ValueError(
                    "sigmas must match the game's site order")
            sig = np.maximum(
                np.nan_to_num(sig, nan=self.config.eps,
                              posinf=self.config.eps,
                              neginf=self.config.eps),
                self.config.eps)
            for i, value in enumerate(sig):
                self.sigma_ref.setdefault(self._site_names[i], float(value))
            self._last_sigmas = sig

        self.steps = int(steps)
        n = self.stacks.size
        tau = self.throttle()
        d_out = d.copy()
        contrib = np.zeros(n, dtype=np.float64)
        fisher_len = np.zeros(n, dtype=np.float64)
        f_arr = None if F is None else np.asarray(F, dtype=np.float64)

        for p, idx in enumerate(self._player_sites):
            if not idx:
                continue
            d_p = d[idx] * tau[p]
            d_out[idx] = d_p
            contrib[p] = float(g[idx] @ d_p)
            if f_arr is not None and f_arr.ndim == 2:
                block = f_arr[np.ix_(idx, idx)]
                length = float(d_p @ block @ d_p)
                fisher_len[p] = float(np.sqrt(max(length, 0.0)))

        a_abs = float(np.abs(contrib).sum())
        a_hat = contrib / max(a_abs, self.config.eps)

        active = (self.steps > self.config.burnin_steps and n > 1)
        transfer = np.zeros(n, dtype=np.float64)
        if active and self.config.transfer_rate > 0.0:
            transfer = self.config.transfer_rate * (
                a_hat - self.stacks * float(a_hat.sum()))
        new = self.stacks + transfer

        fees = np.zeros(n, dtype=np.float64)
        pot = 0.0
        if active and self.config.movement_fee > 0.0:
            denom = float(fisher_len.sum())
            if denom > self.config.eps:
                fees = self.config.movement_fee * fisher_len / denom
                pot = float(fees.sum())
                new = new - fees
                if self.config.pot_mode == "equal":
                    weights = np.ones(n, dtype=np.float64)
                else:
                    weights = np.maximum(a_hat, 0.0)
                    if float(weights.sum()) <= self.config.eps:
                        weights = np.ones(n, dtype=np.float64)
                new = new + pot * weights / float(weights.sum())

        new = np.maximum(new, self.config.share_floor)
        self.stacks = new / float(new.sum())

        self._set_last(
            "active" if active else "burnin",
            contribution=contrib, a_hat=a_hat, transfer=transfer,
            fees=fees, pot=pot, transfer_sum=float(transfer.sum()),
            share_sum=float(self.stacks.sum()))
        return d_out

    # --- diagnostics ---------------------------------------------------------

    @staticmethod
    def _entropy(stacks) -> float:
        s = np.asarray(stacks, dtype=np.float64)
        s = s[s > 0.0]
        if s.size == 0:
            return 0.0
        return float(-np.sum(s * np.log(s)))

    def _set_last(self, stage: str, **kwargs) -> None:
        n = self.stacks.size
        throttle = self.throttle()
        entropy = self._entropy(self.stacks)
        out = {
            "stage": stage,
            "steps": int(self.steps),
            "mode": self.config.throttle_mode,
            "movement_fee": float(self.config.movement_fee),
            "pot_mode": self.config.pot_mode,
            "base_mode": self.config.base_seat,
            "liveness": float(self.config.liveness_fraction),
            "base_seat": self._base_seat().copy(),
            "stacks": self.stacks.copy(),
            "throttle": throttle.copy(),
            "contribution": np.zeros(n, dtype=np.float64),
            "a_hat": np.zeros(n, dtype=np.float64),
            "transfer": np.zeros(n, dtype=np.float64),
            "fees": np.zeros(n, dtype=np.float64),
            "pot": 0.0,
            "transfer_sum": 0.0,
            "share_sum": float(self.stacks.sum()),
            "entropy": entropy,
            "entropy_norm": (entropy / float(np.log(n)) if n > 1 else 0.0),
            "hhi": float(np.sum(self.stacks ** 2)) if n else 0.0,
            "stack_min": float(np.min(self.stacks)) if n else 0.0,
            "stack_max": float(np.max(self.stacks)) if n else 0.0,
            "throttle_min": float(np.min(throttle)) if n else 0.0,
            "throttle_max": float(np.max(throttle)) if n else 0.0,
        }
        for key, value in kwargs.items():
            out[key] = (value.copy() if isinstance(value, np.ndarray)
                        else value)
        self.last = out

    def diagnostics(self) -> dict:
        return {key: (value.copy() if isinstance(value, np.ndarray)
                      else value)
                for key, value in self.last.items()}
