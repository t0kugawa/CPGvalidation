"""Benchmark-only differentiable toy spiking CPG.

A small LIF half-centre (E/F rhythm generators, reciprocal inhibition, tonic
drive, two motor readouts) driving a 1-DOF ankle. It is deliberately separate
from the NEST CPG; it exists so PERSONA, natural-gradient variants, CMA-ES and
a surrogate gradient can optimize the same 14-parameter vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAM_NAMES = (
    "w_tonic_e", "w_tonic_f",
    "w_rec_e", "w_rec_f",
    "w_e_to_f", "w_f_to_e",
    "w_rg_to_mn_e", "w_rg_to_mn_f",
    "tau_m_e", "tau_m_f",
    "bias_e", "bias_f",
    "gain_e", "gain_f",
)

BOUNDS = {
    "w_tonic_e": (0.0, 5.0), "w_tonic_f": (0.0, 5.0),
    "w_rec_e": (0.0, 5.0), "w_rec_f": (0.0, 5.0),
    "w_e_to_f": (0.0, 5.0), "w_f_to_e": (0.0, 5.0),
    "w_rg_to_mn_e": (0.0, 5.0), "w_rg_to_mn_f": (0.0, 5.0),
    "tau_m_e": (5.0, 50.0), "tau_m_f": (5.0, 50.0),
    "bias_e": (-1.0, 1.0), "bias_f": (-1.0, 1.0),
    "gain_e": (0.0, 5.0), "gain_f": (0.0, 5.0),
}

INITIAL = {
    "w_tonic_e": 0.5, "w_tonic_f": 0.5,
    "w_rec_e": 0.2, "w_rec_f": 0.2,
    "w_e_to_f": 0.5, "w_f_to_e": 0.5,
    "w_rg_to_mn_e": 1.0, "w_rg_to_mn_f": 1.0,
    "tau_m_e": 20.0, "tau_m_f": 20.0,
    "bias_e": 0.0, "bias_f": 0.0,
    "gain_e": 1.0, "gain_f": 1.0,
}

# Plant constants (fixed; not optimized in v1).
_PLANT_I = 0.05
_PLANT_K = 1.0
_PLANT_C = 0.5
_TORQUE_SCALE = 1.0
_ADAPT_GAIN = 0.8
_ADAPT_TAU = 0.1
_CLOCK_HZ = 1.0
_CLOCK_GAIN = 1.5
_TARGET_AMPLITUDE = 0.15
_TARGET_FREQ_HZ = 1.0


@dataclass(frozen=True)
class ToyConfig:
    dt: float = 0.01
    steps: int = 100
    n_rg: int = 8
    noise_std: float = 0.0
    seed: int = 0


class ToyCPG:
    def __init__(self, config: ToyConfig = ToyConfig()):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.v_e = np.zeros(self.config.n_rg)
        self.v_f = np.zeros(self.config.n_rg)
        self.a_e = np.zeros(self.config.n_rg)
        self.a_f = np.zeros(self.config.n_rg)
        self.q = 0.0
        self.qd = 0.0
        self.last: dict = {}

    def _params(self, params: dict) -> dict:
        out = dict(INITIAL)
        out.update({name: float(params[name]) for name in PARAM_NAMES
                    if name in params})
        for name, (lo, hi) in BOUNDS.items():
            out[name] = float(np.clip(out[name], lo, hi))
        return out

    def _target(self, t: float) -> float:
        return _TARGET_AMPLITUDE * np.sin(
            2.0 * np.pi * _TARGET_FREQ_HZ * t)

    def step(self, params: dict) -> dict:
        p = self._params(params)
        dt = float(self.config.dt)
        tau = dt / max(p["tau_m_e"] / 1000.0, 1e-9)
        tau_f = dt / max(p["tau_m_f"] / 1000.0, 1e-9)
        # Rates are the previous step's spike fractions.
        rate_e = float(np.mean(self.last.get("spikes_e", np.zeros(
            self.config.n_rg))))
        rate_f = float(np.mean(self.last.get("spikes_f", np.zeros(
            self.config.n_rg))))
        clock_e = _CLOCK_GAIN * 0.5 * (
            1.0 + np.sin(2.0 * np.pi * _CLOCK_HZ * self.t * dt))
        clock_f = _CLOCK_GAIN * 0.5 * (
            1.0 - np.sin(2.0 * np.pi * _CLOCK_HZ * self.t * dt))
        input_e = (p["w_tonic_e"] + p["w_rec_e"] * rate_e + p["bias_e"]
                   - p["w_f_to_e"] * rate_f - _ADAPT_GAIN * self.a_e
                   + clock_e)
        input_f = (p["w_tonic_f"] + p["w_rec_f"] * rate_f + p["bias_f"]
                   - p["w_e_to_f"] * rate_e - _ADAPT_GAIN * self.a_f
                   + clock_f)
        self.v_e = self.v_e + tau * (-self.v_e + input_e)
        self.v_f = self.v_f + tau_f * (-self.v_f + input_f)
        spikes_e = self.v_e >= 1.0
        spikes_f = self.v_f >= 1.0
        self.v_e = np.where(spikes_e, 0.0, self.v_e)
        self.v_f = np.where(spikes_f, 0.0, self.v_f)
        rate_e = float(np.mean(spikes_e))
        rate_f = float(np.mean(spikes_f))
        adapt_step = dt / _ADAPT_TAU
        self.a_e = self.a_e + adapt_step * (-self.a_e + rate_e)
        self.a_f = self.a_f + adapt_step * (-self.a_f + rate_f)
        mn_e = p["w_rg_to_mn_e"] * rate_e
        mn_f = p["w_rg_to_mn_f"] * rate_f
        a_e = p["gain_e"] * mn_e
        a_f = p["gain_f"] * mn_f
        torque = _TORQUE_SCALE * (a_e - a_f)
        qdd = (torque - _PLANT_C * self.qd - _PLANT_K * self.q) / _PLANT_I
        self.qd += qdd * dt
        self.q += self.qd * dt
        self.t += 1
        target = self._target(self.t * dt)
        if self.config.noise_std:
            target += float(self.rng.normal(0.0, self.config.noise_std))
        err2 = (self.q - target) ** 2
        phase = "E" if rate_e >= rate_f else "F"
        self.last = {
            "spikes_e": spikes_e, "spikes_f": spikes_f,
            "rate_e": rate_e, "rate_f": rate_f,
            "q_model": float(self.q), "q_target": float(target),
            "torque": float(torque), "error2": float(err2),
            "phase": phase, "time": float(self.t * dt),
        }
        return self.last

    def rollout(self, params: dict, *, record: bool = False) -> dict:
        self.reset()
        rows = {key: [] for key in (
            "rate_e", "rate_f", "q_model", "q_target", "torque",
            "error2", "phase")}
        for _ in range(self.config.steps):
            row = self.step(params)
            for key in rows:
                rows[key].append(row[key])
        rate_e = np.asarray(rows["rate_e"])
        rate_f = np.asarray(rows["rate_f"])
        error2 = np.asarray(rows["error2"])
        torque = np.asarray(rows["torque"])
        mask_e = rate_e >= rate_f
        mask_f = ~mask_e
        overall = float(np.mean(error2))
        loss_e = float(np.mean(error2[mask_e])) if mask_e.any() else overall
        loss_f = float(np.mean(error2[mask_f])) if mask_f.any() else overall
        tail = slice(self.config.steps // 2, self.config.steps)
        tail_loss = float(np.mean(error2[tail]) + 0.01 * np.mean(
            torque[tail] ** 2))
        out = {
            "loss_vec": np.asarray([loss_e, loss_f]),
            "loss_total": float(np.mean([loss_e, loss_f])
                                + 0.01 * np.mean(torque ** 2)),
            "tail_loss": tail_loss,
            "q_model": np.asarray(rows["q_model"]),
            "q_target": np.asarray(rows["q_target"]),
            "rate_e": rate_e,
            "rate_f": rate_f,
            "tau": torque,
            "phase": tuple(rows["phase"]),
        }
        if record:
            out["error2"] = error2
        return out

    def held_out_loss(self, params: dict) -> float:
        evaluation = ToyCPG(ToyConfig(
            dt=self.config.dt, steps=self.config.steps,
            n_rg=self.config.n_rg, noise_std=0.0, seed=10_000))
        return float(evaluation.rollout(params)["tail_loss"])
