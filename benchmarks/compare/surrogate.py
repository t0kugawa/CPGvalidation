"""Surrogate-gradient BPTT for the toy spiking CPG.

The forward is a soft-spike relaxation of the toy LIF network (a differentiable
rate model); the backward is reverse-mode autodiff through that relaxation.
This is the standard surrogate-gradient training setup: the trained vector is
evaluated on the hard toy. A finite-difference check on the relaxation is
required by the test.
"""
from __future__ import annotations

import numpy as np

from benchmarks.compare.toy_cpg import (
    BOUNDS,
    INITIAL,
    PARAM_NAMES,
    ToyConfig,
    _ADAPT_GAIN,
    _ADAPT_TAU,
    _CLOCK_GAIN,
    _CLOCK_HZ,
    _PLANT_C,
    _PLANT_I,
    _PLANT_K,
    _TARGET_AMPLITUDE,
    _TARGET_FREQ_HZ,
    _TORQUE_SCALE,
)


def _unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=float)
    if shape == ():
        return float(np.sum(grad))
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for index, size in enumerate(shape):
        if size == 1 and grad.shape[index] != 1:
            grad = grad.sum(axis=index, keepdims=True)
    return grad


class Var:
    """Tiny reverse-mode tensor used only by this benchmark."""

    def __init__(self, value, parents=(), backward=None, requires_grad=False):
        self.value = np.asarray(value, dtype=float)
        self.grad = None
        self.parents = tuple(parents)
        self.backward_fn = backward
        self.requires_grad = bool(requires_grad)

    @property
    def shape(self):
        return self.value.shape

    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(self.value + other.value,
                   ((self, lambda g: _unbroadcast(g, self.shape)),
                    (other, lambda g: _unbroadcast(g, other.shape))))

    __radd__ = __add__

    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(self.value - other.value,
                   ((self, lambda g: _unbroadcast(g, self.shape)),
                    (other, lambda g: _unbroadcast(-g, other.shape))))

    def __rsub__(self, other):
        return Var(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(self.value * other.value,
                   ((self, lambda g: _unbroadcast(g * other.value, self.shape)),
                    (other, lambda g: _unbroadcast(g * self.value,
                                                   other.shape))))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return Var(self.value / other.value,
                   ((self, lambda g: _unbroadcast(g / other.value,
                                                  self.shape)),
                    (other, lambda g: _unbroadcast(
                        -g * self.value / (other.value ** 2), other.shape))))

    def __rtruediv__(self, other):
        return Var(other) / self

    def __pow__(self, power):
        if power != 2:
            raise NotImplementedError("only square is needed")
        return Var(self.value ** 2,
                   ((self, lambda g: _unbroadcast(
                       2.0 * self.value * g, self.shape)),))

    def __neg__(self):
        return Var(-self.value, ((self, lambda g: -g),))

    def __getitem__(self, index):
        def backward(grad):
            out = np.zeros_like(self.value)
            out[index] += grad
            return out
        return Var(self.value[index], ((self, backward),))

    def mean(self):
        size = self.value.size if self.value.size else 1
        return Var(float(np.mean(self.value)),
                   ((self, lambda g: _unbroadcast(
                       np.full(self.shape, float(g) / size), self.shape)),))

    def sin(self):
        return Var(np.sin(self.value),
                   ((self, lambda g: _unbroadcast(g * np.cos(self.value),
                                                  self.shape)),))

    def sigmoid(self):
        value = 1.0 / (1.0 + np.exp(-self.value))
        return Var(value,
                   ((self, lambda g: _unbroadcast(
                       g * value * (1.0 - value), self.shape)),))

    def backward(self):
        topo = []
        seen = set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent, _ in node.parents:
                visit(parent)
            topo.append(node)

        visit(self)
        self.grad = np.ones_like(self.value)
        for node in reversed(topo):
            if node.grad is None:
                continue
            for parent, fn in node.parents:
                contribution = np.asarray(fn(node.grad), dtype=float)
                parent.grad = (contribution if parent.grad is None
                               else parent.grad + contribution)


def _clip(params):
    return {name: float(np.clip(float(params[name]), *BOUNDS[name]))
            for name in PARAM_NAMES}


def _soft_graph(params: dict, config: ToyConfig, alpha: float):
    p = {name: Var(_clip(params)[name], requires_grad=True)
         for name in PARAM_NAMES}
    dt = float(config.dt)
    n = int(config.n_rg)
    v_e = Var(np.zeros(n))
    v_f = Var(np.zeros(n))
    a_e = Var(0.0)
    a_f = Var(0.0)
    q = Var(0.0)
    qd = Var(0.0)
    rate_e = Var(0.0)
    rate_f = Var(0.0)
    errors = []
    torques = []
    for step in range(int(config.steps)):
        phase = 2.0 * np.pi * _CLOCK_HZ * step * dt
        clock_e = _CLOCK_GAIN * 0.5 * (1.0 + np.sin(phase))
        clock_f = _CLOCK_GAIN * 0.5 * (1.0 - np.sin(phase))
        input_e = (p["w_tonic_e"] + p["w_rec_e"] * rate_e + p["bias_e"]
                   - p["w_f_to_e"] * rate_f - _ADAPT_GAIN * a_e + clock_e)
        input_f = (p["w_tonic_f"] + p["w_rec_f"] * rate_f + p["bias_f"]
                   - p["w_e_to_f"] * rate_e - _ADAPT_GAIN * a_f + clock_f)
        tau_e = dt / (p["tau_m_e"] / 1000.0)
        tau_f = dt / (p["tau_m_f"] / 1000.0)
        v_e = v_e + tau_e * (-v_e + input_e)
        v_f = v_f + tau_f * (-v_f + input_f)
        spike_e = (alpha * (v_e - 1.0)).sigmoid()
        spike_f = (alpha * (v_f - 1.0)).sigmoid()
        rate_e = spike_e.mean()
        rate_f = spike_f.mean()
        adapt_step = dt / _ADAPT_TAU
        a_e = a_e + adapt_step * (-a_e + rate_e)
        a_f = a_f + adapt_step * (-a_f + rate_f)
        mn_e = p["w_rg_to_mn_e"] * rate_e
        mn_f = p["w_rg_to_mn_f"] * rate_f
        torque = _TORQUE_SCALE * (p["gain_e"] * mn_e - p["gain_f"] * mn_f)
        qd = qd + (torque - _PLANT_C * qd - _PLANT_K * q) / _PLANT_I * dt
        q = q + qd * dt
        target = _TARGET_AMPLITUDE * np.sin(
            2.0 * np.pi * _TARGET_FREQ_HZ * (step + 1) * dt)
        errors.append((q - target) ** 2)
        torques.append(torque ** 2)
    tail = list(range(int(config.steps) // 2, int(config.steps)))
    loss = (sum(errors[index] for index in tail) / len(tail)
            + 0.01 * sum(torques[index] for index in tail) / len(tail))
    return loss, p


def surrogate_loss_and_grad(params: dict, config: ToyConfig,
                            alpha: float = 10.0) -> tuple[float, np.ndarray]:
    loss, tensors = _soft_graph(params, config, alpha)
    loss.backward()
    grad = np.asarray([
        float(tensors[name].grad) if tensors[name].grad is not None else 0.0
        for name in PARAM_NAMES
    ])
    return float(loss.value), grad


def finite_difference_grad(params: dict, config: ToyConfig, eps: float = 1e-5,
                           idx=(0, 1, 2)) -> np.ndarray:
    out = []
    base = _clip(params)
    for index in idx:
        name = PARAM_NAMES[index]
        plus = dict(base)
        minus = dict(base)
        plus[name] = plus[name] + eps
        minus[name] = minus[name] - eps
        loss_plus, _ = surrogate_loss_and_grad(plus, config)
        loss_minus, _ = surrogate_loss_and_grad(minus, config)
        out.append((loss_plus - loss_minus) / (2.0 * eps))
    return np.asarray(out)
