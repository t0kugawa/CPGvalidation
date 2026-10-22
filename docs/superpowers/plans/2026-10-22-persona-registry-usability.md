# PERSONA Registry Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in attribute freezing, a cosine learning-rate training mode, and a registry-created homeostatic spring to PERSONA without changing any existing default behavior.

**Architecture:** `Registry` captures each site's creation defaults, exposes freeze/unfreeze and spring creation, and computes spring forces from the current lr and site variance. `Agent` enforces frozen attributes in its apply phase, drives `FisherMatrix.lr_mu` from `CosineLR`, and adds registry spring forces after the natural-gradient/game step. `FisherMatrix`, `F`, `Xbar`, `G`, H, and the game math are untouched.

**Tech Stack:** Python 3.12, NumPy, pytest, NEST, existing `persona/` package.

**Spec:** `docs/superpowers/specs/2026-10-22-persona-registry-usability-design.md`

## Global Constraints

- All new behavior is opt-in. With no freeze, no spring, and `lr_schedule=None`, `Agent` behavior and history keys must match the current implementation exactly.
- Do not modify `F`, `Xbar`, `G`, the noise channels, the sigma rule, the recursive inverse, H, or the game math.
- Do not modify `tinycpg/`, `memcpg/`, or root `persona.py`.
- Freezeable attributes are exactly `mean` and `sigma`; other attributes raise `ValueError`.
- Creation defaults are the effective values at network finalization, not the values at freeze time.
- `CosineLR` uses exact endpoints: step 1 is `start`, step `total_steps` is `end`, later steps stay at `end`.
- Dynamic spring power is `clamp(gain * lr_t * sigma_ref**2 / max(sigma**2, eps), 0, max_power)`; custom power is a constant in `[0, max_power]`.
- The spring is zero during strategy warm-up and never moves a frozen mean.
- Use `pytest` for all checks; commit after each task.

## Review Focus

- **Freezing after learning, or after a post-create mutation, must restore the effective creation anchor.** Covered by `test_freeze_mean_snaps_and_pins_only_mean` and `test_finalize_refreshes_defaults_after_post_create_mutation` in Task 1.
- **Frozen sigma must record `Δlogσ = 0`, preserving `traced = applied`.** Covered by `test_frozen_sigma_holds_and_trace_records_zero` in Task 2.
- **Cosine endpoints must not be off by one.** Step 1 must be exactly `start` and step `total_steps` exactly `end`. Covered by `test_endpoints_and_midpoint` in Task 3.
- **Dynamic spring must survive zero sigma and zero reference sigma without dividing by zero.** Covered by `test_dynamic_power_zero_sigma_is_finite` in Task 6.
- **The spring must not move the host during warm-up.** Covered by `test_spring_does_not_apply_during_warmup` in Task 7.

---

### Task 1: Registry creation defaults and freeze API

**Files:**
- Modify: `persona/registry.py:54-70` (state), `:161-184` (`create_population`), `:186-225` (`create_connection`), `:227-280` (`create_relay`), `:321-342` (`finalize_connections`)
- Test: `tests/test_registry_freeze.py`

**Interfaces:**
- Consumes: existing `Registry.get_all_optimizable_params()` and canonical keys.
- Produces:
  - `Registry._defaults: Dict[str, Dict[str, float]]`
  - `Registry.freeze(keys, attrs=("mean",)) -> tuple[str, ...]`
  - `Registry.unfreeze(keys, attrs=None) -> tuple[str, ...]`
  - `Registry.frozen_attrs(site_key) -> frozenset[str]`
  - `Registry.is_frozen(site_key, attr) -> bool`
  - `Registry.defaults(site_key) -> Dict[str, float]`
  - `Registry.recapture_defaults(keys=None) -> None`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from persona.parameter import PersonaParam
from persona.population import Population
from persona.registry import Registry


def _reg():
    reg = Registry()
    reg.create_population(Population("izhikevich", 4, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=2.0, sigma=0.4, min=0.0, max=5.0),
    }, "pop"))
    return reg


def test_defaults_captured_at_creation(nest_kernel):
    reg = _reg()
    assert reg.defaults("pop_pop_a")["mean"] == 1.0
    assert reg.defaults("pop_pop_a")["sigma"] == 0.3


def test_freeze_mean_snaps_and_pins_only_mean(nest_kernel):
    reg = _reg()
    p = reg.get_all_optimizable_params()["pop_pop_a"]
    p.mean, p.sigma = 2.5, 0.9
    reg.freeze("pop_pop_a", attrs=("mean",))
    assert p.mean == 1.0
    assert p.sigma == 0.9
    assert reg.is_frozen("pop_pop_a", "mean")
    assert not reg.is_frozen("pop_pop_a", "sigma")


def test_freeze_glob_and_unfreeze(nest_kernel):
    reg = _reg()
    assert reg.freeze("pop_*_a", attrs=("mean",)) == ("pop_pop_a",)
    assert reg.frozen_attrs("pop_pop_a") == frozenset({"mean"})
    reg.unfreeze("pop_pop_a")
    assert reg.frozen_attrs("pop_pop_a") == frozenset()


def test_freeze_unknown_pattern_and_attr(nest_kernel):
    reg = _reg()
    with pytest.raises(KeyError):
        reg.freeze("nope_*")
    with pytest.raises(ValueError):
        reg.freeze("pop_pop_a", attrs=("min",))


def test_recapture_refreshes_the_anchor(nest_kernel):
    reg = _reg()
    p = reg.get_all_optimizable_params()["pop_pop_a"]
    p.mean, p.sigma = 7.0, 0.0
    reg.recapture_defaults("pop_pop_a")
    assert reg.defaults("pop_pop_a")["mean"] == 7.0
    reg.freeze("pop_pop_a", attrs=("mean", "sigma"))
    assert (p.mean, p.sigma) == (7.0, 0.0)


def test_finalize_refreshes_defaults_after_post_create_mutation(nest_kernel):
    reg = _reg()
    p = reg.get_all_optimizable_params()["pop_pop_a"]
    p.mean, p.sigma = 5.0, 0.0
    reg.finalize_connections()
    assert reg.defaults("pop_pop_a")["mean"] == 5.0
    assert reg.defaults("pop_pop_a")["sigma"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_registry_freeze.py -v`

Expected: FAIL because `Registry` has no `defaults`, `freeze`, `frozen_attrs`, `is_frozen`, or `recapture_defaults`.

- [ ] **Step 3: Implement the registry state and API**

In `Registry.__init__`, add:

```python
self._defaults: Dict[str, Dict[str, float]] = {}
self._frozen: Dict[str, set[str]] = {}
```

Add a private `_record_defaults(site_key, param)` that stores `mean`, `sigma`, `min`, `max`, `sigma_min`, `sigma_max` as floats. Call it for every site in `create_population`, `create_connection` (after `clamp_weight_sign`), and `create_relay` (for `pop_<name>_rate`).

Add `recapture_defaults(keys=None)`: resolve all sites when `keys is None`, otherwise use the same exact-key/glob resolver used by `freeze`, and call `_record_defaults` for each. Call `recapture_defaults()` at the end of `finalize_connections`.

Add `freeze(keys, attrs=("mean",))`:

```python
def freeze(self, keys, attrs=("mean",)) -> tuple[str, ...]:
```

Resolve keys with `fnmatch.fnmatchcase` against `get_all_optimizable_params()`. Normalize `attrs` to a tuple; raise `ValueError` for anything outside `{"mean", "sigma"}`. For each resolved key, require `_defaults[key]`, set the param attribute to the stored default, and merge the attr into `_frozen[key]`. Return the sorted resolved keys.

Add `unfreeze(keys, attrs=None)`, `frozen_attrs(site_key)`, `is_frozen(site_key, attr)`, and `defaults(site_key)` (return a copy). `unfreeze` with `attrs=None` removes all pins for the site.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_registry_freeze.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/registry.py tests/test_registry_freeze.py
git commit -m "feat(persona): capture creation defaults and freeze site attributes"
```

---

### Task 2: Agent freeze enforcement and trace parity

**Files:**
- Modify: `persona/agent.py:186-292` (`step`)
- Test: `tests/test_agent_freeze.py`

**Interfaces:**
- Consumes: `Registry.frozen_attrs(site_key)` from Task 1.
- Produces: frozen `mean` keeps `param.mean` and returns `delta_means[key] == 0.0`; frozen `sigma` keeps `param.sigma` and records `Δlogσ == 0.0` in the strategy's site tuple.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry


def _agent():
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
    }, "pop"))
    return Agent(registry=reg, lr_mu=0.5, lr_sigma=0.5, warmup=0,
                 dt_ms=10.0, tau_mem_s=0.2), reg


def test_frozen_mean_holds_and_sigma_still_updates(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.freeze(key, attrs=("mean",))
    sigma0 = agent.params[key].sigma
    last = None
    for t in range(20):
        agent.sample_actions(t_ms=t * 10.0)
        last = agent.step(np.zeros(2), np.ones(2) * t, phase="stance",
                          t_ms=t * 10.0)
    assert agent.params[key].mean == 1.0
    assert agent.params[key].sigma != sigma0
    assert last["delta_means"][key] == 0.0


def test_frozen_sigma_holds_and_trace_records_zero(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.freeze(key, attrs=("sigma",))
    sigma0 = agent.params[key].sigma
    seen = {}
    orig = agent.strategy.step

    def spy(sites, noise, advantage, **kwargs):
        seen.update(sites)
        return orig(sites, noise, advantage, **kwargs)

    agent.strategy.step = spy
    for t in range(20):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * t, phase="stance",
                   t_ms=t * 10.0)
    assert agent.params[key].sigma == sigma0
    assert seen[key][2] == 0.0
    assert agent.params[key].mean != 1.0


def test_freezing_both_then_unfreezing_resumes_updates(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.freeze(key, attrs=("mean", "sigma"))
    sigma0 = agent.params[key].sigma
    for t in range(20):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * t, phase="stance",
                   t_ms=t * 10.0)
    assert (agent.params[key].mean, agent.params[key].sigma) == (1.0, sigma0)
    reg.unfreeze(key)
    for t in range(20, 25):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2) * t, phase="stance", t_ms=t * 10.0)
    assert (agent.params[key].mean, agent.params[key].sigma) != (1.0, sigma0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_agent_freeze.py -v`

Expected: FAIL because the Agent applies the frozen mean/sigma updates anyway.

- [ ] **Step 3: Implement freeze enforcement in `Agent.step`**

After `apply_updates` is computed, build `frozen = {name: registry.frozen_attrs(name) for name in self.params}`.

In the `sites` construction, compute `d_log_sigma = 0.0` when `"sigma" in frozen[name]`.

In the apply loop, call `param.update_mean(...)` only when `"mean" not in frozen[name]`. Call `param.update_sigma(...)` only when `"sigma" not in frozen[name]`. After the apply loop, set `delta_means[name] = 0.0` for every site whose mean is frozen. Do not change sampling, recording, strategy rows, or history.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_agent_freeze.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/agent.py tests/test_agent_freeze.py
git commit -m "feat(persona): enforce frozen mean and sigma in the agent"
```

---

### Task 3: Cosine learning-rate schedule

**Files:**
- Create: `persona/schedule.py`
- Modify: `persona/__init__.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Produces:
  - `CosineLR(start: float, end: float, total_steps: int, warmup_steps: int = 0)` with `.at(step: int) -> float`
  - `resolve_lr_schedule(mode: str, start: float, end: float | None, total_steps: int) -> CosineLR | None`

- [ ] **Step 1: Write the failing tests**

```python
import math

import pytest

from persona.schedule import CosineLR, resolve_lr_schedule


def test_endpoints_and_midpoint():
    s = CosineLR(start=0.1, end=0.0, total_steps=11)
    assert s.at(1) == 0.1
    assert s.at(11) == 0.0
    assert s.at(100) == 0.0
    assert math.isclose(s.at(6), 0.05, rel_tol=1e-12)


def test_warmup_plateau_then_decay():
    s = CosineLR(0.1, 0.0, 10, warmup_steps=3)
    assert s.at(1) == 0.1
    assert s.at(4) == 0.1
    assert s.at(5) < 0.1
    assert s.at(10) == 0.0


def test_one_step_and_constant():
    assert CosineLR(0.1, 0.0, 1).at(1) == 0.1
    constant = CosineLR(0.05, 0.05, 10)
    assert all(constant.at(i) == 0.05 for i in range(1, 11))


@pytest.mark.parametrize("kwargs", [
    {"start": -0.1, "end": 0.0, "total_steps": 10},
    {"start": 0.1, "end": 0.0, "total_steps": 0},
    {"start": 0.1, "end": 0.0, "total_steps": 10, "warmup_steps": 10},
    {"start": float("nan"), "end": 0.0, "total_steps": 10},
])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        CosineLR(**kwargs)


def test_resolve_lr_schedule():
    assert resolve_lr_schedule("constant", 0.03, None, 10) is None
    s = resolve_lr_schedule("cosine", 0.03, 0.003, 10)
    assert isinstance(s, CosineLR)
    assert (s.start, s.end, s.total_steps) == (0.03, 0.003, 10)
    with pytest.raises(ValueError):
        resolve_lr_schedule("cosine", 0.03, None, 10)
    with pytest.raises(ValueError):
        resolve_lr_schedule("bogus", 0.03, 0.0, 10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_schedule.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'persona.schedule'`.

- [ ] **Step 3: Implement `persona/schedule.py`**

Implement the frozen `CosineLR` dataclass with the validation from the spec. The formula is: one-step returns `start`; `step <= warmup_steps + 1` returns `start`; `step >= total_steps` returns `end`; otherwise `p = (step - warmup_steps - 1) / (total_steps - warmup_steps - 1)` and `lr = end + 0.5 * (start - end) * (1 + cos(pi * p))`.

Implement `resolve_lr_schedule`: `"constant"` returns `None`; `"cosine"` raises `ValueError` when `end is None`, otherwise returns `CosineLR`; any other mode raises `ValueError`.

Export `CosineLR` and `resolve_lr_schedule` from `persona/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_schedule.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/schedule.py persona/__init__.py tests/test_schedule.py
git commit -m "feat(persona): add cosine learning-rate schedule"
```

---

### Task 4: Agent schedule wiring and history

**Files:**
- Modify: `persona/agent.py:25-104` (constructor), `:186-292` (`step`)
- Test: `tests/test_agent_schedule.py`

**Interfaces:**
- Consumes: `CosineLR` from Task 3.
- Produces:
  - `Agent(..., lr_schedule: CosineLR | None = None)`
  - `Agent.current_lr_mu` property
  - `Agent.history["lr_mu"]` only when a schedule is configured

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.schedule import CosineLR


def _agent(schedule=None):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 3,
                          {"a": PersonaParam(mean=1.0, sigma=0.1)}, "pop"))
    return Agent(registry=reg, lr_mu=0.2, lr_sigma=0.0, warmup=0,
                 dt_ms=10.0, tau_mem_s=0.2, lr_schedule=schedule), reg


def test_schedule_drives_strategy_lr(nest_kernel):
    agent, _ = _agent(CosineLR(start=0.2, end=0.0, total_steps=4))
    seen = []
    orig = agent.strategy.step

    def spy(sites, noise, advantage, **kwargs):
        seen.append(agent.strategy.lr_mu)
        return orig(sites, noise, advantage, **kwargs)

    agent.strategy.step = spy
    for t in range(4):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=t * 10.0)
    assert seen == pytest.approx([0.2, 0.15, 0.05, 0.0])
    assert agent.history["lr_mu"] == pytest.approx([0.2, 0.15, 0.05, 0.0])


def test_no_schedule_keeps_fixed_lr_and_no_history_key(nest_kernel):
    agent, _ = _agent()
    assert agent.current_lr_mu == 0.2
    agent.sample_actions(t_ms=0.0)
    agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert agent.strategy.lr_mu == 0.2
    assert "lr_mu" not in agent.history
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_agent_schedule.py -v`

Expected: FAIL because `Agent` has no `lr_schedule` parameter or `current_lr_mu` property.

- [ ] **Step 3: Implement the wiring**

Add `lr_schedule: CosineLR | None = None` to `Agent.__init__` and store it. When it is not `None`, initialize `self.history["lr_mu"] = []`.

Add `current_lr_mu` as a property returning `self.lr_schedule.at(self.current_step)` when a schedule exists, otherwise `self.lr_mu`.

In `step`, after `self.current_step += 1`, compute `lr_t = self.current_lr_mu`, set `self.strategy.lr_mu = lr_t` before `strategy.step`, and append `lr_t` to `self.history["lr_mu"]` when the schedule exists.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_agent_schedule.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/agent.py tests/test_agent_schedule.py
git commit -m "feat(persona): drive the agent learning rate from a schedule"
```

---

### Task 5: Cosine CLI mode and run-horizon wiring

**Files:**
- Modify: `personalize.py:56-194` (`build_personalizer`), `:197-273` (`run_personalization`), `:304-334` (`main` and `__main__`)
- Test: `tests/test_cli_lr.py`

**Interfaces:**
- Consumes: `resolve_lr_schedule` from Task 3, `Agent(lr_schedule=...)` from Task 4.
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `parse_cli(argv=None) -> argparse.Namespace`
  - `build_personalizer(..., lr_mode="constant", lr_start=0.03, lr_end=None)`
  - `run_personalization(..., lr_mode="constant", lr_start=0.03, lr_end=None)`
  - CLI flags `--steps`, `--lr-mode`, `--lr-start`, `--lr-end`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from personalize import build_parser, parse_cli


def test_cli_defaults_keep_constant_mode():
    args = parse_cli([])
    assert args.steps == 200
    assert args.lr_mode == "constant"
    assert args.lr_start == 0.03
    assert args.lr_end is None


def test_cli_cosine_accepts_explicit_endpoints():
    args = parse_cli([
        "--lr-mode", "cosine", "--lr-start", "0.03",
        "--lr-end", "0.003", "--steps", "50",
    ])
    assert (args.lr_mode, args.lr_start, args.lr_end, args.steps) == (
        "cosine", 0.03, 0.003, 50)


def test_cli_cosine_without_end_is_an_error():
    with pytest.raises(SystemExit):
        parse_cli(["--lr-mode", "cosine"])


def test_parser_still_exposes_the_flags():
    args = build_parser().parse_args(["--lr-mode", "cosine", "--lr-end", "0.0"])
    assert (args.lr_mode, args.lr_end) == ("cosine", 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cli_lr.py -v`

Expected: FAIL with `ImportError: cannot import name 'build_parser'`.

- [ ] **Step 3: Implement the CLI and wiring**

Move argument construction into `build_parser()` and add the four flags. Add `parse_cli(argv=None)` that calls `build_parser().parse_args(argv)` and, when `lr_mode == "cosine"` and `lr_end is None`, calls `parser.error("--lr-end is required with --lr-mode cosine")`. Use `parse_cli()` in `__main__`, then call `main(n_steps=args.steps, ..., lr_mode=args.lr_mode, lr_start=args.lr_start, lr_end=args.lr_end)`.

Add `lr_mode`, `lr_start`, and `lr_end` to `main` and `run_personalization`, passing them through to `build_personalizer`. In `build_personalizer`, compute `lr_schedule = resolve_lr_schedule(lr_mode, lr_start, lr_end, max_steps)` and construct the Agent with `lr_mu=lr_start` and `lr_schedule=lr_schedule`. In constant mode this reproduces the current `lr_mu=0.03`; `lr_schedule` is `None`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cli_lr.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personalize.py tests/test_cli_lr.py
git commit -m "feat: add cosine learning-rate CLI training mode"
```

---

### Task 6: Homeostatic spring creation and power math

**Files:**
- Create: `persona/spring.py`
- Modify: `persona/registry.py:54-70` (state), after `defaults`/freeze API (new methods)
- Modify: `persona/__init__.py`
- Test: `tests/test_registry_spring.py`

**Interfaces:**
- Consumes: `Registry._defaults`, `Registry._resolve_site_keys` from Task 1.
- Produces:
  - `SpringSpec(name: str, keys: tuple[str, ...], power: float | None, gain: float, max_power: float)` with `.power_for(sigma, sigma_ref, lr_t) -> float`
  - `Registry.create_spring(name, keys, *, power=None, gain=1.0, max_power=0.5) -> SpringSpec`
  - `Registry.remove_spring(name) -> None`
  - `Registry.springs() -> Dict[str, SpringSpec]`
  - `Registry.spring_state(means, sigmas, lr_t) -> Dict[str, Dict[str, float]]`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from persona.parameter import PersonaParam
from persona.population import Population
from persona.registry import Registry


def _reg():
    reg = Registry()
    reg.create_population(Population("izhikevich", 4, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=2.0, sigma=0.4, min=0.0, max=5.0),
        "c": PersonaParam(mean=0.0, sigma=0.0, min=0.0, max=1.0),
    }, "pop"))
    return reg


def test_custom_spring_force_and_anchor(nest_kernel):
    reg = _reg()
    reg.create_spring("s", "pop_pop_a", power=0.25)
    state = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.3}, 0.1)
    assert state["pop_pop_a"]["power"] == 0.25
    assert state["pop_pop_a"]["force"] == pytest.approx(-0.25)
    assert state["pop_pop_a"]["anchor"] == 1.0


def test_dynamic_power_uses_lr_and_variance(nest_kernel):
    reg = _reg()
    reg.create_spring("s", "pop_pop_a", power=None, gain=1.0, max_power=0.5)
    at_ref = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.3}, 0.1)["pop_pop_a"]["power"]
    tighter = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.15}, 0.1)["pop_pop_a"]["power"]
    looser = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.6}, 0.1)["pop_pop_a"]["power"]
    assert at_ref == pytest.approx(0.1)
    assert tighter > at_ref
    assert looser < at_ref
    assert tighter <= 0.5


def test_dynamic_power_zero_sigma_is_finite(nest_kernel):
    reg = _reg()
    reg.create_spring("s", "pop_pop_a", power=None, gain=1.0, max_power=0.5)
    state = reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.0}, 1.0)
    assert state["pop_pop_a"]["power"] == pytest.approx(0.5)
    reg.create_spring("zero_ref", "pop_pop_c", power=None, gain=1.0, max_power=0.5)
    zero = reg.spring_state({"pop_pop_c": 0.0}, {"pop_pop_c": 0.0}, 1.0)
    assert zero["pop_pop_c"]["power"] == 0.0


def test_spring_skips_frozen_mean(nest_kernel):
    reg = _reg()
    reg.freeze("pop_pop_a", attrs=("mean",))
    reg.create_spring("s", "pop_pop_a", power=0.25)
    assert reg.spring_state({"pop_pop_a": 2.0}, {"pop_pop_a": 0.3}, 0.1) == {}


def test_spring_overlap_and_validation(nest_kernel):
    reg = _reg()
    reg.create_spring("s1", "pop_*")
    with pytest.raises(ValueError):
        reg.create_spring("s2", "pop_pop_a")
    reg.remove_spring("s1")
    with pytest.raises(ValueError):
        reg.create_spring("s3", "pop_pop_b", power=0.9, max_power=0.5)
    assert reg.springs() == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_registry_spring.py -v`

Expected: FAIL because `Registry` has no `create_spring` or `spring_state`.

- [ ] **Step 3: Implement `SpringSpec` and registry methods**

`SpringSpec.power_for`: custom power returns it unchanged; dynamic returns 0 when `sigma_ref <= 0` or `lr_t <= 0`, otherwise `min(max(gain * lr_t * sigma_ref**2 / max(sigma**2, 1e-12), 0.0), max_power)`.

`create_spring`: validate `name` is new, `0 < max_power <= 1`, `gain` finite and non-negative, and `power` (when given) finite and in `[0, max_power]`. Resolve keys with the Task 1 resolver; require every key to have creation defaults; reject any overlap with an existing spring. Store and return the spec.

`spring_state`: iterate springs; skip keys absent from `means`/`sigmas`; skip frozen means; compute power via `power_for`; emit `force`, `power`, `anchor` (default mean), `mean`, and `lr_mu`.

Export `SpringSpec` from `persona/__init__.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_registry_spring.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/spring.py persona/registry.py persona/__init__.py tests/test_registry_spring.py
git commit -m "feat(persona): add registry-created homeostatic springs"
```

---

### Task 7: Agent spring application and diagnostics

**Files:**
- Modify: `persona/agent.py:186-292` (`step`)
- Test: `tests/test_agent_spring.py`

**Interfaces:**
- Consumes: `Registry.spring_state(...)` from Task 6.
- Produces: spring force added to the applied mean delta, `Agent.history["spring"]` with per-site `force`/`power`/`anchor`/`mean`/`lr_mu`, no spring during warm-up, no spring on a frozen mean.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry


def _agent(warmup=0):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
    }, "pop"))
    return Agent(registry=reg, lr_mu=0.0, lr_sigma=0.0, warmup=warmup,
                 dt_ms=10.0), reg


def test_custom_spring_moves_mean_and_is_logged(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    reg.create_spring("s", key, power=0.5)
    agent.params[key].mean = 2.0
    agent.sample_actions(t_ms=0.0)
    res = agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert agent.params[key].mean == pytest.approx(1.5)
    assert res["delta_means"][key] == pytest.approx(-0.5)
    assert agent.history["spring"][-1][key]["force"] == pytest.approx(-0.5)


def test_spring_does_not_apply_during_warmup(nest_kernel):
    agent, reg = _agent(warmup=3)
    key = "pop_pop_a"
    reg.create_spring("s", key, power=0.5)
    agent.params[key].mean = 2.0
    for t in range(3):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=t * 10.0)
    assert agent.params[key].mean == 2.0
    assert agent.history["spring"][-1] == {}
    agent.sample_actions(t_ms=30.0)
    agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=30.0)
    assert agent.params[key].mean == pytest.approx(1.5)


def test_spring_skips_frozen_mean(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    agent.params[key].mean = 2.0
    reg.freeze(key, attrs=("mean",))          # snaps back to the creation anchor
    reg.create_spring("s", key, power=0.5)
    agent.sample_actions(t_ms=0.0)
    res = agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert agent.params[key].mean == 1.0
    assert res["delta_means"][key] == 0.0
    assert agent.history["spring"][-1] == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_agent_spring.py -v`

Expected: FAIL because the Agent ignores registry springs.

- [ ] **Step 3: Implement spring application**

In `step`, when `self.registry.springs()` is non-empty and `apply_updates` is true, build `means` and `sigmas` from the sampling-time snapshot and call `spring = self.registry.spring_state(means, sigmas, lr_t)`. Otherwise use `{}`.

In the apply loop, for a non-frozen mean, add `spring.get(name, {}).get("force", 0.0)` to the strategy delta before `param.update_mean`; store the summed value back into `delta_means[name]`. For a frozen mean, keep `delta_means[name] = 0.0`.

When springs exist, initialize `self.history["spring"]` lazily and append the `spring` dict each step (empty during warm-up or when all spring keys are frozen). Do not add the key when no spring exists.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_agent_spring.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/agent.py tests/test_agent_spring.py
git commit -m "feat(persona): apply registry springs to agent mean updates"
```

---

### Task 8: Integration, regression, and docs

**Files:**
- Test: `tests/test_usability_integration.py`
- Modify: `docs/superpowers/specs/2026-10-22-persona-registry-usability-design.md` (status/ledger line only)

**Interfaces:**
- Consumes: all previous tasks.

- [ ] **Step 1: Write the integration tests**

```python
import numpy as np
import pytest

from persona.agent import Agent
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.schedule import CosineLR


def test_freeze_spring_and_cosine_coexist(nest_kernel):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5, {
        "a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
        "b": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0),
    }, "pop"))
    reg.freeze("pop_pop_a", attrs=("mean",))
    reg.create_spring("s", "pop_pop_b", power=0.5)
    agent = Agent(registry=reg, lr_mu=0.2, lr_sigma=0.0, warmup=0,
                  dt_ms=10.0, lr_schedule=CosineLR(0.2, 0.0, 5))
    agent.params["pop_pop_b"].mean = 2.0
    for t in range(5):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=t * 10.0)
    assert agent.params["pop_pop_a"].mean == 1.0
    assert agent.params["pop_pop_b"].mean < 2.0
    assert len(agent.history["lr_mu"]) == 5
    assert "pop_pop_b" in agent.history["spring"][-1]


def test_default_agent_has_no_new_history_keys(nest_kernel):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 3,
                          {"a": PersonaParam(mean=1.0, sigma=0.1)}, "pop"))
    agent = Agent(registry=reg, warmup=0)
    agent.sample_actions(t_ms=0.0)
    agent.step(np.zeros(2), np.ones(2), phase="stance", t_ms=10.0)
    assert "lr_mu" not in agent.history
    assert "spring" not in agent.history
```

- [ ] **Step 2: Run the integration tests**

Run: `pytest tests/test_usability_integration.py -v`

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `pytest -q`

Expected: all existing tests plus the new tests pass.

- [ ] **Step 4: Update the spec status line**

Change the spec's status line to record that the design was approved and implemented on 2026-10-22. Do not change the design decisions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_usability_integration.py docs/superpowers/specs/2026-10-22-persona-registry-usability-design.md
git commit -m "test(persona): cover freeze, cosine lr, and spring together"
```

---

## Self-Review

- **Spec coverage:** §3 freeze → Tasks 1–2; §4 cosine → Tasks 3–5; §5 spring → Tasks 6–7; §6 interactions, §7 files, §8 tests, §9 order → Task 8 and the task mapping above.
- **Step scan:** every step names one action and one check; no TBD/TODO placeholders.
- **Type consistency:** `freeze`/`unfreeze` return `tuple[str, ...]`; `frozen_attrs` returns `frozenset[str]`; `spring_state` returns `Dict[str, Dict[str, float]]`; `CosineLR.at` takes a 1-based `int`; `Agent.history["lr_mu"]` is a list of floats.
- **Review Focus:** each line has a named test in Task 1, 2, 3, 6, or 7.
- **Proportion:** the plan names signatures, values, and tests; implementation bodies are left to the builder except for the two formulas the spec fixes.
