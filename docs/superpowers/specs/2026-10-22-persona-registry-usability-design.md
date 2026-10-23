# PERSONA registry usability: attribute freeze, cosine LR, homeostatic spring — design

**Status:** implemented (approved 2026-10-22; verified 2026-10-23). Opt-in; no production default changes.
First production migration target: the `t2xcpg` flexor freeze (see §2.1).

This is a usability layer, not a change to the PERSONA concept. It does not
modify the H design, the conserved-stack game design, `F`, `Xbar`, `G`, the
noise channels, the sigma rule, or the recursive inverse. The H and game specs
rule out absolute setpoints **inside their own regulators**; the freeze and
spring here are explicit caller controls on individual sites, requested
separately, and they do not alter those regulators' math.

## 1. Intent

Three independent, opt-in controls:

1. **Freeze selected attributes of selected sites.** A site exposes `mean` and
   `sigma` as learned attributes; freezing pins only the named attributes to
   their creation defaults. The others keep changing normally. Freezing `mean`
   does not stop exploration or sigma adaptation; freezing `sigma` does not
   stop mean learning.
2. **Cosine learning-rate training mode.** The mean learning rate follows a
   cosine from an explicit initial value to an explicit target value over the
   configured training horizon. No schedule is used unless requested.
3. **Homeostatic spring.** An explicit force pulls selected site means toward
   their creation defaults. Its stiffness ("power") is either a custom constant
   or a dynamic value derived from the current learning rate and the site's
   current parameter variance. It is created through the registry.

All three are usable together. Defaults are unchanged when unused.

## 2. What is already settled by the existing specs

- `Registry` is the single creation route (`create_population`,
  `create_connection`, `create_relay`) and owns canonical site keys
  `pop_<name>_<param>` / `conn_<key>_<param>`.
- A site has one `sigma` used for both exploration and dispersion; `mean` is
  the learned centre.
- `PersonaParam` has no `default` field. The creation seed `mean` / `sigma` is
  the only "default" that exists, so the registry must capture it.
- `Agent.step` applies `delta_means` first, then the log-sigma update, both
  from the sampling-time snapshot.
- `lr_mu` is one scalar on `FisherMatrix`; `Agent` passes it at construction.
- `FisherMatrix` owns `F/Xbar/G`, the recursive inverse, and the game/H hooks.

### 2.1 Current migration target (2026-10-22)

The first production caller is the CPG flexor freeze currently implemented as
`min=max`/`sigma=0` in `t2xcpg/leg.py`: the six Renshaw weights on `mean` and
the M-F Izhikevich `a/b/c/d` on `mean` and `sigma`. Migrate it to
`registry.freeze(...)` after `finalize_connections`; `recapture_defaults` must
anchor the post-creation M-F values. **Migrated (2026-10-23):** `t2xcpg/leg.py`
now calls `registry.freeze` for the six Renshaw means and the M-F
`mean`+`sigma`; `finalize_connections` captures the post-creation `sigma=0`
anchor, and the `min=max` fallback is gone. This is objective C2 of
`2026-10-15-persona-variantB-cpg-mvp-design.md`; the frozen values are listed
in its §5.10/V24.

## 3. Feature 1 — attribute freeze

### 3.1 Registry state

On every `create_population`, `create_connection`, and `create_relay` site
registration, the registry records the creation defaults:

```text
_defaults[site_key] = {
    "mean": current_mean,
    "sigma": current_sigma,
    "min": current_min,
    "max": current_max,
    "sigma_min": current_sigma_min,
    "sigma_max": current_sigma_max,
}
```

The snapshot is taken at registration, after the connection sign clamp (the
clamp changes `min`/`max`, not `mean`/`sigma`). Some builders mutate a
`PersonaParam` after creation (for example `t2xcpg/leg.py` silences the flexor
motor pool by setting its post-creation `sigma=0`). To make the anchor the
effective creation configuration, `finalize_connections` refreshes the
snapshot for every registered site, and the registry exposes
`recapture_defaults(keys=None)` for a caller that mutates a site later.
`freeze` must be called after the final capture (normally after network
construction); calling it earlier is allowed but anchors to the provisional
values. A site with no recorded creation default cannot be frozen; `freeze`
raises a clear error instead of silently anchoring to a moved value.

Freezable learned attributes are `mean` and `sigma`. The other constructor
attributes are static configuration and are not freezable in v1. Unknown
attributes raise `ValueError`.

### 3.2 API

```python
registry.freeze(keys, attrs=("mean",)) -> tuple[str, ...]
registry.unfreeze(keys, attrs=None) -> tuple[str, ...]
registry.frozen_attrs(site_key) -> frozenset[str]
registry.is_frozen(site_key, attr) -> bool
registry.defaults(site_key) -> dict
registry.recapture_defaults(keys=None) -> None
```

- `keys` is one string or an iterable of strings. Each string is either an
  exact canonical key or an `fnmatch` glob (`*`, `?`, `[...]`). A pattern that
  matches no site raises `KeyError`; the error lists the unmatched pattern.
- `attrs` is one attribute name or an iterable of names. Repeated calls merge.
- `freeze` immediately snaps each pinned attribute to its creation default, so
  freezing after some learning still restores the default. `unfreeze` removes
  the pin but does not restore or reset anything; the attribute continues from
  its current value.
- Freeze patterns are resolved at call time to exact keys. A site created
  later is not automatically frozen by an earlier glob; call `freeze` again.
- The method is intended to be called after the network is built and before
  the training loop. It is safe to call again to add or remove pins.

### 3.3 Enforcement

`Agent.step` is the only enforcement point, because it owns the apply phase:

- **`mean` frozen** — the mean update is skipped. `param.mean` stays at the
  creation default. The returned `delta_means[key]` is `0.0` for that site so
  diagnostics show what was actually applied.
- **`sigma` frozen** — the log-sigma update is skipped and the trace's
  `Δlogσ` for that site is `0.0`, preserving `traced = applied`. `param.sigma`
  stays at the creation default.
- **both frozen** — both updates are skipped; the site still occupies its
  `F/Xbar/G` rows, and the strategy still scores its exploration samples.
- **neither frozen** — behaviour is exactly the current behaviour.

Sampling and realization are unchanged. A site with `mean` frozen still draws
`mean + sigma·ε` and still receives per-subject dispersion; only the learned
centre is held. A site with `sigma` frozen still learns its mean but explores
with the fixed sigma.

The strategy still includes frozen sites in the Fisher metric and trace. This
is deliberate: removing rows would change the strategy's dimensions at
runtime, and the site's exploration is still real data. Suppressing only the
applied update is the minimal, reversible change.

### 3.4 Bounds and type identity

Freeze never changes `min`/`max`, never flips a sign, and never changes a
population model or connection type. A frozen `mean` remains inside its existing
bounds because the captured default was already inside them.

## 4. Feature 2 — cosine learning-rate training mode

### 4.1 Schedule object

Add `persona/schedule.py`:

```python
@dataclass(frozen=True)
class CosineLR:
    start: float
    end: float
    total_steps: int
    warmup_steps: int = 0

    def at(self, step: int) -> float: ...
```

Validation: `start`, `end` finite and `>= 0`; `total_steps >= 1`;
`0 <= warmup_steps < total_steps`. If `start == end`, the schedule is constant.

Formula (`step` is the 1-based strategy step):

```text
if total_steps == 1:                 lr = start
elif step <= warmup_steps + 1:       lr = start
elif step >= total_steps:            lr = end
else:
    p = (step - warmup_steps - 1) / (total_steps - warmup_steps - 1)
    lr = end + 0.5 (start - end) (1 + cos(pi p))
```

The endpoints are exact: `at(1) == start` for any valid warm-up, and every
step at or after `total_steps` is `end`. The `warmup_steps + 1` plateau means
the first post-warm-up step is still `start`, and the cosine runs over the
remaining `total_steps - warmup_steps - 1` intervals. A one-step schedule
returns `start`. The horizon is the configured training horizon, not an
adaptive estimate. If a run stops early (`loss_stop`, wall clock, or
`max_steps` reached), the schedule simply stops where the run stopped; it does
not rescale.

### 4.2 Wiring

- `Agent` accepts `lr_schedule: CosineLR | None = None`.
- `Agent.current_lr_mu` returns `lr_schedule.at(current_step)` when a schedule
  is configured, otherwise the fixed `lr_mu`.
- Immediately before `strategy.step`, `Agent` sets
  `strategy.lr_mu = current_lr_mu`. The strategy's math is unchanged; only the
  scalar multiplier changes per step.
- The schedule affects `lr_mu` only. `lr_sigma` stays fixed; sigma has its own
  rule and is not part of this request.
- `Agent.history["lr_mu"]` is appended only when a schedule is configured, so
  an unscheduled run's history is unchanged.
- Direct `Agent` users construct `CosineLR(start=..., end=..., total_steps=...)`.
  `Personalizer`/`personalize.py` set `total_steps` from the configured run
  horizon (`max_steps`), so there is one source of truth for the horizon.

### 4.3 CLI

`personalize.py` gains a training-mode switch:

```text
--steps INT                   default: 200 (the current run horizon)
--lr-mode {constant,cosine}   default: constant
--lr-start FLOAT              default: 0.03 (the current hard-coded value)
--lr-end FLOAT                required when --lr-mode cosine
```

`constant` reproduces the current `lr_mu=0.03`; `--lr-start` is the fixed rate
in this mode, so passing `0.05` changes constant-mode behaviour deliberately.
`cosine` builds a `CosineLR` with the explicit start, end, and a horizon equal
to `--steps`. There is no separate `--lr-steps`: the run horizon is the single
source of truth, so the schedule can never silently diverge from the run.
`--lr-end` must be explicit for cosine; there is no hidden target value.

## 5. Feature 3 — homeostatic spring

### 5.1 Meaning

A spring is a direct pull on a site's **mean** toward its creation default:

```text
Δθ_spring,p = power_p (θ_default,p − θ_p)
```

`power_p` is a dimensionless per-step stiffness, not an exponent. A power of
0.05 removes 5% of the current displacement per step before clipping. The force
is added to the strategy's mean delta after the natural-gradient and game
steps, and before `PersonaParam.update_mean`, so the hard bounds still apply.

Only the mean is sprung in v1. The anchor is always the creation default
recorded in §3.1. Springing a frozen mean is a no-op; freeze wins.

### 5.2 Creation through the registry

```python
spec = registry.create_spring(
    name, keys, *, power=None, gain=1.0, max_power=0.5)
registry.remove_spring(name)
registry.springs() -> mapping
```

- `keys` uses the same exact-key / glob resolution as `freeze`; at least one
  site must match.
- `power` is `None` for dynamic power, or a finite constant in
  `[0, max_power]` for custom power.
- `gain` and `max_power` are used only by dynamic power.
- Two springs may not own the same site; a second creation that overlaps an
  existing spring raises `ValueError`. This keeps the force unambiguous and
  makes the per-site power log meaningful.
- The anchor is fixed at creation-default capture time; it does not follow the
  site if learning moves it before the spring is created.

### 5.3 Power laws

Custom power:

```text
power_p = power
```

Dynamic power:

```text
power_p = clamp(gain · lr_t · σ_ref,p² / max(σ_p², ε), 0, max_power)
```

where:

- `lr_t` is the current mean learning rate (the scheduled value when cosine is
  active, otherwise `Agent.lr_mu`);
- `σ_p` is the sampling-time sigma;
- `σ_ref,p` is the site's creation sigma;
- `ε = max(σ_min,p², 1e-12)`.

At creation sigma, `power_p = gain · lr_t`; a site whose current sigma is
tighter than its creation sigma is pulled harder, and a looser site is pulled
less. Using the ratio makes the power invariant to the site's absolute scale.
The upper clamp is a stability guard: a per-step stiffness above `max_power`
could overshoot the anchor or fight the natural-gradient step.

### 5.4 Application order in `Agent.step`

The spring shares the existing warm-up gate: it is applied only when
`strategy.steps + 1 > strategy.warmup`. During warm-up no host mean moves and
`delta_means` stays all zero, preserving the current invariant.

For each site, after `strategy.step` returns and after the game's throttle (the
game is applied inside the strategy):

```text
if not apply_updates:              applied = 0
elif mean is frozen:               applied = 0
else:
    applied = delta_means[key] + spring_delta[key]
param.update_mean(applied)
```

`delta_means` returned by `Agent.step` is the **pre-clip requested** mean
displacement for the step (strategy + spring), or `0.0` for a frozen mean. It
keeps the existing pre-clip convention; `param.mean` after the step shows the
bounds-clipped actual value. `spring_delta[key]` is computed from the
sampling-time `mean` and `sigma` and the current `lr_t`, then clipped by
`update_mean` to the site bounds. A spring does not rewrite `G`, `Xbar`, `F`,
the credit, or the advantage; it only adds a mean displacement for this step. It coexists with H and the game but is not
part of either: H adjusts the credit vector, the game throttles the policy
delta, and the spring adds an explicit restoring force.

When no spring exists, the spring path is skipped exactly and no new
computation or RNG call occurs.

### 5.5 Defaults and diagnostics

- `power=None`, `gain=1.0`, `max_power=0.5` are the generic defaults; these
  are the only defaults introduced, and they apply only to an explicitly
  created spring.
- `Agent.history["spring"]` is appended only when springs exist, with per-site
  `force`, `power`, `anchor`, `mean`, and the current `lr_mu`.
- `registry.springs()` is inspectable for tests and callers.

## 6. Interaction matrix

| Situation | Result |
|---|---|
| `mean` frozen + spring owns the site | spring force is skipped; mean stays at default |
| `sigma` frozen + dynamic spring | power uses the frozen sigma; mean still learns and is sprung |
| both frozen + spring | no mean update at all; spring is a no-op |
| cosine + dynamic spring | `lr_t` is the current scheduled value, so the spring weakens as lr decays |
| cosine + custom spring | power is constant; lr does not enter |
| freeze/unfreeze mid-run | pin changes apply from the next `Agent.step`; no state is reset |
| no freeze/spring/schedule | exact current behaviour; no new history keys |

## 7. Files and interfaces

- `persona/registry.py` — creation-default capture, freeze state/API, spring
  specs/creation, `spring_deltas(...)`.
- `persona/parameter.py` — unchanged. Freeze is enforced by the Agent, not by
  changing `PersonaParam` semantics.
- `persona/agent.py` — freeze checks in `step`, schedule wiring, spring
  application, conditional history keys.
- `persona/schedule.py` — new `CosineLR`.
- `persona/spring.py` — new `SpringSpec` (or keep the small dataclass in
  `registry.py`; either way the public creation method is on the registry).
- `persona/personalization.py` — unchanged; its loop already consumes the
  Agent's per-step `lr_mu`.
- `personalize.py` — `build_personalizer` sets the schedule horizon from
  `max_steps`; CLI flags `--steps`, `--lr-mode`, `--lr-start`, `--lr-end`.
- `tests/` — new focused suites plus the existing suite unchanged.

## 8. Tests

### 8.1 Freeze

- Creation defaults are captured for population, static-connection, STDP
  hyper-parameter, and relay-rate sites.
- A post-creation mutation before `finalize_connections` is reflected by the
  refreshed default; `recapture_defaults` does the same for a later mutation
  (the flexor-pool `sigma=0` case).
- `freeze("...", attrs=("mean",))` restores the creation mean and holds it
  across steps while sigma changes and sampling still explores.
- `freeze("...", attrs=("sigma",))` holds sigma across steps while the mean
  learns; the recorded trace's `Δlogσ` is exactly 0.
- Freezing both holds both; `unfreeze` resumes updates from current values.
- Glob matching, exact-key matching, unmatched-pattern `KeyError`, unknown
  attribute `ValueError`, and missing creation default error.
- Freeze does not change bounds, types, signs, `F`, `Xbar`, or `G`.
- The returned `delta_means` is `0.0` for a frozen mean and pre-clip
  (strategy + spring) otherwise.

### 8.2 Cosine LR

- Endpoint values are exact: step 1 is `start`, step `total_steps` is `end`,
  later steps stay at `end`; midpoint matches the cosine midpoint; a one-step
  schedule returns `start`.
- `start == end` is constant; all steps through `warmup_steps + 1` hold
  `start`, then the cosine decays to `end`.
- `Agent` sets `strategy.lr_mu` before every step; an unscheduled Agent keeps
  the fixed value.
- History `lr_mu` is present only with a schedule.
- `Personalizer` uses `max_steps` as the horizon.
- Invalid schedules raise at construction.

### 8.3 Spring

- Custom power produces exactly `power · (anchor − mean)` and respects bounds.
- Dynamic power equals `gain · lr_t` at creation sigma, rises when sigma
  shrinks, falls when sigma grows, falls when lr decays, and respects
  `max_power`.
- The spring is zero during warm-up; the first applied force uses the first
  post-warm-up step's `lr_t`.
- A frozen mean skips the spring; a frozen sigma still allows the spring.
- The spring does not change `G`, `Xbar`, `F`, credit, or advantage.
- Overlapping springs raise; `remove_spring` stops the force.
- No spring → exact old deltas and no new history key.

### 8.4 Integration and regression

- An Agent with all three enabled runs finite steps and logs the expected
  diagnostics.
- `Agent(no freeze, no spring, lr_schedule=None)` reproduces the current
  behaviour exactly.
- Full existing suite passes without changing expected numbers.

## 9. Implementation order

1. Registry creation-default capture and freeze tests/API.
2. Agent freeze enforcement and trace parity tests.
3. `CosineLR` tests, then Agent/Personalizer/CLI wiring.
4. Spring spec, registry creation/overlap tests, then Agent application.
5. Interaction and regression tests.
6. Documentation/ledger update.

## 10. Decisions left to the builder (calls)

These are settled here so the plan is unambiguous; the user can override them
at approval.

- **Freeze anchor** = the effective value at creation, not the value at freeze
  time. `finalize_connections` refreshes it after post-creation mutations, and
  `recapture_defaults` covers later changes. Freezing snaps the attribute back
  to that value.
- **Freeze scope** = exact keys or globs resolved at call time; future sites
  are not auto-frozen.
- **Freezeable attributes** = `mean`, `sigma`; other constructor attributes
  are static and rejected.
- **Frozen sites stay in the strategy**; only the applied update is
  suppressed. Returned `delta_means` is `0.0` for a frozen mean.
- **Cosine endpoints** = exact `start` at step 1 and exact `end` at
  `total_steps`; horizon = the run's step count; no adaptive rescaling.
  `lr_sigma` is not scheduled.
- **Spring power** = per-step stiffness, custom constant or the normalized
  dynamic law in §5.3; `max_power=0.5` is the stability cap.
- **Spring anchor** = creation mean; spring acts on mean only in v1.
- **Spring timing** = after natural-gradient/game, before `update_mean`, and
  inside the existing warm-up gate (no spring during warm-up).
- **CLI scope** = cosine gets `--lr-mode/--lr-start/--lr-end`, with `--steps`
  as the single run horizon; freeze and spring are Python/registry APIs in v1.

## 11. Non-goals

- No change to the default Agent, strategy, H, game, sigma rule, or inverse.
- No change to `tinycpg/`, `memcpg/`, or root `persona.py`.
- No absolute setpoints inside H or the game; this layer is separate and
  opt-in.
- No per-site asynchronous clock, no new RNG streams, no new state in
  `PersonaParam`.
- No CLI parser for freeze/spring in v1.
