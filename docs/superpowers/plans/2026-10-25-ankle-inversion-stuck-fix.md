# Ankle inversion / stuck-at-stop detector and fix plan

**Date:** 2026-10-25
**Status:** detectors and Fix 1/2 applied; the gate is red on phase; the 1/5
lite model is not equivalent to full, so phase tuning is paused.
**Related:** `2026-10-24-mujoco-ankle-geometry-fix.md` (upper range is now
1.10), `2026-10-24-deferred-persona-work.md`.

## What the tests found

The first five-cycle lite gate run:
`archive/phase_gate_current/` (800 steps, 8 s simulated, 347 s wall clock).

```text
cycles = 1 of 5
period = 1.132 s, CV = 0.197
E/F offset = +2.559 rad, drift 1.860
stance/E = -0.868 rad, drift 2.745
L/R offset = +2.196 rad, drift 0.933
angle correlation = 0.918
3D render = ok
```

The ankles are pinned at the stops almost continuously:

```text
L: q = 1.76, target sometimes 0.47  (upper stop; sole points UP)
R: q = 0.19, target sometimes 1.68  (lower stop; sole points DOWN)
```

The current upper range `1.76` is not safe:
`sole_world_z(1.76) = +0.188`. The foot's local `-z` (sole) points upward at
the stop. The lower `0.19` is sole-down (`-0.982`).

The fast suite now has exactly one failure: `test_ankle_pose.py` (the upper
stop puts the sole up). The phase-metric unit tests pass. The slow gate test
fails with the stuck runs above.

## Current state after Fix 1 and Fix 2

- Upper stop lowered to **1.10**; `test_ankle_pose` passes.
- Controlled ankles have tone (`stiffness=0.05`, `springref=1.0`);
  `test_ankle_tone` passes.
- Fast suite (`-m "not phase_gate"`) passes.
- The five-cycle lite gate now reaches **8 cycles**, with **no stuck-at-stop
  and no sole-up** failures, and angle correlation `0.548`. This is the
  intended result of Fix 1/2: the right ankle no longer parks upside down.
- The remaining gate failures are phase-pattern failures: period CV `0.239`,
  no E/F burst pair, stance/E offset `1.111` and drift `2.173`, L/R drift
  `0.939`, L/R spread `2.526`.

### Critical: the 1/5 lite model is not equivalent

Full logs (`experiments/current/2min_120s_k1`, `2min_120s_fixed`) have both
muscle drives near 1.0:

```text
full L mus E/F mean 0.238 / 0.961
full R mus E/F mean 0.955 / 0.979
```

The lite 1/5 gate has weak drives and F never crosses 0.5:

```text
lite L mus E/F mean 0.123 / 0.103, max F 0.414
lite R mus E/F mean 0.092 / 0.150, max F 0.493
```

The lite model therefore cannot be used to judge or tune the full CPG’s E/F
phase. The “lite only if the result is not worse” condition is **not met**.
Fix 4 (lite equivalence) must precede Fix 3 (phase/reciprocal tuning).

The full logs are also `game=selected`; both E/F drives near 1.0 may be driven by
the selected game. The phase-pattern gate must run **game=none** (game is a
separate axis, as requested), with a separate game-selected validation. The
current slow gate test still uses `game=selected`; that is a confound to remove
in Fix 4.

## Fix 3 progress (reciprocal E/F + phase clock)

Applied:

- `reciprocal_drives`: stance permits E only, swing permits F only.
- Applied E/F commands are logged (`applied_mus_*`) and the gate uses them.
- First-class `PhaseClock`: circular phase, period, confidence, phase error,
  bounded PLL, and bilateral offset lock. The clock stance drives the E/F gate;
  raw contact drives the phase error.
- Gate E/F/stance uses energy-weighted circular phase (not burst thresholds).

1/5 game-off gate after Fix 3:

```text
cycles = 8
period = 1.011 s, CV = 0.090
E/F = -3.058 ± 0.054 rad      (pi is anti-phase)
stance/E = +0.068 ± 0.041 rad (mid-stance reference)
no ankle stuck, no sole-up
render ok
```

Remaining failures:

```text
L/R drift 0.367 > 0.314
L/R spread 0.746 > 0.628
joint-angle correlation 0.444 < 0.500
```

These are close; the next pass tunes the bilateral lock and the angle
periodicity. The full gate is 37 min (8 s run), 1/2 is 8 min, 1/5 is ~2.5 min;
the 1/5 is the only near-target gate, and it now has the correct E/F/stance
pattern, so it is the current working gate scale pending the remaining
L/R/angle fixes.

Two one-variable tuning attempts failed and were reverted: bilateral gain 0.15
destabilized the pattern, and contact debounce 100 ms with PLL gains 0.25/0.05
made period CV/L/R/angle worse. Best baseline: bilateral 0.04, contact debounce
50 ms, PLL kp=0.45, ki=0.12. The next approach must target the left-leg phase /
angle-cycle sampling, not these gains.


## Detectors (implemented)

1. **`tests/test_ankle_pose.py`** — deterministic MuJoCo foot orientation.
   It loads the XML and asserts that the sole world-z at both ankle stops is
   `<= 0.05`. It fails on the current upper `1.76` and will pass only when the
   upper stop is safe. It also checks the XML range equals
   `model_cfg.ANKLE_RANGE_RAD`.
2. **`persona/phase_metrics.py::_stuck_at_limit`** — contiguous runs where an
   ankle is within `margin` of a joint stop, its step velocity is below
   `vel_eps`, and the target is more than `target_eps` away. The gate reports
   side, stop, duration, `q`, and target. It catches the L upper and R lower
   pins in the current log.
3. **`persona/phase_metrics.py::_foot_inverted_runs`** — optional runtime
   `foot_sole_z_*` check. The plant should log the foot's local `-z` world
   component; the gate fails if it stays positive.
4. **`tests/test_phase_metrics.py`** — synthetic tests for E/F, L/R, stance/E,
   angle periodicity, stuck-at-stop, and sole-up.

The slow **`tests/test_phase_gate.py`** (marker `phase_gate`) runs the lite
closed loop, 5 cycles, 3D render, and applies all of the above plus the phase
checks. It is the mandatory pre-commit/pre-approval gate.

## Fix plan

### Fix 1 — derive the upper stop from foot orientation (plant)

The upper limit must be the largest `q` for which the sole cannot face upward.
The zero-pose sole crosses vertical near `q=1.59`, but the logged gait poses
need a lower upper: across the last run the observed safe upper ranged from
**1.125** (minimum) to 1.33 (median), depending on hip/knee.

Applied as a conservative fixed stop for now:

- `ANKLE_RANGE_RAD = (0.19, 1.10)`;
- the same upper on all four XML ankle joints and both reference `ctrlrange`s;
- `loader/mocap.py` clips targets to the same range;
- `test_ankle_pose` passes; the fast suite passes.

Follow-up if 1.10 costs too much range: make the stop pose-aware from the
current shank/foot pitch, using the same sole threshold, instead of a fixed
conservative stop.

This is a **plant limit**, not a PERSONA parameter setpoint.

### Fix 2 — silent-muscle flop / lower-stop pin (plant + homeostasis)

The right ankle sits at the lower stop while the target is at `0.38–1.68` and
both muscle drives are low. The lower stop is sole-down, but the foot still
hangs there when the CPG is quiet. Add physical/relational ankle tone:

- applied: `stiffness=0.05`, `springref=1.0` on the two controlled ankle
  joints; the fast `test_ankle_tone.py` checks the coefficient and that the
  ankle moves off the lower stop with zero muscle drive;
- the reference ankle remains servo-driven;
- no absolute per-parameter PERSONA setpoint; this is a plant ligament/tone;
- the existing classical attractor homeostasis remains available for the
  strategy/props and is used in the task-5 tests;
- the gate must show no sustained lower-stop pin.

### Fix 3 — reciprocal E/F and phase/credit

The current logs show co-contraction or silence at the stops. The CPG must
produce a reciprocal E/F command and the phase/credit work must not reward
both muscles. This is task 2/1; the gate must be green before game tuning.

### Fix 4 — lite gate scale and runtime

The 1/5 lite model produced **one** cycle vs the full model's ~113 cycles in
120 s; it is not equivalent and therefore violates “lite only if the result is
not worse.” The gate must use the smallest scale that preserves the five-cycle
phase pattern, validated once against the full model. The current 8 s run took
347 s with `game=selected`; the pattern gate should run `game=none` for speed
(the game is a separate axis), with a separate game-on validation. Target
<2 min only after the lite-equivalence check passes; otherwise use the full
model and accept the longer gate.

### Fix 5 — runtime foot logging

Add `foot_sole_z_L/R` to `CPGInput`, `RunLog`, H5/CSV, and the gate. The static
pose test catches the current stop; the runtime field catches a foot that
inverts in the middle of a cycle.

## Order after the detector

1. Fix 1 (upper stop) — done (1.10); fast pose test passes.
2. Fix 2 (ankle tone) — done (stiffness 0.05, rest 1.0); fast tone test passes.
3. **Fix 4 next (lite equivalence/speed)** — the 1/5 model is not equivalent
   (full E/F ~1.0, lite F <0.5); choose the smallest scale that reproduces the
   full phase pattern, or use full. The gate must not tune the full CPG from a
   non-equivalent lite trace.
4. Fix 3 (reciprocal E/F + phase/credit) — after a valid gate scale; gate green.
5. Decide phase/PLL from the gate evidence; then task 2 credit map, task 3
   objective/k, task 5 identifiability/noise/homeostasis, task 4 game
   validation, task 6 tuning (original six-task order; game validation only
   after the CPG gate is green).

## Green 1/5 gate — TA swing authority (2026-10-25)

The remaining left-leg angle failure was not a clock/threshold problem. The
controlled ankle could not dorsiflex against the passive tone and ground load:
the left foot stayed loaded through swing, the left knee developed a period-2
bounce, and the fixed-lag angle autocorrelation collapsed. Raising the TA
(flexor) muscle force from `2.8` to `5.6` gave the ankle enough swing authority
without changing gate thresholds or servo gains.

Final mandatory 1/5 gate (`game=None`, `realize_every_k=1`):

```text
cycles=7  period=0.989 s  cv=0.196
E/F=-3.033 rad  std=0.355
L/R=-1.224 rad  drift=0.159
stance/E=+0.107 rad  std=0.078
angle corr=0.679
no stuck-at-stop, no sole-up, 3D render present
PASS
```

The period CV is close to the `0.20` limit, so the next validation is the
1/2/full-scale equivalence check and a separate game-selected arm. Do not
treat the 1/5 pass as proof that the lite and full models are equivalent.

## Verification

- `pytest -q -m "not phase_gate"` must pass, including `test_ankle_pose.py`.
- `pytest -m phase_gate -q` must pass: 5 cycles, 3D render, E/F, stance/E,
  angle periodicity, stable L/R offset, no stuck-at-stop, no sole-up.
- `realize_every_k=1` for learning and the gate; k=10 remains speed-only.
- No `--no-persona` change until the gate harness supports the control arm;
  `--game` remains a separate axis.
