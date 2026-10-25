# Ankle inversion / stuck-at-stop detector and fix plan

**Date:** 2026-10-25
**Status:** detector implemented; fix pending; the mandatory gate is red.
**Related:** `2026-10-24-mujoco-ankle-geometry-fix.md` (upper range is now
provisional), `2026-10-24-deferred-persona-work.md`.

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
From the model sweep, the sole crosses vertical near `q ~= 1.59`; use a safe
upper of about **1.55 rad** with the existing `<= 0.05` test. Exact value must
be computed and recorded, not guessed:

1. Sweep `q` with `tests/test_ankle_pose.py`'s `_sole_world_z`.
2. Pick `q_safe` such that `sole_world_z(q_safe) <= 0.05` with a margin.
3. Set `model_cfg.ANKLE_RANGE_RAD` upper, all four XML ankle joints, the
   reference-ankle `ctrlrange`, and `loader/mocap.py` clipping to `q_safe`.
4. Keep the lower stop at `0.19` for now; it is sole-down.
5. Update the MuJoCo spec addendum and the target/loss expectations.
6. The fast `test_ankle_pose` must pass; the gate must show no sole-up run.

This is a **plant limit**, not a PERSONA parameter setpoint.

### Fix 2 — silent-muscle flop / lower-stop pin (plant + homeostasis)

The right ankle sits at the lower stop while the target is at `0.38–1.68` and
both muscle drives are low. The lower stop is sole-down, but the foot still
hangs there when the CPG is quiet. Add a physical/relational ankle tone:

- preferred: a small passive ankle spring/damper (ligament tone) at a
  phase-neutral rest, or the existing **classical attractor homeostasis** on
  the ankle target/mean;
- no absolute per-parameter PERSONA setpoint; the attractor is relational and
  phase-conditioned;
- test: with zero muscle drive, the ankle does not remain pinned at a stop for
  more than one cycle; with the CPG, the stuck detector is clean.

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

1. Fix 1 (upper stop) + fast pose test; gate.
2. Fix 2 (ankle tone/attractor) + stuck test; gate.
3. Fix 3 (reciprocal E/F + phase/credit); gate.
4. Fix 4 (lite equivalence/speed); gate.
5. Then decide phase/PLL from the gate evidence; then task 2 credit map, task 3
   objective/k, task 5 identifiability/noise/homeostasis, task 4 game
   validation, task 6 tuning (original order; game validation only after the
   CPG gate is green).

## Verification

- `pytest -q -m "not phase_gate"` must pass, including `test_ankle_pose.py`.
- `pytest -m phase_gate -q` must pass: 5 cycles, 3D render, E/F, stance/E,
  angle periodicity, stable L/R offset, no stuck-at-stop, no sole-up.
- `realize_every_k=1` for learning and the gate; k=10 remains speed-only.
- No `--no-persona` change until the gate harness supports the control arm;
  `--game` remains a separate axis.
