# MuJoCo ankle geometry and range fix

**Date:** 2026-10-24
**Status:** implemented on `mujoco_sim`
**Scope:** `dual_rat_hindlimb.xml`, `model/model_cfg.py`, `loader/mocap.py`,
`personalize.py`, `model/leg.py`, and ankle-focused tests.

## Symptom

The controlled ankle could sit upside down (about `-0.196` rad) or at the
opposite stop (about `+1.99` rad), with both muscle drives active, while the
target was around `1.2–1.7` rad. The reference ankle also lagged its target.
The 120 s logs showed:

```text
ankle_model_L = -0.196   ankle_target_L = 1.666
ankle_model_R = +1.992   ankle_target_R = 1.484
```

The old ankle joint range was `[-1.0, 2.6]`, then narrowed at runtime to the
raw mocap min/max of the run (`-0.196 … 1.992`). That is exactly where the
model parked.

## Root cause 1: SOL moment arm reversed at `q ≈ 1.42`

The controlled ankle is muscle-driven. The SOL insertion site was above the
ankle joint (`sol_insert_m_* = -0.009, ±0.007, +0.003`). At the target range the
SOL tendon moment arm changed sign:

| ankle `q` | SOL `dL/dq` (old) | TA `dL/dq` |
|---:|---:|---:|
| 1.00 | -0.0090 | +0.0090 |
| 1.42 | +0.0003 | +0.0069 |
| 1.66 | +0.0075 | +0.0056 |
| 1.90 | +0.0093 | +0.0042 |

With `q` increasing in plantarflexion, a plantarflexor must have `dL/dq < 0`.
Above `1.42` the SOL became a dorsiflexor; the TA was already a dorsiflexor.
Both muscles then pulled the ankle the same way. From `q=1.66`, full SOL drove
the ankle **down** to `1.414`, full TA to `-0.574`, and both to `1.276`: no
muscle command could hold the target.

The OpenSim reference inserts the SOL near the calcaneus. Moving the MuJoCo
insertion to the heel site (`-0.006, ±0.007, -0.0045`) keeps the SOL
plantarflexor across the range:

| ankle `q` | SOL `dL/dq` (fixed) | TA `dL/dq` |
|---:|---:|---:|
| 1.00 | -0.0069 | +0.0090 |
| 1.42 | -0.0075 | +0.0069 |
| 1.66 | -0.0073 | +0.0056 |
| 1.90 | -0.0065 | +0.0042 |

Full SOL from `1.66` now moves **up** to `2.372`; full TA still moves down to
`-0.574`. The controlled ankle is controllable.

## Root cause 2: raw mocap min/max opened a non-physical range

The runner set the joint limits from `min/max` of the marker-derived ankle
angles over the whole window. A few extreme frames therefore opened the joint
to `-0.196` and `+1.992`, far outside the OpenSim reference:

```text
OpenSim ankle_flx: default 0.87266463 rad, range [-0.52359878, 1.04719755]
MuJoCo neutral:    about 1.59 rad
offset:            1.59 - 0.8727 ≈ 0.71
physical range:    [0.19, 1.76] rad
```

The fix is a plant limit, not a PERSONA parameter setpoint:

- `model_cfg.ANKLE_RANGE_RAD = (0.19, 1.76)`.
- All four ankle joints and the two reference-ankle `ctrlrange`s use that range
  in `dual_rat_hindlimb.xml`.
- `loader/mocap.py` clips marker-derived ankle angles to the same range, so the
  target and loss do not chase non-physical extremes.
- `personalize.py` no longer widens the joint from the raw min/max; the old
  `ANKLE_RANGE_MARGIN_RAD` guard is removed.

## Verification

New `tests/test_mujoco_ankle.py`:

- SOL is plantarflexor and TA is dorsiflexor at `q = 0.3, 1.0, 1.4, 1.66, 1.9`.
- Full SOL from `q=1.66` lifts the ankle.
- All four ankle joints and the reference `ctrlrange` equal
  `ANKLE_RANGE_RAD`.

The tests fail on the old model (SOL at `1.66` and `1.9`, full-SOL direction)
and pass after the fix. The existing ankle-guard, mocap, adapter, and review
tests were updated for the physical range; the full suite passes.

A 2 s selected-game run
(`archive/check_fixed_2s_2s/`, ignored) gives:

```text
last10 loss 0.276   (old 2 s game-on: 0.4366)
ankle model L 1.5936 -> 1.7605; R 1.5541 -> 0.1899
no ankle below 0.187 or above 1.764
```

The old `-0.196` inversion is gone. The remaining saturation is visible in the
same run: at the end both `mus_E` and `mus_F` are `1.0` on the left, so the
CPG/PERSONA is still pushing the ankle into the physical upper stop. That is a
controller/credit problem, not a reason to widen the plant stops.

## Deliberately not changed

- **Muscle forces** remain `SOL 8.0`, `TA 2.8`. A temporary OpenSim-strength
  swap (SOL 1.4, TA 8.6) made a plant-only PD replay worse, so there is no
  evidence to change them yet.
- **No passive ankle spring** was added. The OpenSim coordinate stiffness is
  zero; the remaining saturation is driven by muscle commands, not a missing
  stop.
- **CPG drive, phase entrainment, credit assignment, and game tuning** are
  deferred; see `2026-10-24-deferred-persona-work.md`.

## Files

- `dual_rat_hindlimb.xml` — SOL insertion and physical ankle ranges.
- `model/model_cfg.py` — `ANKLE_RANGE_RAD`.
- `loader/mocap.py` — target clipping.
- `personalize.py` — removed the raw-min/max runtime guard.
- `model/leg.py` — `set_ankle_range` documentation; method retained for tests.
- `tests/test_mujoco_ankle.py` — new regression tests.
- `tests/test_ankle_guard.py`, `tests/test_mocap.py`, `tests/test_adapter.py` —
  updated for the physical range.
