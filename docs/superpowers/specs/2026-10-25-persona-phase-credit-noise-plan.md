# PERSONA phase gate, phase/credit, identifiability, and game plan

**Date:** 2026-10-25
**Status:** recorded for review. Tests-first phase is in progress; the
mandatory gate is currently red, so nothing is committed/approved.
**Related:**
`docs/superpowers/plans/2026-10-25-ankle-inversion-stuck-fix.md`,
`docs/superpowers/specs/2026-10-24-mujoco-ankle-geometry-fix.md`.

## 0. Mandatory five-cycle phase gate

The gate is one behavioral contract, run before every phase/credit/game
commit or approval. It is not a reference-tracking test.

**Run**

- Lite CPG: 1/5 of production neurons in every site (initial target).
- MuJoCo closed loop.
- PERSONA **on by default**.
- `--no-persona` is an explicit opt-out for a control arm.
- `--game` is a separate axis (`none` / `selected`), not implied by PERSONA.
- Five complete stance/swing cycles.
- 3D render required.
- Target wall clock: under ~2 minutes, **conditional** on the lite result not
  being worse than full.

**Checks**

- stable period and at least five complete cycles per side;
- E/F anti-phase within a limb;
- stance/swing onset aligned with the E burst;
- periodic hip/knee/ankle joint traces;
- a **stable, non-zero L/R offset** — not forced to π;
- no ankle pinned at a joint stop while the target is far away;
- no foot sole pointing upward (the ankle-inversion detector);
- the 3D render exists and is non-empty.

**No reference matching.** Reference comparison and optimization are PERSONA
validation jobs and require longer runs.

**Lite-equivalence check.** A one-time lite-vs-full comparison on the phase
metrics. The lite model is acceptable only if it preserves the five-cycle
pattern. Current evidence: after Fix 1/2 the 1/5 lite gate reaches 8 cycles
with no stuck/sole failures, but its motor drives differ sharply from full
(full E/F ~1.0; lite F max <0.5). **1/5 is not accepted.** The gate scale must
be raised or the full model used; the 2-minute target is conditional on
equivalence.

**Current implementation (tests first)**

- `persona/phase_metrics.py` — period, E/F, stance/E, L/R, angle, stuck-at-stop,
  sole-up metrics.
- `tests/test_phase_metrics.py` — synthetic metric tests (pass).
- `tests/test_ankle_pose.py` — fast MuJoCo sole-orientation test (passes after
  the upper stop was lowered to 1.10).
- `tests/test_ankle_tone.py` — fast passive-tone test (passes after
  `stiffness=0.05`, `springref=1.0`).
- `tests/test_phase_gate.py` — slow five-cycle lite gate (`phase_gate` marker),
  currently fails on phase metrics: 8 cycles, no stuck/sole, period CV 0.239,
  no E/F pair, stance/E and L/R drift.
- `pytest.ini` — `phase_gate` marker.

**Gate 0 status:** implemented; ankle Fix 1/2 green; phase gate red; lite 1/5
not equivalent; no commit/approval. The first gate log is
`archive/phase_gate_current/`; the 3D render is
`locomotion_oblique_view.mp4`.

### Production run modes (backlog)

Formalize `full`, `1/2`, and `1/5` as first-class production run modes across
`run_2min.py`, `run_validation.py`, and `personalize.py`. They must be
orthogonal to `--persona/--no-persona`, `--game`, seed, `realize_every_k`, and
render, with one central size/provenance config. Current gaps: `run_2min`
hardcodes full, `run_validation` has `--debug-small`, `personalize` has
small/normal, and the gate has its own LITE 1/5. Deferred; recorded for the
next production pass.

**Game-axis confound:** the full logs are `game=selected`, and both E/F drives
are near 1.0 there, while the lite gate has F max <0.5. The phase-pattern gate
should run `game=none` (game is a separate axis); game-selected validation
belongs with task 5. The slow gate test currently uses `game=selected`; remove
that in Fix 4.

**Fix 3 progress:** reciprocal phase-gated E/F, applied-command logging, and a
first-class `PhaseClock` (period/confidence/error/PLL/bilateral offset) are in.
The 1/5 game-off gate now passes cycles (8), period CV (0.090), E/F
(-3.058 ± 0.054), stance/E (0.068 ± 0.041), stuck/sole, and render. Remaining:
L/R drift 0.367, L/R spread 0.746, angle correlation 0.444. No commit yet.

## 1. Phase/frequency/bilateral state

Add a first-class circular phase state per limb:

- phase, period, confidence, phase error;
- derived from E/F bursts and contacts;
- logged and available to the loss, advantage, credit, and game.

Add a **bounded explicit phase-error / PLL correction** for self-entrainment and
a **bilateral frequency/offset stabilizer**. The L/R offset is stabilized, not
forced to half a cycle.

**Decision point:** whether a PLL is needed, and its exact form, is decided
after the current-model tests and gate evidence. The gate must pass before any
credit or game tuning. Reference-phase locking remains a PERSONA/long-run
feature; the gate uses self-consistency.

## 2. Credit per limb/phase

Replace `g = G.sum(axis=1)` with an explicit player map. Each parameter belongs
to a **limb × phase × module** player; cross-limb anti-correlated advantages are
never summed.

**User clarification:** every player should be a **separated SNN site** building
a “physical” map of the SNN. The player map must be derived from the actual
NEST populations/connections/sites, not an abstract grouping. The game must use
the same player IDs.

Tests:

- synthetic opposite-sign L/R advantages do not cancel;
- each site maps to exactly one player;
- player responsibilities are explicit and auditable;
- game and strategy player IDs agree.

## 3. Objective and k

- Keep a smooth tracking loss.
- p90 remains a diagnostic, not the objective.
- `realize_every_k=1` is the learning path.
- `k=10` is preserved as an experimental speed mode only; it is not a default,
  not a gate, and not used for learning/validation. A matched 2 s test showed
  last-10 loss `0.276` at k=1 vs `1.001` at k=10, with the right ankle at the
  wrong stop.

## 4. Identifiability, structural freezes, homeostasis, noise

- Audit influence/Fisher and parameter identifiability.
- Freeze structural constants, including the C1 flexor seed and the
  Renshaw/M-F constraints.
- Group sites into modules/players where appropriate, respecting the physical
  SNN site map from task 2.
- The classical attractor-based homeostasis is already implemented; use it in
  tests and as an optional regulator.
- Full noise audit:
  - all recorders;
  - no recorders;
  - specific recorder sites;
  - self-noise;
  - model-noise.
- Exact noise estimates on the strategy; recorder-consistency tests
  (recorded = applied, no stale deltas, no hidden state); rebuild the recorder
  logic if the audit finds a mismatch.

## 5. Game validation

Only after the CPG gate is green:

- longer runs (at least 120 s);
- multiple seeds and references from `data/`;
- per-player resource/entropy distributions, not only active counts;
- zero-sum binding diagnostics;
- no seed losing more than 10%;
- `q > 1` not worsening.

## 6. Game tuning

Last. Tune against the multi-seed/multi-reference results, not the 20 s
benchmark. Avoid overfitting the small benchmark.

## Order

Fixed prefix: **0 gate → 1 phase → 2 credit → 3 objective/k**. The original
order then puts **game validation (4) before identifiability/noise/homeostasis
(5)** and game tuning (6) last. The 2026-10-25 summary at the top of this
request listed identifiability as 4 and game validation as 5; the earlier
instruction and the explicit “keep the original order” answer put game
validation at 4. This is recorded as an open numbering ambiguity; do not start
4/5 until the tests decide, and confirm the number then.

## Immediate sequence

1. Keep the current-model test evidence.
2. Fix 1: upper ankle stop from foot orientation — done (1.10); fast
   `test_ankle_pose` passes.
3. Fix 2: ankle tone/attractor — done (stiffness 0.05, rest 1.0); fast tone test
   passes.
4. **Fix 4 next: lite equivalence and gate runtime.** The 1/5 lite is not
   equivalent to full (full E/F ~1.0; lite F <0.5). Use the smallest scale that
   preserves the full pattern, or full. Game-none fast gate; game-selected
   separate.
5. Fix 3: reciprocal E/F and phase/credit; gate green.
6. Then decide phase/PLL, fine-grained credit, objective/k, and the rest in the
   recorded order.

## Constraints

- No absolute per-parameter setpoints, target bands, or trust regions in
  PERSONA. Type/sign identity clamps are allowed.
- Stability is relational; p90 is a diagnostic.
- One CPG and one MuJoCo.
- `realize_every_k=1` for learning.
- Do not push unless asked.
- A commit/approval requires the mandatory gate and the fast tests to pass.
