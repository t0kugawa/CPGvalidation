# Deferred PERSONA work after the MuJoCo ankle fix

**Date:** 2026-10-24
**Status:** deferred; MuJoCo is the current source of truth.

The ankle model no longer permits the old inverted `-0.196` pose, and the SOL
moment arm no longer reverses. The next symptoms are in the controller and the
personalization model, not the plant. These are recorded so they are not lost.

## 1. CPG drive saturates both muscles at the physical stops

The 2 s selected-game check still ends with `mus_E = mus_F = 1.0` on the left
and the ankle at the physical upper stop. The plant can move both ways; the
controller is driving both muscles and the stronger SOL wins. This is the
first control issue to inspect after the plant fix:

- why both E and F saturate together;
- whether the CPG output should be a reciprocal E/F pair rather than two
  independent drives;
- whether the controlled hip/knee servos and the reference ankle servo have
  enough authority (`kp = 1.0` is low for a loaded limb);
- whether the loss should include a muscle co-contraction penalty.

Do not widen the plant stops or swap the muscle forces to hide this.

## 2. Phase/frequency entrainment

Right-leg phase agreement has been around `0.12` (left `0.62–0.64`). A
phase-conditioned advantage is only meaningful if the model is near the
reference phase. The reference-phase advantage fix is correct, but it does not
create entrainment. Make phase/frequency a first-class state (phase-locked
loop or explicit phase-error term) before tuning the game or the Fisher
strategy.

## 3. Per-limb / per-phase credit assignment

`g = G.sum(axis=1)` mixes players across anti-correlated limbs. In a gait,
left and right advantages often have opposite signs; summing cancels the
signal or creates common-mode noise. The game's players should be
limb-and-phase conditioned, and the payoff should not be summed across
anti-correlated modules. This is the most concrete structural issue in
PERSONA.

## 4. Game validation beyond the short benchmark

`prop_unc_05_a25` is selected on 20 s / 800-step windows. The next validation
needs:

- longer runs (at least 120 s) and multiple seeds/references;
- per-player resource and entropy distributions, not just active counts;
- whether the zero-sum simplex is binding (`at_bound` counts);
- whether the game fights co-adaptation when many parameters should rise
  together;
- the `prop_none` collapse as the control (7/137 active vs 110+).

The game is a promising relational regularizer, not yet a validated theory.

## 5. Objective and variance

The acceptance metric is p90, a tail statistic. Keep the online objective a
smooth tracking loss and keep p90 as a held-out diagnostic. One rollout per
step gives a high-variance advantage estimate; the game throttles updates but
does not reduce estimator variance. Consider paired/common-random-number
advantages or a stronger baseline when the plant/CPG costs allow it.

## 6. Identifiability and player structure

With ~554 sites, many redundant or symmetric, the Fisher matrix is
ill-conditioned and per-site advantage is noisy. The C1 flexor seed is still
learnable. Freeze structural constants (type/sign identity is allowed), group
sites into functional modules (phase/frequency, amplitude, bilateral coupling,
stiffness), and test whether fewer, structured players are more stable than
137 site-level players.

## 7. `realize_every_k` is rejected for learning

`--realize-every-k 10` gave a 3x speedup in the 0.5 s microbenchmark, but a
matched 2 s A/B on the fixed model degraded last-10 loss from `0.276` (k=1) to
`1.001` (k=10) and drove the right ankle to the upper stop. The 120 s k=10 run
ends with both E/F saturated at 1.0. Use k=1 for learning/validation; k=10 is a
speed-only probe until the stale-realization effect is understood.

## 8. Classical homeostasis with attractor parameters

If the game's zero-sum pressure fights co-adaptation, test a classical
homeostatic regulator with attractor parameters as a complementary plant/PERSONA
mechanism. Prefer relational/difference constraints over absolute per-parameter
bands, consistent with the accepted design.

## Immediate next step

After this MuJoCo fix is committed, run the 120 s selected-game validation with
one CPG/MuJoCo and `--realize-every-k 1`, then inspect:

- no ankle outside `[0.19, 1.76]`;
- the fraction of steps with both E/F saturated at 1.0;
- phase agreement L/R;
- last-window loss and the seed sweep.
