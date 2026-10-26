# PERSONA algorithm comparison: toy CPG, CMA-ES, natural-gradient variants, surrogate gradient — design

**Status:** decisions approved 2026-10-26. Benchmark-only; no production
default change and no change to the PERSONA model.

**Spec index:** `docs/superpowers/specs/2026-10-26-persona-algorithm-comparison-design.md`

## 1. Intent

Compare the updated PERSONA approach against three families on the same
personalization problem:

1. **Optuna CMA-ES** — offline black-box search;
2. **another natural-gradient approach** — diagonal Fisher and block-diagonal
   per-player Fisher, in the same online loop;
3. **surrogate gradient** — BPTT through a small spiking CPG with a surrogate
   derivative.

Primary testbed: a small differentiable spiking CPG toy where all algorithms
optimize the same parameter vector under the same budget. Secondary testbed:
the current NEST fast CPG for PERSONA/Evo/NG/CMA-ES; surrogate gradient is not
applicable to NEST and is not claimed for it.

The comparison is descriptive. No algorithm is declared the winner in advance;
the report states where each wins and at what cost.

**The Evo arm is a saved partition, not a full Evo run.** `persona_evo` means
"`persona_game` running on the final partition produced by the separate
PERSONA-Evo task" (`--player-map PATH`). The full generation loop, the
manual/reference/Evo partition comparison, and ARI/NMI belong to the Evo
benchmark, not here. If no saved map is supplied, the arm is the current
137-segment `PhysicalMap.from_registry` map and is labeled `persona_manual`.

## 2. Testbeds

### 2.1 Toy differentiable spiking CPG (primary)

Location: `benchmarks/compare/toy_cpg.py` (benchmark-only; never imported by
the production CPG path).

Architecture, discrete time `dt = 0.01 s`, episode `T = 100` steps (1 s):

- two rhythm-generator populations (E/F), `N_rg = 8` LIF neurons each;
- reciprocal inhibition E→F and F→E;
- tonic drive to E and F;
- two motor pools (E/F), rate readout over a 10 ms window;
- one agonist/antagonist muscle pair driving a 1-DOF ankle plant
  (`I q̈ + c q̇ + k q = τ_muscle`);
- a sinusoidal target ankle trajectory with a fixed phase and amplitude;
- observation noise optional and seeded.

Optimized parameter vector (14, all bounded, same for every algorithm):

```
w_tonic_e, w_tonic_f,
w_rec_e, w_rec_f,
w_e_to_f, w_f_to_e,
w_rg_to_mn_e, w_rg_to_mn_f,
tau_m_e, tau_m_f,
bias_e, bias_f,
gain_e, gain_f
```

LIF forward:

```
v[t+1] = v[t] + dt/tau * (-v[t] + I[t])
spike   = v[t+1] >= 1
v[t+1] = 0 where spike
```

`I[t]` includes the tonic drive, recurrent and reciprocal weights, and bias.
The muscle activation is the MN spike rate times `gain`; the plant is
integrated with the same `dt`.

The strategy needs two advantage components. The toy's loss vector is

```
L_E = mean over E-dominated steps of (q_model - q_target)²
L_F = mean over F-dominated steps of (q_model - q_target)²
L_vec = [L_E, L_F]
L_effort = mean_t (tau_muscle[t])²
L_total = mean(L_vec) + 0.01 * L_effort
```

`E-dominated`/`F-dominated` are the steps where the RG-E/RG-F population rate
is larger. The strategy phase label is the tuple `("E", "F")`, so each
component keeps its own baseline; there is no stance/contact claim in the toy.
The reported metric is the **tail-window** loss over the last 50% of the
evaluation episode; the offline arms optimize that same tail-window loss.

One seed = one initial parameter vector (paired across arms) plus a fixed
target/plant and an observation-noise seed. The initial vectors are generated
from the seed and saved. The toy is deterministic for a given vector and seed.

An episode is 100 steps (1 s). Every arm gets **200 episodes = 20,000
environment steps** in the main comparison. Online arms keep one strategy
across the episode sequence; the plant resets each episode. Offline arms get
200 trials/epochs. The final parameter vector of every arm is re-scored on a
common held-out evaluation episode (same target, tail window) so the headline
loss is not an artifact of a different objective.

### 2.2 NEST fast CPG (secondary)

The current fast CPG path (`build_personalizer`, `FAST_SIZES`, 400/800 steps,
`eps_pose=0.10`, `pose_hold_steps=3`, `loss_stop=-1.0`, `ref_hz=700`,
`resolution_ms=0.2`, game selected/off as the arm requires). It is the same
path used by `bench_game.py` and `bench_homeostasis.py`.

Surrogate gradient is **not** run on NEST: there is no differentiable path
through the spiking network. The report says so explicitly. If a
differentiable CPG model is wanted later, it is a new model, not this
benchmark.

## 3. Algorithms

All online arms use the same `PersonaParam` exploration, the same per-step
loss, the same `FisherMatrix` accumulation, and the same game hook when the
arm enables it. The only difference is the preconditioner or the update rule.

### 3.1 `persona_game` — updated PERSONA

- `FisherMatrix` with `F/Xbar/G`, recursive inverse, σ rule, `lr_mu`.
- `StackGame` with `GameConfig.current_best()` on the arms that need it.
- On the toy, the game uses four synthetic players (RG-E, RG-F, motor,
  plant/feedback) built with `PhysicalMap.from_specs`; on the CPG it uses the
  saved Evo partition (or the registry map if none).
- `persona_nogame` is the same strategy with `game=None`.
- **H is off in every arm.** "Updated PERSONA" here means the game/Evo layer,
  not `HomeostaticConfig`. This keeps the preconditioner comparison free of
  H's credit adjustment and halves the CPG runtime.

### 3.2 `persona_diag` — diagonal Fisher natural gradient

A benchmark-only `VariantFisherMatrix` subclass. It keeps `F/Xbar/G` and the
same score/trace but returns `P_diag = diag(1/(diag(F)+λ))` for the μ step.
This is the cheapest natural-gradient approximation and the direct test of
whether the full Fisher coupling matters. It is **game-off**, to isolate the
preconditioner from the game.

### 3.3 `persona_block` — block-diagonal per-player Fisher

The same subclass with `P_block`: zero all off-block entries of `F` using the
player site groups, invert each block with the same ridge, and apply. It tests
whether cross-player coupling is necessary once players exist. On the toy the
blocks are the four synthetic players; on the CPG they are the Evo/manual
player groups. It is **game-off**, for the same reason as `persona_diag`.

### 3.4 `cmaes` — Optuna CMA-ES

- Optuna `CmaEsSampler` (`optuna.samplers.CmaEsSampler`) with the small
  `cmaes` package.
- Parameter suggestions use the same bounds and initial vector.
- One trial = one full episode (`T` toy steps, or `steps` CPG steps) with the
  trial's parameter vector held fixed.
- Objective = the tail-window loss (the same metric the online arms report).
- Budget: 200 trials on the toy = 20,000 environment steps, matching the
  online arms. On the CPG, `cmaes_sub` uses the top-k sites (k=16) from a
  **single pilot PERSONA game-off run at seed 0**; the pilot's steps and wall
  time are reported separately and are not counted in the CMA-ES budget.
  `cmaes_full` gets a small fixed trial budget (default 16) to show the
  dimensionality cost.
- CMA-ES is offline: it does not update during an episode.

### 3.5 `surrogate` — BPTT with a surrogate derivative

Benchmark-only, pure numpy:

- forward the same toy LIF network for `T` steps;
- backward through the membrane dynamics with a fast-sigmoid surrogate
  `d spike / d v = α / (1 + α|v|)²` (α=10);
- optimize the same 14 parameters with Adam (implemented in the benchmark);
- one epoch = one episode; batch = 1; **200 epochs**, matching the 20,000
  environment steps of the other arms;
- objective = the same tail-window loss;
- gradient check on a 3-parameter subset against finite differences
  (relative error < 1e-4) is a required test.

This is a genuine surrogate-gradient method on the spiking toy. It is not run
on NEST.

## 4. Fairness and budgets

- Within a seed, the same initial parameter vector, bounds, target, plant, and
  seed are used by every arm. Across seeds, both the initial vector and the
  observation-noise stream change; the initial vector is saved per seed.
- Same parameter vector definition and ordering.
- Online arms: one environment step per update. Offline arms: one episode per
  trial/epoch. Headline parity is **20,000 environment steps** for every arm;
  wall time is reported separately.
- Report both **environment steps** and **wall time**. A win at matched wall
  time and a win at matched environment steps are different claims; the report
  keeps them separate.
- Toy: 5 seeds per arm. CPG: 2 seeds per arm for screening, 3 seeds for the
  finalists. Seeds are paired across arms.
- No early stopping that favors one arm; the horizon is fixed per arm and
  reported.
- CMA-ES trial budget and PERSONA step budget are both printed in the report;
  no arm gets a hidden multiplier. The `cmaes_sub` pilot run is reported
  separately and is not counted as a CMA-ES trial.

## 5. Benchmarks and metrics

### 5.1 Toy metrics

- final loss on the common held-out evaluation episode, tail window (last 50%
  of the episode);
- area under the learning curve (AUC) over the 20,000-step budget;
- episodes to reach a fixed loss threshold (threshold reported);
- success rate over seeds (held-out tail loss below threshold);
- wall time and held-out loss per wall-second;
- parameter error to the best-known vector (from the best arm, clearly
  labeled as an empirical reference, not ground truth).

### 5.2 CPG fast metrics

Reuse the existing metrics:

- `bench_homeostasis.inflation_metrics`: `loss_eval`, `r_p90`, `r_median`,
  `r_max`, `r_gt_0_25`, `q_gt_1`, `at_bound`;
- `type_sign_violations`: must be zero;
- `phase_metrics.phase_gate_metrics`: cycles, period CV, E/F offset,
  stance/E, L/R drift/spread, angle correlation, ankle stuck, render presence
  when rendered. **Sole-up is not measured** because `RunLog` does not record
  `foot_sole_z_*`; the report says "not measured (not logged)" rather than
  claiming a pass by omission. Adding that field to production logging is out
  of scope for this benchmark-only task;
- `personalize.acceptance_from_log`: `biological_acceptance` (acceptable,
  angle coverage, duty factor in distribution);
- `loader.stats.LocomotionMetrics`: duty factor L/R, double support,
  step/stride length, synergy contributions when available;
- co-contraction fraction: fraction of steps with both E/F applied drives
  near 1.0 (a known CPG issue; reported, not hidden);
- active players, stack entropy, fraction min/max for game arms.

### 5.3 Bio-plausibility table

Report a per-criterion table, not an invented composite scalar:

| criterion | value | pass/fail |
|---|---|---|
| phase gate | all sub-metrics | pass/fail |
| biological acceptance | acceptable, coverage, duty | pass/fail |
| locomotion metrics | duty L/R, double support, step/stride | in distribution |
| type/sign violations | count | must be 0 |
| bounds | at_bound, q>1, r_p90 | reported |
| co-contraction | fraction | reported |

If a single rank is needed for a figure, use "number of passed criteria" and
label it as a convenience, not a scientific score.

## 6. Harness

`bench_persona_compare.py` at the repository root:

```
python bench_persona_compare.py --smoke
python bench_persona_compare.py --toy --seeds 0 1 2 3 4
python bench_persona_compare.py --cpg --seeds 0 1 --steps 800 \
    --arms persona_manual persona_nogame persona_diag persona_block cmaes_sub
python bench_persona_compare.py --cpg --player-map benchmarks/evo/final_partition.json \
    --arms persona_evo
python bench_persona_compare.py --cpg --arms cmaes_full --cmaes-trials 16
```

`persona_manual` uses the 137-segment registry map; `persona_evo` requires a
saved `--player-map` and is otherwise an error. The full Evo generation loop is
not run by this harness.

Outputs under `benchmarks/compare/`:

- `toy/runs.csv`, `toy/summary.csv`, `toy/summary.json`, `toy/curves.png`;
- `cpg/runs.csv`, `cpg/summary.csv`, `cpg/summary.json`, `cpg/bio.md`,
  `cpg/phase.md`;
- `report.md` with the fair-budget statement, the winner per metric, and the
  caveats.

The harness reuses `bench_homeostasis` and `phase_metrics`; it does not
duplicate their metric code. Fresh-process isolation is used for the CPG arms,
as in `bench_game.py`.

## 7. Tests

Fast, simulator-free:

- `tests/test_toy_cpg.py` — deterministic forward, bounded outputs, finite
  loss, target shape, parameter-order stability; `L_vec` is `[L_E, L_F]`;
  tail-window and held-out evaluation are deterministic; 200 episodes is
  20,000 steps.
- `tests/test_surrogate_gradient.py` — finite-difference gradient check on a
  3-parameter subset; relative error < 1e-4; BPTT is finite; 200 epochs is
  20,000 environment steps.
- `tests/test_compare_optimizers.py` — each optimizer runs one tiny iteration,
  returns a finite loss, respects bounds, and reports its budget; `persona_diag`
  and `persona_block` are game-off; every arm is H-off.
- `tests/test_compare_metrics.py` — bio table assembly, zero-violation
  handling, "sole-up not measured" label; no ARI/NMI (that belongs to the Evo
  spec).
- `tests/test_compare_harness.py` — `--smoke` toy run writes the expected files;
  CPG smoke is marked slow and runs 20 steps with PERSONA/NG only.

Slow, WSL CPGV:

- `tests/test_compare_cpg_smoke.py` — one CPG arm, 20 steps, finite metrics,
  files written. Not a performance claim.

## 8. Acceptance and reporting

- The toy comparison is the primary scientific comparison: all algorithms on
  the same parameter vector and budget, 5 seeds, with learning curves and
  wall time.
- The CPG comparison is the applied check: PERSONA/Evo/NG/CMA-ES on the real
  fast CPG; surrogate gradient is explicitly out of scope there.
- The report must state:
  - the exact budget of every arm;
  - which metrics favor which arm;
  - where CMA-ES was run on a reduced subspace and why;
  - that the toy is not the NEST CPG;
  - all bio-plausibility criteria, including failures;
  - any seed where an arm violates type/sign or fails the phase gate.
- No production default changes as a result of this benchmark.

## 9. Dependencies

- Already installed: `numpy`, `scipy`, `optuna`, `mujoco`, NEST (WSL CPGV).
- Add `cmaes` (small, pure Python) for Optuna's `CmaEsSampler`. No `torch`,
  `jax`, or scikit-learn.
- ARI/NMI are not used here; they are part of the Evo benchmark.

## 10. Non-goals and risks

- Not a claim that one optimizer is universally better.
- Not a differentiable NEST CPG; surrogate gradient is toy-only.
- Not a full hyperparameter sweep of every algorithm; the budgets are fixed
  and reported.
- CMA-ES on the full 269-site CPG is expected to be sample-inefficient; the
  reduced-subspace arm is a transparent compromise, not a fair full-space win.
- The toy's 14 parameters are much smaller than the CPG's 269 sites; the toy
  tests optimizer behavior, not CPG biology.
- The updated PERSONA arm includes the game/Evo only where the testbed has
  players; the toy's four synthetic players are not a physiological result.

## 11. Implementation order

1. Toy CPG model + determinism tests.
2. Surrogate gradient + finite-difference test.
3. Variant Fisher preconditioners (diag/block) + smoke tests.
4. Optuna CMA-ES arm + budget accounting.
5. Toy harness, 5-seed run, report.
6. CPG arms (PERSONA/Evo/NG/CMA-ES) + bio/phase tables.
7. Final report.
