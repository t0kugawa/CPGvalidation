# PERSONA algorithm comparison — first empirical report

**Date:** 2026-10-26
**Spec:** `docs/superpowers/specs/2026-10-26-persona-algorithm-comparison-design.md`
**Plan:** `docs/superpowers/plans/2026-10-26-persona-algorithm-comparison.md`

## 1. Scope and fairness

- **Primary testbed:** the 14-parameter toy spiking CPG
  (`benchmarks/compare/toy_cpg.py`), 5 seeds, **200 episodes × 100 steps =
  20,000 environment steps** per arm.
- **Online arms** use the same `PersonaParam` exploration, `FisherMatrix`
  recurrences, and the same per-step objective. `persona_game` uses four
  synthetic players; `persona_diag`/`persona_block` use the benchmark
  preconditioner variants with the game off.
- **Offline arms:** Optuna CMA-ES (200 trials) and a soft-spike surrogate
  gradient (200 BPTT epochs, Adam). The surrogate gradient is checked against
  finite differences on the soft relaxation (relative error < 1e-4).
- **Held-out evaluation:** every arm's final or best-seen vector is scored on
  a common evaluation episode (seed 10,000, tail 50%). Online arms are unstable
  at the end, so both **best-seen** and **final** held-out losses are reported;
  CMA-ES/surrogate return their best vector, so best = final for them.
- **Secondary testbed:** the NEST fast CPG, 400 steps, seed 0, H off, for
  `persona_manual`, `persona_nogame`, `persona_diag`, `persona_block`, and
  `persona_evo` (the saved P=24 territory partition from Task 1).
- No production default changed. `cmaes` (and `optuna` in WSL) are
  benchmark-only dependencies.

## 2. Toy comparison (5 seeds)

Mean held-out tail loss (lower is better; std in parentheses):

| arm | best-seen held-out | final held-out | best training curve | mean wall |
|---|---:|---:|---:|---:|
| CMA-ES | **0.00296** (0.00081) | **0.00296** | **0.00296** | 4.6 s |
| surrogate gradient | 0.00462 (0.0) | 0.00462 | 0.00232 | 25.8 s |
| PERSONA block | 0.01311 (0.00572) | 0.02835 | 0.01292 | 27.6 s |
| PERSONA diag | 0.02218 (0.00788) | 0.04007 | 0.01193 | 24.2 s |
| PERSONA game | 0.06437 (0.08638) | 0.03836 | 0.02442 | 38.6 s |
| PERSONA no-game | 0.06980 (0.08319) | 1.64061 | 0.02276 | 25.0 s |

**Reading.** CMA-ES is the clear winner on this toy: 4.6 s for 200 episodes and
a 4.5× lower best-seen loss than the best PERSONA variant. The surrogate
gradient is close (0.0046) and is the best *online-style* method in cost: 25.8 s
for 200 epochs. Among PERSONA arms, the block-diagonal preconditioner is the
best (0.0131), then diagonal (0.0222); the full Fisher is weaker and the game
hurts (0.0644/0.0698) and is high-variance. The no-game final policy blows up
on one seed (final 1.64), which is exactly the online instability the toy
exposes.

**Caveats.** The online arms are scored by **best-seen** because the final
policy is unstable; the final numbers are included for honesty. The toy's
soft-spike surrogate is not the hard spiking forward: the surrogate is a
relaxation used for BPTT, then evaluated on the hard toy. The toy's loss is
dominated by a plant-target mismatch; it is not a biological CPG benchmark.

## 3. NEST fast CPG comparison (400 steps, seed 0)

| arm | loss_eval | r_p90 | at_bound | q>1 | violations | entropy | phase gate | bio accept | duty in dist |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `persona_manual` | 0.272 | 0.00348 | 3 | 2 | 0 | 0.528 | FAIL | False | True |
| `persona_nogame` | 0.163 | 0.0457 | 30 | 5 | 0 | — | FAIL | False | True |
| `persona_diag` | 0.312 | 0.000273 | 20 | 0 | 0 | — | FAIL | False | True |
| `persona_block` | 0.276 | 0.000281 | 10 | 0 | 0 | — | FAIL | False | True |
| `persona_evo` (P=24) | **0.147** | 0.00496 | 10 | 0 | 0 | 0.694 | FAIL | False | True |
| `cmaes_sub` (100-step smoke, 2 trials) | 0.282 | — | — | — | — | — | — | — | — |

**Reading.**

- The **Evo partition** gives the best loss (0.147), no `q>1` parameters,
  10 at-bound parameters, and the highest stack entropy (0.694). It is the
  best applied result in this one-seed comparison.
- With the manual 141-segment map, the **game** reduces saturation (`q>1` 5→2,
  at_bound 30→3) at a loss cost (0.163→0.272). That is the conserved-stack
  trade-off, visible on the CPG.
- `diag`/`block` preconditioners reduce saturation strongly (`q>1` 0,
  at_bound 20/10) but cost loss (0.312/0.276). They are stabilizers, not
  loss winners.
- `cmaes_sub` is only a 100-step, 2-trial smoke; it is not a fair CPG
  comparison and is reported as a wiring check, not a result.

## 4. Bio-plausibility table

| criterion | manual | no-game | diag | block | evo |
|---|---|---|---|---|---|
| phase gate | FAIL | FAIL | FAIL | FAIL | FAIL |
| biological acceptance | False | False | False | False | False |
| angle coverage | low | low | low | low | low |
| duty factor in distribution | True | True | True | True | True |
| type/sign violations | 0 | 0 | 0 | 0 | 0 |
| at_bound | 3 | 30 | 20 | 10 | 10 |
| q>1 | 2 | 5 | 0 | 0 | 0 |
| sole-up | not measured (not logged) | same | same | same | same |

The phase gate fails for **every** arm; the failures are the known CPG issues
(stance/E offset, L/R drift, joint-angle cycle correlation). This is not an
algorithm result and must not be presented as one. `RunLog` does not record
`foot_sole_z_*`, so sole-up is "not measured", not a pass. The bio comparison
therefore supports only the loss/stability claims in §3, not a bio-plausibility
claim.

## 5. Winners per metric

| metric | winner | note |
|---|---|---|
| toy best-seen loss | CMA-ES | 0.00296 |
| toy wall-time efficiency | CMA-ES | 4.6 s |
| toy among PERSONA variants | block | 0.0131 |
| CPG loss | evo partition | 0.147 |
| CPG saturation (`q>1`, at_bound) | diag/block/evo | all q>1=0; evo/block at_bound 10 |
| CPG participation/entropy | evo | 0.694 |
| phase gate / bio acceptance | none | all fail |

## 6. Interpretation and next steps

1. **CMA-ES is a strong offline baseline on a 14-D deterministic toy.** This is
   expected and should be reported, not hidden.
2. **PERSONA's online one-step natural gradient is noisy on this toy.** The
   best-seen metric is competitive with the surrogate only for the block
   variant; the final policy is unstable. The toy should be made smoother or
   the online advantage should be windowed before drawing a general conclusion.
3. **The block preconditioner is the best PERSONA variant on the toy**, and it
   strongly reduces CPG saturation. The game helps CPG saturation but hurts
   the toy; the Evo partition helps both loss and saturation on the CPG.
4. **Phase/bio work is separate.** No algorithm arm passes the phase gate;
   tune the CPG/phase loop before using these arms for a bio-plausibility
   claim.
5. **Next:** run the CPG comparison with 2–3 seeds and 800–2000 steps; run
   `cmaes_sub` with a real trial budget and top-16 pilot; add a held-out
   validation episode for online early stopping; and test a smoother toy or a
   longer credit window for PERSONA.

## 7. Reproduce

```powershell
# toy, 5 seeds
python bench_persona_compare.py --toy --seeds 0 1 2 3 4

# toy smoke
python bench_persona_compare.py --smoke

# CPG, 400 steps, one seed
python bench_persona_compare.py --cpg --seeds 0 --cpg-steps 400 \
  --arms persona_manual persona_nogame persona_diag persona_block persona_evo \
  --player-map benchmarks/evo/reduced_0_24_territory_partition/final_partition.json \
  --out benchmarks/compare/cpg
```
