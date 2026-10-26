# PERSONA-Evo — first empirical report

**Date:** 2026-10-26
**Spec:** `docs/superpowers/specs/2026-10-26-persona-evo-design.md`
**Plan:** `docs/superpowers/plans/2026-10-26-persona-evo.md`

## 1. Scope

This is the **small first pass**, not the full 800-step × 6-generation ×
2-seed matrix. It uses the fast CPG (`FAST_SIZES`, 141 segments / 273 sites in
the current tree), the conserved-stack game (`GameConfig.current_best()` with
`share_floor=0`), and one seed unless stated. The final partition is evaluated
on the same fast CPG with 400 steps.

Run artifacts:

- `benchmarks/evo/smoke/` — P=24, 1 generation, 400 steps, territory + random.
- `benchmarks/evo/` — P=24 and P=137, 3 generations, 400 steps, territory + random, partition stop.
- `benchmarks/evo/evaluation/` — manual 141-segment, reference Level-2, evo-territory and evo-random partitions at 400 steps.

The current tree has 141 segments / 273 sites (not the earlier 137/269
snapshot) because the working tree contains the reciprocal-inhibition edits.

## 2. Partition recovery

Fast-CPG 400-step generations, seed 0, 3 generations, `stop=partition`:

| P | init | final players | ARI | NMI | purity | V-measure | pairwise F1 | loss_eval | wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | territory | 21 | 0.154 | 0.564 | 0.454 | 0.564 | 0.203 | 0.148 | 611 s |
| 24 | random | 24 | 0.0008 | 0.412 | 0.270 | 0.412 | 0.047 | 0.176 | 436 s |
| 137 | territory | 84 | 0.054 | 0.651 | — | — | — | 0.259 | — |
| 137 | random | 91 | 0.038 | 0.657 | — | — | — | 0.183 | — |

**Territory initialization is clearly better than pure random at P=24**:
ARI 0.154 vs 0.0008, NMI 0.564 vs 0.412, purity 0.454 vs 0.270. At P=137 the
advantage is smaller but still positive (ARI 0.054 vs 0.038). The P=24 run
ends at 21 players, close to the 19 functional modules, but the absolute ARI is
modest: the game does not recover the 19-module reference exactly.

The P=137 runs are a different granularity control; their ARI is low because a
partition with 84–91 players cannot match 19 modules, and the NMI is inflated
by the large number of players.

## 3. What the P=24 territory players actually contain

Seed 0, generation 3, 21 players. Dominant reference label and purity:

| player | n | dominant label(s) | purity | comment |
|---:|---:|---|---:|---|
| 0 | 5 | L_rg_ext (3) + global_drive (2) | 0.60 | BS_E and BASE→RGE-L absorbed into left extensor |
| 1 | 4 | R_rg_ext (3) + global_drive (1) | 0.75 | BASE→RGE-R absorbed into right extensor |
| 2 | 11 | R_commissural (3), L_rg_flex (2), L_commissural (2) | 0.27 | left flexor mixed with both commissural sides |
| 3 | 16 | global_drive (5), L_commissural (4), R_rg_flex (2) | 0.31 | BS_F/BASE + right flexor + left commissural |
| 4 | 8 | L_recip_inh (3), L_ia (2), L_ib (2) | 0.38 | left inhibition/sensory |
| 5 | 6 | R_ib (4), R_recip_inh (1) | 0.67 | right Ib + reciprocal |
| 6 | 2 | L_cutaneous (2) | 1.00 | clean cutaneous pair |
| 7 | 4 | R_cutaneous, R_ia, R_rg_ext, R_recip_inh (1 each) | 0.25 | right afferent mix |
| 8 | 3 | R_cutaneous, R_ia, R_ib (1 each) | 0.33 | right afferent mix |
| 9 | 6 | L_ia (4), L_recip_inh (1), L_rg_flex (1) | 0.67 | left Ia |
| 10 | 16 | R_ia (5), R_motor (3), R_renshaw (3), R_ib (2) | 0.31 | right sensorimotor territory |
| 11 | 14 | L_ia (4), L_ib (3), L_motor (3), L_renshaw (3) | 0.29 | left sensorimotor territory |
| 12 | 8 | R_ia (3), R_renshaw (2), R_recip_inh (2) | 0.38 | right Ia/Renshaw |
| 13 | 5 | L_ia (3), L_ib, L_rg_flex (1 each) | 0.60 | left Ia |
| 14 | 4 | R_ia (3), R_ib (1) | 0.75 | right Ia |
| 15 | 8 | L_ib (3), L_renshaw (3), L_motor, L_rg_flex | 0.38 | left Ib/Renshaw |
| 16 | 4 | R_motor (3), R_renshaw (1) | 0.75 | right motor/Renshaw |
| 17 | 5 | L_motor (2), L_rg_flex (2), L_recip_inh (1) | 0.40 | left motor/flexor |
| 18 | 2 | R_rg_flex, R_commissural (1 each) | 0.50 | small mixed pair |
| 19 | 6 | R_commissural (4), R_rg_flex (2) | 0.67 | right commissural |
| 20 | 4 | L_commissural (4) | 1.00 | clean left commissural |

The group names are in `benchmarks/evo/reduced_0_24_territory_partition/report.md`.

**Physiological reading.** The game recovers some clean units (L cutaneous,
L commissural, R Ia, R motor/Renshaw) and a strong left/right separation for
many populations. It does **not** recover the full nine-module per-limb
reference: the large players 10/11 mix Ia, Ib, motor, Renshaw and the flexor
RG; players 2/3 mix flexor with commissural across limbs. The pattern looks
more like **per-limb sensorimotor territories plus commissural bridges** than
the fine functional modules. That is a meaningful result, not a failure of the
matrix: it says the game's fraction currency groups these units together.

## 4. Global drive verdict

The global-drive hypothesis (BS_E, BS_F, BASE_pg and their eight projections as
one player) is **not supported** in this run. The classifier returns `other`:

- rate segments: `pop_BS_E` → player 0; `pop_BS_F` and `pop_BASE_pg` → player 3;
- projections: players 0, 1, 2, 3, 4, 5;
- `BS_E` sits with left extensor, `BS_F`/`BASE` with right flexor; the
  projection weights fan out into the limb they drive.

The rates are not one global player, and the projections are not kept with
their rate population. Two readings are possible:

1. the game is doing the right physiological thing — a drive projection
   belongs to the limb it drives, while the rate is a shared source that gets
   absorbed by the dominant limb territory;
2. the proximity matrix does not make the shared-source relation strong enough
   to keep the drive together.

The current evidence favors (1) at this scale, but it is one seed and three
generations. A controlled ablation that forces the three rate segments into one
player and compares performance is the next test.

## 5. Performance and bio-plausibility

Fast CPG, 400 steps, seed 0, same game:

| arm | players | loss_eval | r_p90 | at_bound | q>1 | viol | entropy | ARI | NMI | phase gate | bio accept | duty in dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| manual 141-segment | 141 | 0.263 | 0.558 | 4 | 50 | 0 | 0.480 | 0 | 0.729 | FAIL | False | True |
| reference Level-2 | 19 | 0.199 | 0.482 | 30 | 50 | 0 | 0.392 | 1 | 1 | FAIL | False | True |
| evo territory (P=24) | 21 | **0.148** | 0.517 | 20 | 50 | 0 | 0.719 | 0.154 | 0.564 | FAIL | False | True |
| evo random (P=24) | 24 | 0.176 | 0.559 | 10 | 50 | 0 | 0.638 | 0.0008 | 0.412 | FAIL | False | True |

- No type/sign violations in any arm.
- The evo-territory partition has the lowest loss in this one-seed comparison,
  better than both the manual 141-segment map and the 19-module reference.
- The phase gate fails for every arm; the failures are the known CPG issues
  (stance/E offset, L/R drift, joint-angle cycle correlation, and for some arms
  E/F offset). See `benchmarks/evo/evaluation/evaluation.json` for the exact
  failures.
- `biological_acceptance` is false and angle coverage is low in every arm
  (manual 0.056, reference 0.221, evo territory 0.229, evo random 0.251).
- Duty factors and double support remain in distribution.

This is a **partition-level** result: it says the auto player map is not worse
than the manual/reference maps at this scale, and is better on loss. It does
not say the CPG is bio-plausible; the phase gate is red for all arms and must
be fixed in the CPG/phase work, not in Evo.

## 6. Caveats

- One seed, three generations, 400 steps per generation, fast sizes.
- The full 800-step × 6-generation × 2-seed × {P=24,137} × {territory,random}
  × {partition,both} matrix has not been run; the harness supports it
  (`python bench_evo.py --compare`).
- The 141/273 inventory reflects the current working tree; the spec's 137/269
  snapshot is stale.
- The reference hierarchy is my physiological reading, not a laboratory
  ground truth. The user will evaluate the picks.
- The global-drive verdict is one run and may change with more generations,
  seeds, or a stronger shared-source weight.
- No functional (Fisher/advantage) affinity is used in redistribution in v1;
  only the structural matrix.

## 7. Next experiments

1. Run the full matrix (`--compare`) with 2 seeds and 800-step generations;
   report the seed spread, not a single number.
2. Test P=19/21 directly; does starting at the reference granularity recover
   the modules with higher purity?
3. Ablate the global drive: force the three rate segments into one player and
   compare loss/ARI; this tests interpretation (1) vs (2).
4. Add functional Fisher affinity to redistribution as a later sensitivity
   analysis, as the backlog proposed.
5. Re-run the partition evaluation with 1200–2000 steps and 3 seeds; only then
   treat the loss ranking as evidence.
6. Keep the phase gate and bio acceptance as separate CPG gates; do not use
   Evo to hide a red phase gate.

## 8. Reproduce

```powershell
# smoke
python bench_evo.py --smoke

# reduced first pass (what this report used)
python bench_evo.py --players 24 137 --seeds 0 \
  --inits territory random --stops partition \
  --steps 400 --generations 3 --prefix reduced

# partition evaluation
python bench_evo.py --evaluate --eval-seeds 0 --eval-steps 400 \
  --eval-partition benchmarks/evo/reduced_0_24_territory_partition/final_partition.json \
  --eval-partition benchmarks/evo/reduced_0_24_random_partition/final_partition.json
```
