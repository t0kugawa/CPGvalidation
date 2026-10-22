# Design (revision): PERSONA perturbation–advantage natural gradient + CPG bio-plausible init + render fix

> **Superseded for CPG wiring (§5.3–5.4)** by
> `2026-10-20-persona-variantB-cpg-vcells-design.md` (V-cell commissural relay,
> scheme-confirmed Ia/Ib wiring). The PERSONA and render sections here remain
> current.

Date: 2026-10-15 (strategy revised 2026-10-16; ground contact revised 2026-10-22;
structural freezes 2026-10-22; current objectives/future development 2026-10-22)
Status: approved (chat, 2026-10-15; strategy revision agreed 2026-10-16;
ground-contact revision approved for implementation 2026-10-22;
structural freezes approved 2026-10-22; current objectives recorded 2026-10-22)
Supersedes the PERSONA-strategy and CPG-initialisation sections of
`2026-10-14-persona-cpg-mujoco-design.md`. Everything else there still holds.

## 1. Intent

Close the final gap to a ready MVP. Three coupled problems:

1. **PERSONA's strategy does not move the parameters.** The output-sensitivity
   step `J·A` has `E[J·A] = J·E[A] = 0`, because the reward advantage
   `A = baseline − E` is baselined to zero mean while `J = ∂y/∂θ` is a
   near-constant plant property; it discards the per-sample perturbation `Δθ`
   that carries the correlation. The fix: estimate each parameter's impact from
   the **covariance of its perturbation with the advantage**, `Cov(Δθ, A)`,
   which is the (unbiased) score-function gradient and does not vanish.
2. **The CPG starts from non-biological, badly-named parameters**, and cannot
   plausibly personalise in a 20 s run.
3. **The render muscle geoms collide with the limb** and are invisible.

## 2. Goals / non-goals / anti-goals

### Goals
- **G1 PERSONA natural gradient on perturbation–advantage covariance.** The
  metric is `F = Cov(Δθ) + λ_F I`; the gradient is `g = Cov(Δθ, A)`; the step is
  `Δμ = η·F−1·g`. This is a true natural gradient (equivalently the ridge
  regression of the advantage on the perturbations) and equals ascent on the
  reward / descent on the tracked error. `influence() = |Cov(Δθ, A)|`.
- **G2 CPG bio-plausible init.** `t2xcpg` topology, names and initial values
  follow `tinycpg/cpg_2legs_fast.py`. Strengthen the two SOL/TA-alternation
  mechanisms (`Ia → Ia-IN → antagonist MN`; half-centre through InE/InF).
- **G3 Render** — thin muscle geoms, pushed off the bone axis, so activity is
  visible.

### Non-goals
- No synthetic oscillators; no change to the CPG neuron set beyond G2.
- **Only** connection `p` and σ_param are excluded from the optimised set (NEST
  cannot re-apply `p` post-hoc without rebuilding synapses, which is slow).
  Izhikevich `a,b,c,d` **stay** in the optimised set.

### Anti-goals
- A strategy that produces finite-but-meaningless updates again.
- Reference values copied blindly where they fight the sensory-driven design.

## 2.1 Current objectives (2026-10-22)

**C1 — Fix the L/R asymmetry.** The exact Renshaw/M-F freeze restored the L
flexor (`musF_L` 0.167, 6.2% zero; 20 s loss 0.632) but the R flexor still
collapses (`musF_R` 0.002, 93.9% zero; R ankle error +0.510). The next actuation
task is the R M-F pool and its Renshaw/Ia/Ib/commissural inputs, plus the
leading-leg/commissural asymmetry. Acceptance: both flexors active
(`musF` > 0.02, < 50% zero) and the R ankle error reduced without regressing L.

**Resolved for the flexor-collapse gate (2026-10-23).** The collapse is
policy-driven, not a wiring bug. In the 20 s baseline, PERSONA drove
`rg_e_R` self-recurrence to ~101 (L ~11.5), `rg_f_R`'s Izhikevich `c/d` to
−77/10.2 (L −62/1.1), `BS_F→RGF_R` to 0, and the flexor M-F inputs down
(`rg_f→m_f` 2.75→1.09, `Ia_F→m_f` 12→1.29). The 2.75/12 seed underdrives the
reduced-size flexor even before learning: the healthy L side learns
`rg_f→m_f` to ~43, which is why only L survives. Parameter-homeostasis screens
(a per-site spring on the RG core at power 0.05, and full L/R symmetrization of
the means) did **not** fix the activity winner-take-all: with identical
parameters the sensory/plant loop still latched one RG (L 0.011 / R 0.015
`musF`, or the mirror image). Size-compensating the tonic drive restored
FINAL-like RG rates but `musF_R` stayed ~0.01 because the output seed is too
weak. Seeding the flexor M-F drive at 4x (`rg_f→m_f` 11.0, `Ia_F→m_f` 48.0,
left optimizable) keeps both flexors active over 20 s: `musF_L` **0.093**
(11.3 % zero), `musF_R` **0.053** (19.1 % zero). The R ankle error is
unchanged (0.752 vs 0.752; L 0.531 vs 0.515), so the **tracking** error
remains the actuation blocker (V19/V21); the silent-R-flexor defect is fixed.

**C2 — Migrate the freeze hacks to the native PERSONA registry.** The current
`t2xcpg/leg.py` freeze uses `min=max`/`sigma=0` on `PersonaParam` (a fallback).
Replace it with `Registry.freeze` / `recapture_defaults` from
`2026-10-22-persona-registry-usability-design.md`: freeze the six Renshaw
weights on `mean` and the M-F Izhikevich `a/b/c/d` on `mean` and `sigma`, after
`finalize_connections` (the `recapture_defaults` path covers the post-creation
M-F mutation). Keep every projection; no deletion. Tests must cover the native
API, the freeze anchor after the CPG mutation, and the A/B toggles.

## 2.2 Future development (2026-10-22)

- **Homeostasis on different parameter groups.** With the homeostatic penalty
  for maximisation available by default, run an A/B matrix: H off vs H on
  (default) vs H on + freeze; parameter groups: CPG static weights, Izhikevich
  `a/b/c/d`, rates, STDP hyperparameters, Renshaw/Ia/Ib, and the SOL/TA muscle
  actuators. Measure `musF`/`musE` symmetry, ankle tracking/contact, loss, and
  whether the H penalty alone can replace the structural freeze.
- **Freeze vs spring vs H.** Compare the native freeze, the registry
  homeostatic spring (registry-usability spec §5), and H credit adjustment on
  the same sites; decide which sites should be frozen, which sprung, and which
  left free.
- **L/R asymmetry mechanisms.** After C1, test whether the R M-F collapse is
  driven by the R Renshaw loop, the R Ia/Ib inputs, the commissural V-cells, or
  the leading-leg gain; add a symmetry acceptance metric.
- **Long-run stability.** 120 s with the native freeze + H, checking the
  `musF`/`musE` ratio and ankle error; the current 20 s freeze is a
  short-horizon result.

## 3. Current state (verified)

- `persona/strategy.py`: `FisherMatrix.compute_correlation_matrix` standardizes
  the perturbation columns and returns `(1−λ)·C + λ·I` (unit diagonal);
  `compute_natural_gradient_step` solves it against a scalarised
  `mean(advantage)` score.
- `persona/agent.py`: `step` samples per-parameter actions, applies them,
  computes vector error, phase-baselined advantage, and logs loss.
- `t2xcpg/leg.py`: the scheme topology (direct `RGE⊣RGF`, InE/InF excitatory
  relays to MN) — **to be replaced** by the `cpg_2legs_fast` topology.
- Verified: a 1.4 s/step cost was OpenBLAS thread thrashing (fixed with
  `OPENBLAS_NUM_THREADS=1`); a 322×322 solve goes 1305 ms → 1.7 ms.

## 4. Architecture

```
loop: a = μ + σ_ctrl·ε  →  apply(a)  →  nest+MuJoCo  →  y = [ankleL, ankleR]
      E = |y* − y|;  A = phase_baseline(E) − E        # reward advantage (2-vec)
      record (Δθ = a − μ, A)                          # rolling window W
      F = E[score·scoreᵀ] + λI,  score_i = Δθ_i/σ_ctrl,i²   # score Fisher
      G = Cov(Δθ, A)                                  # per-parameter impact
      Δμ = η · F−1 · E[score·ΣA]                      # ascent on the advantage
```

## 5. Detailed design

### 5.1 Strategy — perturbation–advantage covariance (natural gradient)

- **Output** `y = [ankle_model_L, ankle_model_R]` (the tracked output);
  target `y*`.
- **Error / advantage**: `E = |y* − y|` (2-vector); each component keeps its
  **own** phase baseline `b_i(phase_i)` (moving average), so `A_i = b_i − E_i`
  compares the left ankle in stance with its own previous stance, independently
  of the right leg. Model output and target share one moving-average filter.
- **Record** the pair `(Δθ_k, A_k)` each step in a rolling window `W`:
  `Δθ = a − μ` (the exploration perturbation), `A` the advantage above.
- **Impact / influence:** `G = (1/W)·ΔΘcT Ac = Cov(Δθ, A)` (shape `n_θ × 2`);
  `influence() = |Cov(Δθ, A)|` normalised to [0,1].
- **Metric / step:** policy (score) Fisher `F = E[score·scoreᵀ] + λ_F I` with
  `score_i = Δθ_i / σ_ctrl,i²`; gradient `g = E[score·ΣA]`; step
  `Δμ = η·F⁻¹·g`. For the diagonal case `Δμ_i = Cov(Δθ_i, ΣA)` — the correct
  natural gradient, which does **not** amplify an inert low-σ parameter's noise
  (dividing by `Cov(Δθ_i)` would). `E[Δμ_i] = −η·∂E[ΣE]/∂θ_i`, i.e. ascent on
  the reward = descent on the tracked error.
- **σ_ctrl**: sampling std, updated by the policy-gradient variance rule.
- **σ_param**: see §5.5.
- **Noise features**: latest Poisson-rate deltas and STDP-weight deltas appended
  as extra columns of `θ` and **scored** like ordinary parameters (empirical
  perturbation std), so they contribute to both `F⁻¹` and the step.
- **Rank note:** `F` is ridge-regularised; a short window (`W < 3`) returns a
  zero step.

### 5.2 Influence (for logging/figures)

`influence() = normalised row norms of Cov(Δθ, A)` — the per-parameter impact
`|Cov(Δθ_i, A)|` on the tracked error, in [0,1].

### 5.3 CPG topology change — `t2xcpg/leg.py`

Replace the scheme wiring with `cpg_2legs_fast`:

Ipsilateral (per leg):
- reciprocal half-centre: `RGE→InE (exc)`, `InE⊣RGF (inh)`,
  `RGF→InF (exc)`, `InF⊣RGE (inh)`.
- RG self-recurrence: `RGE→RGE`, `RGF→RGF`.
- RG → MN **direct**: `RGE→M-E`, `RGF→M-F`.
- MN → muscle: `M-E→mE`, `M-F→mF`.
- Ia: `Ia_in→M (homonymous, exc)`, `Ia_in→Ia-IN (exc)`, `Ia-IN⊣antagonist MN`,
  `Ia_in→In (exc)`, `Ia(homonymous)→RG`.
- Ib (project addition, keeps): `GTO→InIb-E/F → {M⊣, antagonist M exc,
  contralateral-phase RG exc}`, with Ia/CPG convergence onto InIb.
- Renshaw (scheme only, no reference): `M→R→⊣M`, `R⊣Ia-IN`.
- CUT: `CUT→RGE (STDP)`, `CUT→InE`.
Commissural per `cpg_2legs_fast` (`P_COMM_F=0.22,W_COMM_F_INH=−20`;
`P_COMM_E=0.10,W_COMM_E_INH=−8`).

**Remove**: direct `RGE⊣RGF`/`RGF⊣RGE`; `InE/InF→MN`.

### 5.4 Parameter names + bio-plausible initial values (from `cpg_2legs_fast`)

- `rg_e→in_e` = `W_RG2INE` = 12.0
- `in_e→rg_f` = `W_INE2RGF` = −8.0
- `rg_f→in_f` = `W_RG2INF` = 18.0
- `in_f→rg_e` = `W_INF2RGE` = −48.0
- densities `P_RG_RECIP_E=0.15`, `P_RG_RECIP_F=0.30`
- `rg_e→rg_e` = `W_RG_REC_E` = 4.0 (`P_RG_REC=0.12`)
- `rg_f→rg_f` = `W_RG_REC_F` = 5.5
- `rg_e→m_e` = `W_RG2MN` = 2.75; `rg_f→m_f` = **11.0** (C1 4x override, §2.1)
- `ia_in→homonymous M` = **48.0** (C1 4x override of the port's 12.0, §2.1)
- `m→mus` = `W_M2MUS` = 1.0 (`P_M2MUS=0.8`)
- `ia_in→homonymous RG` = `IA2RG_W` = 12.0 (`IA2RG_P=0.4`)
- `ia_in→Ia-IN` = `W_IA_IN2INT` = 6.0
- `Ia-IN→antagonist MN` = `W_IA_INT2ANT` = −10.0
- `ia_in→in` = `W_IA2IN` = 6.0 (`P_IA2IN=0.25`)
- `cut→rg_e` = `W_CUT2RGE_STATIC` = 14.0 (`P_CUT2RGE_STATIC=0.35`)
- `cut→in_e` = `W_CUT2INE` = 6.0 (`P_CUT2INE=0.30`)
- BS `BS_REGULAR_HZ` = 60.0; base `BASE_DRIVE_W/HZ/P` = 1.0/2.0/0.10

**Plastic projections (D12).** `CUT→RGE` and `Ia→RG` use STDP. A plastic
projection's **weight is owned by STDP** and is only seeded; PERSONA optimizes
the STDP hyper-parameters (`tau_plus`, `lambda`, `alpha`, `mu_plus`, `mu_minus`,
`Wmax`) instead. Connection probability `p` is a static structural constant and
is never sampled or optimized.

**Note:** the values above are the *unscaled* `cpg_2legs_fast` reference, so
`w_scale = 1.0` in every run (A6).
- izh exc `a .02 b .2 c −65 d 8`; izh inh `a .1 b .2 c −65 d 2`;
  RG-F bursting `c −55 d 4`

Ib and Renshaw keep bio-plausible project values (no reference); any reference
value that fights the sensory design is overridden (marked in code).

### 5.5 σ_param

**MVP: held fixed.** The biological dispersion stays at its initial value; the
`sign(adv)` hack is removed. `log σ_param` is promoted to a feature in θ in a
later revision (`TODO(sigma_param)`), once the registry can re-sample the
population dispersion during a run.

### 5.6 Optimised parameter set

Optimise: every **static** connection weight, every **rate**, Izhikevich
`a,b,c,d`, and the **STDP hyper-parameters** of the plastic projections
(`CUT→RGE`, `Ia→RG`). Exclude the STDP weight itself (owned by plasticity),
connection `p`, and σ_param (see §5.5).

### 5.7 Render — XML / `model/render.py`

Thin the SOL/TA tendon capsules, shrink the origin/insert site sizes, offset the
muscle path off the bone axis so the tendons are visible when they shorten.

### 5.8 Reference limb ground contact (revised 2026-10-22)

**Problem.** The target limb was a purely kinematic mirror with collision
disabled (V8). Its foot/toe geometry is identical to the controlled limb's
(both are scaled to the same per-trial mean segment lengths, V10), but it was
not stopped by the belt. Over `RatA_x1p50_Lfast_T1` its toe went up to 26 mm
below the dataset ground and stayed below it for 72-77% of frames, while the
controlled foot sat on the belt. The geometric `phase_ref`
(`foot_z_ref <= belt_z + CONTACT_Z_EPS`) therefore reported stance for ~80% of
the trial, and the target was not physically reachable.

**Decision.** Give the reference limb the same ground constraint as the
controlled limb, without bending joints or rescaling bones:

- `foot_ref_geom_L/R` and `toe_ref_geom_L/R` get `contype="2" conaffinity="1"`
  (the controlled foot uses the same pair; the treadmill is `contype=1
  conaffinity=1`).
- `toe_ref_ctrl_L/R` `kp` is raised from `0.1` to `2.0` to match
  `toe_m_ctrl_L/R`.
- `phase_ref` / `contact_ref_*` are derived from the actual reference/treadmill
  contacts, as for the controlled limb, instead of the height threshold.
- The belt stays at the dataset ground (V9); no bone is scaled (V10).

The reference remains servo-driven by the mocap angle; contact may push its
actual joint angles away from the command by the same amount as the controlled
limb. The commanded trajectory is unchanged, so the target and the reward are
unchanged.

**Spike evidence** (`RatA_x1p50_Lfast_T1`, 16 s, both limbs driven by position
servos to the same mocap trajectory, belt -3.34 mm; `m` = controlled,
`ref` = target):

| variant | ref below belt | contact ref / m | IoU | mean / p95 foot z difference |
|---|---:|---:|---:|---:|
| current (no ref contact) | -26.10 mm | 0% / 62% | 0.00 | 6.52 / 13.69 mm |
| ref contact | -6.07 mm | 63.5% / 62% | 0.83 | 0.31 / 0.97 mm |
| **ref contact + matched toe kp** | **-6.64 mm** | **64.0% / 62.0%** | **0.84** | **0.23 / 0.87 mm** |
| lift reference base per frame | -0.34 mm | 0% / 62% | 0.00 | 2.10 / 11.43 mm (25 mm hip offset) |
| lower belt to the model foot | -1.61 mm | 0% / 0.1% | 0.50 | 0.57 / 1.13 mm (no contact) |

The matched-toe-contact variant preserves the commanded ankle range
(`[-0.02, +1.75]` vs `[-0.02, +1.76]` rad) and makes the reference contact and
foot level match the controlled limb. It is the selected approach.

**Resolved (2026-10-22): keep the soft contact.** The shared ~6.6 mm
penetration is accepted as the model's contact compliance. Stiffening the
treadmill (`solref=0.012`) reduced it to ~4.7 mm, but moved the right-leg
contact fraction from 68% to 64% (the RatA x1d50 Lfast duty-factor band is
62.5-82.8%) and `0.010` fell below the band at 60%; the ankle torque limit
reduced the penetration but pushed the reference ~0.16 rad off the mocap. The
belt stays at `solref="0.02 1"`. Revisit only if a later model revision adds a
compliant foot pad or a different contact model.

**Closed-loop check (2026-10-22).** `run_validation.py --fast` (4 s, RatA T1,
`--belt dataset`) now reports the target foot at **-4.95 mm (L) / -6.98 mm (R)**
relative to the belt instead of -13 / -19 mm, and `phase_ref` matches the actual
contact on every frame. The controlled foot is at 42% / 37% contact because the
muscle-driven ankle still does not track the target (the existing actuation
blocker, V19/V21); the plant-level equality is verified by the servo probe and
by `tests/test_reference_contact.py`, not by the un-converged CPG run.

### 5.9 Ankle guard (2026-10-22)

With the still-weak muscle drive (V19/V21) the controlled ankle collapsed onto
the old wide range (`[-1.0, 2.6]` rad), and the soft limit let it reach
**+2.97 rad**: the paw visually trailed behind the knee. Without a fracture the
ankle cannot bend that far, so the joint gets a physiological stop.

- `build_personalizer` computes the run window's mocap ankle envelope (both
  sides, first `max_steps` frames) and calls `Leg.set_ankle_range(lo, hi)`,
  which narrows all four ankle joints and the reference ankle servo's
  `ctrlrange` to the observed envelope (`ANKLE_RANGE_MARGIN_RAD = 0.0`). An
  earlier 0.10 rad margin let the controlled sit at +1.82 rad while the target
  max was +1.694, so the guard now equals the observed max.
- The four ankle joints carry `solreflimit="0.004 1"` and
  `solimplimit="0.95 0.99 0.001"`, so the soft-limit overshoot is small.
- This is a guard, not a tracking fix: it stops the visual flip and the
  constraint violation, but the controlled ankle still sits at the limit while
  the muscle drive is low.

Measured on the 4 s `--fast` run: controlled ankle max **+1.72 rad** (guard
`hi = +1.69`, overshoot 0.03) instead of +2.81/+2.96 before; the reference
ankle stays inside the same guard.

### 5.10 Frozen CPG parameters (2026-10-22)

PERSONA's homeostatic update moved the Renshaw recurrent inhibition and the
M-F Izhikevich parameters away from their reference values. Over 20 s it drove
`r_f -> m_f` from **-4** to **-19/-24** and M-F `a/b/d` from 0.02/0.2/8 to
0.076/0.084/8.9; the L flexor went from `musF_L` **0.111** in the first 100
steps to **0.008** (90% zero). The flexor must not be maximised away, so the
following parameters are **frozen** (nothing is removed or disconnected):

| parameter (both L and R) | frozen value |
|---|---:|
| `m_e -> r_e`, `m_f -> r_f` | +3.0 |
| `r_e -> m_e`, `r_f -> m_f` | -4.0 |
| `r_e -> ia_e`, `r_f -> ia_f` | -2.4 |
| M-F Izhikevich `a` | 0.02 |
| M-F Izhikevich `b` | 0.2 |
| M-F Izhikevich `c` | -65.0 |
| M-F Izhikevich `d` | 8.0 |

**Implementation (2026-10-23).** The freeze is native:
`Registry.freeze` pins the six Renshaw weights on `mean` and the M-F
Izhikevich `a/b/c/d` on `mean` and `sigma`, called after
`finalize_connections`; `finalize_connections` refreshes the captured defaults
so the post-creation M-F `sigma=0` mutation is the anchor. Bounds and
exploration are otherwise unchanged, and `Agent.step` skips the frozen
updates. `CPG_RENSHAW_CLAMP=0` and `CPG_MF_IZH_FREEZE=0` are the A/B toggles.
PERSONA still samples these parameters and they stay in `F/Xbar/G`; it simply
**no longer maximises them**. This is intentional: the homeostatic update was
silencing the flexor. The rest of the CPG and PERSONA parameter set continues
to optimise, and every Renshaw projection stays in the network.

Measured on the exact-freeze 20 s run (`test_results/cpg_fixes/freeze_exact_20s`):
loss **0.632**, `musF_L` **0.167** (6.2% zero), `musF_R` **0.002** (93.9% zero);
L/R ankles 1.683/1.578 (errors +0.486/+0.510). The L flexor is restored; the R
flexor still collapses, so the L/R asymmetry is the remaining actuation issue,
not a drift of the frozen parameters. The earlier ±0.5-band probe gave loss
0.651 and `musF` 0.025/0.024.
`CPG_RENSHAW_CLAMP=0` and `CPG_MF_IZH_FREEZE=0` reproduce the unfrozen baseline
for A/B runs.

## 6. Interfaces (delta)

- `FisherMatrix.record_response(delta_theta, advantage)`; `compute_metric()` →
  `(F, scales)` with `F = Cov(Δθ) + λ_F I`; `compute_gradient()`; `influence()`
  = normalised `|Cov(Δθ, A)|`; `compute_natural_gradient_step(...)` (signature
  unchanged) → `Δμ = η·F−1·g`.
- `Agent` records `(Δθ, A)`; logs `influence` + metric.
- `t2xcpg` registry keys renamed to the reference names.

## 7. Constraints

- OOP, existing style; keep `TODO(advantage)`/`TODO(fisher)` hooks.
- Do not modify `tinycpg/`, `memcpg/`, `persona.py`.
- WSL venv; `OPENBLAS_NUM_THREADS=1` required for speed.

## 8. Testing / verification

- `FisherMatrix`: a parameter whose **perturbation covaries with `A`** outranks
  one that does not in `influence()`; the step moves the high-impact parameter
  more and is unit-invariant; short window stays finite.
- `Agent`: loss decreases on a synthetic target where a known parameter controls
  the output.
- CPG: `RGE→InE→RGF` exists, `RGE→M-E` direct, **no** direct `RGE→RGF`;
  population + initial-value checks against §5.4.
- Render: muscle geoms do not overlap the bone geoms.
- Reference-limb ground contact: the four `*_ref_geom` foot/toe geoms compile
  with `contype=2 conaffinity=1`; `phase_ref` follows the actual contacts; the
  servo-driven plant probe gives contact IoU(ref, m) >= 0.8 and mean
  `|foot_z_m - foot_z_ref| < 1 mm`. A `run_validation.py --fast` (the CPG run)
  confirms `phase_ref` matches actual contact and the target penetration is at
  the shared contact level; the controlled contact fraction there is a tracking
  target, not a geometry gate.
- Ankle guard: the four ankle joints use the stiff limit; `set_ankle_range`
  applies to both limbs and the reference servo; a 4 s `run_validation.py
  --fast` keeps the controlled ankle within ~0.1 rad of the guard instead of
  the old +2.8..+2.97 rad.
- CPG structural freezes: the six Renshaw weights and the M-F Izhikevich
  `a/b/c/d` are pinned through `Registry.freeze` (mean; mean+sigma for M-F)
  after `finalize_connections`; `Agent.step` cannot move them while their
  bounds stay normal; the A/B env toggles restore the unfrozen ranges.
  `tests/test_cpg_freezes.py` covers the anchor after the M-F mutation and the
  toggles.
- C1 flexor seed: `rg_f→m_f` and `Ia_F→m_f` are 4x the port reference and are
  **not** frozen; a 20 s run keeps both flexors active (`musF` > 0.02,
  < 50 % zero). The tracking error is reported separately (V19/V21).

## 9. Decisions (check these)

- **V1** Metric = policy (score) Fisher `E[score·scoreᵀ] + λ_F I`,
  `score_i = Δθ_i/σ_ctrl,i²`.
- **V2** Gradient/impact = `Cov(Δθ, A)` — the per-parameter covariance with the
  reward advantage `A_i = b_i(phase_i) − E_i`, each component baselined against
  its own previous same-phase loss; `influence() = |Cov(Δθ, A)|`.
- **V3** Step `Δμ = η·F⁻¹·g` (natural gradient; diagonal case `Δμ_i = Cov(Δθ_i,ΣA)`);
  ascent on the advantage = descent on error; drift-free for inert parameters.
- **V4** Optimise static connection weights + rates + Izh `a,b,c,d` + the
  hyper-parameters of the plastic (STDP) projections; exclude the STDP weight,
  connection `p`, and σ_param.
- **V5** σ_param held fixed for the MVP (`TODO(sigma_param)`); `sign(adv)` hack
  removed.
- **V6** `cpg_2legs_fast` topology authoritative; remove scheme direct half-centre
  and `InE/InF→MN`. `CUT→RGE` is STDP, not static.
- **V7** Reference names/values per §5.4; Ib/Renshaw keep project values; runs
  use the unscaled reference values (`w_scale = 1.0`).
- **V8 (revised 2026-10-22)** Target limb is a **physical mirror**: it keeps the
  dataset joint trajectory as its servo command, but its foot/toe geoms collide
  with the treadmill (`contype=2 conaffinity=1`) and its toe servo gain matches
  the controlled limb (`kp=2.0`). Its phase is taken from the actual
  foot/treadmill contacts (like the controlled limb), not from a geometric
  height threshold. It remains validation/reference-only: the loop still uses
  the model `phase_L/R`, and the commanded mocap angles are unchanged.
- **V9** **Belt = dataset ground**, one plane for both limbs:
  `min` over the run window of the foot-marker (MH) height (dataset frame, m) plus
  `TREADMILL_PENETRATION_M`. Not derived from the target limb's rig geometry.
- **V10** The **output limb is built at the target's size**: per side, the loader's
  sagittal mean segment lengths are applied to both limbs (`Leg.set_limb_lengths`),
  preserving the rest directions. L/R lengths stay exactly as recorded.
- **V11** `markers_to_angles` **unwraps** each segment direction over time and clamps
  each joint to `JOINT_RANGES`; the rig's hinge ranges (both limbs, both sides) are
  widened to match so every target angle is reachable.
- **V12** `stride_length_L/R` are computed from the controlled foot's horizontal
  excursion over the detected stance phases (contact cycles) and compared against the
  Zenodo distributions; `step_length_L/R` likewise use foot positions.
- **V13** σ_ctrl is updated in **log-variance** coordinates with a per-step cap
  (`grad = ((a-mu)/sigma)^2 - 1`), removing the `1/sigma^3` blow-up.
- **V14** The settling ("pose") stage has a step cap (`pose_max_steps`) and the run
  can stop on a wall-clock budget (`wall_clock_s`), so contact-driven CPG activity
  cannot stall the loop and a run fits a server time limit.
- **V15** MVP scope is **RatA**; per-side recorded lengths, no symmetrisation.
- **V16** A run's duration is **simulated** time (`--duration-s`, default 120 s) and the
  target trajectory **loops** (period = trial length), so runs longer than the 60 s
  reference trial keep walking. `--budget-s` is an optional wall-clock cap, not the
  server contract.
- **V17** `run.h5` carries `log/` (all per-step channels: both-limb CUT, Ia
  length/velocity/rate, Ib activation/force/rate, hip/knee/ankle model+target, muscle
  drives, phases, contacts, foot x/z, loss, per-leg advantage), `stdp/` (per-step mean
  weight change and update count from the weight recorders, sampled live weight
  distribution, final weight vectors) and `agent/` (means, σ_ctrl, σ_param, influence,
  advantage/error vectors, the final Fisher metric matrix and feature scales). Figures:
  tracking, model_vs_target, learning, sensory, stdp, param_shift, fisher, influence,
  metric.
- **V18** Weight recorders are **drained every step** (`read_plastic_events` reads the new
  events then clears `n_events`), so a long run costs O(new events)/step instead of
  re-scanning and growing the whole history; within a step both limbs share the batch.
- **V19** Belt placement is selectable (`--belt dataset|model`). Head-to-head on RatA
  Lfast T1 (4 s, small populations): `dataset` (belt −3.3 mm) closes the loop
  (model stance L 0.57/R 0.41, 2.6/3.2 N, CUT 47–112 Hz) but the muscle-driven ankle
  saturates (1.19…2.95 rad) and the tracking error is 0.94 rad; `model` (belt −17.0 mm)
  opens the loop (stance 0.00/0.12, CUT 0, no contact) with a smaller tracking error
  (0.31/0.39 rad) because the ankle merely drifts passively. `dataset` is the default:
  it is the specified treadmill position and the only setting that exercises the
  contact→afferent loop. The remaining blocker is the ankle actuation, not the belt.
  `--belt model` is **not** the reference-contact fix (it lowers the ground away
  from the controlled foot); it remains non-default and is kept as a diagnostic.
- **V20** `eps_pose` is the settling tolerance of stage 1 only: the pose stage ends after
  `pose_hold_steps` consecutive steps with `max(|ankle_model − ankle_target|) < eps_pose`
  (or when `pose_max_steps` is reached). It gates the stage transition, never the
  optimization. `TODO(perf)` marks `Registry.apply_explored_parameters`.
- **V21** Long-run (12000-step) readiness: all four per-leg spike recorders are read
  **and cleared** each step (RGE/RGF activity is returned by `Leg.get_rg_rates` and
  logged as `rg_*`), and `NoiseRecorder.samples` is a bounded deque. The only
  remaining linear consumer is `Agent.history` (≈16 kB/step); measured RSS is 403 MB
  after build and 448.8 MB at step 1550, projecting ≈640–760 MB at 12000 steps, and the
  measured step rate at `--final` sizes is ≈0.45 s/step (≈90 min for 12000 steps).
  MuJoCo 3.14.0 enforces the joint limits (`jnt_limited=True`, soft), so the observed
  +2.947 rad is a constraint violation, not a missing limit. In the 4 s `dataset`-belt
  run the `cut_rge_L` STDP weight grows 14.3 → 27.9 monotonically (Wmax = 120) while the
  `ia_*` weights stay ≈ 11.7–11.9, and the tracking-stage loss is flat at ≈ 1.2:
  personalization moves the STDP hyper-parameters but cannot move the loss, which the
  saturated ankle (C1) dominates.
- **V22** Reference-limb ground contact (2026-10-22): enable collisions on the
  reference foot/toe (`contype=2 conaffinity=1`), match `toe_ref_ctrl_*` kp to
  `toe_m_ctrl_*` (2.0), and derive `phase_ref` from the actual
  reference/treadmill contacts. Keep the dataset belt (V9) and do not scale bones
  (V10). This supersedes the "no collision / geometric phase" part of V8. The
  selected approach and its spike evidence are in §5.8; the shared ~6.6 mm
  soft-contact penetration is accepted as contact compliance.
- **V23** Ankle guard (2026-10-22): narrow all four ankle joints to the run
  window's mocap ankle envelope (`ANKLE_RANGE_MARGIN_RAD = 0.0`) and stiffen
  the joint limit (`solreflimit="0.004 1"`,
  `solimplimit="0.95 0.99 0.001"`). This is a physiological stop against
  hyper-plantarflexion while the muscle drive is low (V19/V21), not a tracking
  fix. The guard and its 4 s `--fast` result are in §5.9.
- **V24** Structural CPG freezes (2026-10-22): freeze the six Renshaw weights
  (`m_e->r_e` +3.0, `r_e->m_e` -4.0, `r_e->ia_e` -2.4, `m_f->r_f` +3.0,
  `r_f->m_f` -4.0, `r_f->ia_f` -2.4, both sides) and the M-F Izhikevich
  `a/b/c/d` (0.02/0.2/-65/8.0). Nothing is removed. The freeze is native
  (`Registry.freeze`, C2, 2026-10-23): Renshaw on `mean`, M-F on `mean` and
  `sigma`, after `finalize_connections`; bounds stay normal. PERSONA **no
  longer maximises these parameters**: its homeostatic update drove
  `r_f->m_f` to -19/-24 and the M-F `a/b/d` away from reference, silencing the
  flexor. The remaining parameter set still optimises. See §5.10;
  `CPG_RENSHAW_CLAMP=0`/`CPG_MF_IZH_FREEZE=0` are the A/B toggles.
- **V25** Flexor M-F drive seed (2026-10-23): `rg_f→m_f = 11.0` and the
  homonymous `Ia_F→m_f = 48.0` (4x the port's 2.75/12), both **optimizable**.
  The reference seed underdrives the reduced-size plant and lets PERSONA
  suppress the R side to zero; the 4x seed keeps both flexors active over 20 s
  (`musF_L` 0.093, `musF_R` 0.053) while the ankle errors stay at their
  previous level (R 0.752, L 0.531). FINAL can adapt the seed down. This
  overrides the §5.4 values and is marked in `t2xcpg/leg.py`.