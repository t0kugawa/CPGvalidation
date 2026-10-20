# Design (revision): PERSONA perturbation–advantage natural gradient + CPG bio-plausible init + render fix

Date: 2026-10-15 (strategy revised 2026-10-16)
Status: approved (chat, 2026-10-15; strategy revision agreed 2026-10-16)
Supersedes the PERSONA-strategy and CPG-initialisation sections of
`2026-10-14-persona-cpg-mujoco-design.md`. Everything else there still holds.

> **Superseded (CPG circuit, 2026-10-20):** §3, §5.3, §5.4 and the CPG part of
> §5.6 are superseded by
> `2026-10-20-persona-variantB-cpg-mvp-v2-design.md` (V-cell commissural relay,
> excitatory Ib→RG, no M↔M, explicit Ia project additions). The PERSONA
> strategy sections remain as revised.

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
- `rg_e→m_e`, `rg_f→m_f` = `W_RG2MN` = 2.75
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
- **V8** Target limb is a **kinematic mirror** (no collision, `contype/conaffinity=0`):
  its phase is geometric (`foot_z_ref <= belt_z + CONTACT_Z_EPS`), `contact_ref_*`
  is a 0/1 stance indicator, and it is validation-only (the loop uses the model
  `phase_L/R`).
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