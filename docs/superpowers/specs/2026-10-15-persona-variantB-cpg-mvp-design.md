# Design (revision): PERSONA perturbation–advantage natural gradient + CPG bio-plausible init + render fix

Date: 2026-10-15 (strategy revised 2026-10-16)
Status: approved (chat, 2026-10-15; strategy revision agreed 2026-10-16)
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
- **Error / advantage**: `E = |y* − y|` (2-vector); per-leg phase baseline `b`
  (moving average); `A = b − E` (2-vector reward advantage).
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
  as extra columns of `θ` (scored with their empirical perturbation std).
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
- CUT: `CUT→RGE (static)`, `CUT→InE`.
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

Optimise: every connection **weight**, every **rate**, Izhikevich `a,b,c,d`.
**Exclude only connection `p` and σ_param** (see §5.5).

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
  reward advantage `A = phase_baseline(E) − E`; `influence() = |Cov(Δθ, A)|`.
- **V3** Step `Δμ = η·F⁻¹·g` (natural gradient; diagonal case `Δμ_i = Cov(Δθ_i,ΣA)`);
  ascent on the advantage = descent on error; drift-free for inert parameters.
- **V4** Optimise connection weights + rates + Izh `a,b,c,d`; exclude only
  connection `p` and σ_param.
- **V5** σ_param held fixed for the MVP (`TODO(sigma_param)`); `sign(adv)` hack
  removed.
- **V6** `cpg_2legs_fast` topology authoritative; remove scheme direct half-centre
  and `InE/InF→MN`.
- **V7** Reference names/values per §5.4; Ib/Renshaw keep project values.