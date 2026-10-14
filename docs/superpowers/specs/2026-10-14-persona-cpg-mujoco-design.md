# Design: PERSONA-personalized spiking CPG coupled to a MuJoCo rat hindlimb
(Parameter-Exploration and Reinforcement via Natural-gradient Approximation - PERsoNA)
Date: 2026-10-14
Status: draft for approval

## 1. Intent and why

Personalize a biologically detailed spiking spinal CPG so that a simulated rat
hindlimb reproduces reference locomotion kinematics. The CPG is a NEST network;
the limb is a MuJoCo model; the optimization is the PERSONA scheme (online
natural-gradient / Fisher-matrix search over the CPG's neuronal and synaptic
parameters). The framework already exists as a scaffold (see §3); this work
corrects, completes, unifies and validates it.

Success = the controlled hindlimb's ankle angle tracks the mocap-driven target
limb through a locomotion bout, and the resulting muscle/afferent/spike patterns
match representative rat locomotion statistics.

## 2. Goals / non-goals / anti-goals

### Goals
- **G1 PERSONA framework** (`persona/`): error/loss objective (minimize),
  component-wise **vector** error and advantage, phase-separated (stance/swing)
  moving-average baselines, empirical **correlation** Fisher matrix over all
  parameters **plus noise features**, natural-gradient step, dual standard
  deviation (`sigma_ctrl` for exploration/exploitation, `sigma_param` for
  biological dispersion), noise recorders that capture Poisson-rate changes
  **and** STDP weight changes, and unit unification without precision loss.
- **G2 CPG** (`t2xcpg/`): the two-leg circuit exactly per the Diagram-2 scheme
  (RGE/RGF, reciprocal InE/InF, M-E/M-F, Renshaw R-E/R-F, Ia-E/Ia-F, In1,
  V2a/V0v/V0d/V3-F, CUT, brainstem BS→E/BS→F), with the **reflex arc** built
  correctly, **Ia and Ib implemented as separate afferents**, and every
  population and connection registered through the PERSONA `Registry`
  (poisson / izhikevich / static / STDP as applicable).
- **G3 Adapter** (`model/`): sensory feedback driven live by the MuJoCo physics
  (CUT from plantar contact, **Ia from muscle length + stretch velocity**, **Ib
  from muscle activation / tendon force**), for **both** legs; all joints
  except the ankle held to the target so the ankle is the only free variable
  (MVP).
- **G4 Personalization** (`personalize.py`): two-stage optimization (reach the
  target's initial pose within a margin → track the mocap-driven target
  through the bout), with a loss-tracked early stop, plus matplotlib validation
  figures against the Zenodo mocap data and locomotion statistics.

### Non-goals
- Optimizing hip/knee/other joints (explicitly deferred; ankle SOL/TA only for
  the MVP).
- A NEURON backend or mixing NEURON with NEST.
- Vectorized (non-scalarized) advantage — deferred by the human partner; the
  code must contain `TODO` hooks where this will be reworked.
- Real-time constraints or MPI/HPC parallelization.

### Anti-goals
- **Timer-driven CUT/Ia** (the mistake in `tinycpg/cpg_2legs_fast.py`): no
  afferent may be scheduled by a clock; every afferent rate must come from the
  MuJoCo state.
- Precision/entropy loss from unit unification — normalization must be
  numerically reversible.
- Editing the reference folders (`tinycpg/`, `memcpg/`) or the legacy
  `persona.py` script.

## 3. Current state (verified)

- `persona/`: `PersonaParam` (dual sigma, done), `Population`, `Connection`,
  `Registry`, `NoiseRecorder`, `FisherMatrix` + `PhaseBaselines`, `Agent`.
  Gaps: Fisher uses raw covariance of perturbations; `Agent`/`FisherMatrix`
  collapse the vector advantage to a scalar and the objective is `-error`;
  noise captures only Poisson rates; `Connection`/`Population` mostly wired but
  not all NEST objects flow through the registry.
- `t2xcpg/`: NEST/izhikevich `L2CPG`, `Leg`, `BrainStem` built through
  `Registry`; `update_sensory_feedback(...)` already accepts live Hz. Wiring is
  close to Diagram 2 but needs audit (reflex-arc weights/probabilities,
  Ia/Ib separation, Renshaw placement, commissural V-pathways).
- `model/`: `adapter.py` (`CPGInput`/`CPGOutput`/`CPGAdapter`), `leg.py`
  (MuJoCo `Leg.step`, one controlled limb + one reference limb, computes
  CUT/Ia/Ib), `cut.py` (`CUTAfferent`, `IaAfferent`, `IbAfferent`),
  `model.py` (closed-loop `Model`).
- `personalize.py`: single-stage Agent + L2CPG + MuJoCo loop + 5-panel plot.
- Runtime (WSL Ubuntu, `/home/t0kgw/.virtualenvs/KinematicsCPG`): Python 3.12,
  NEST 3.10, MuJoCo 3.14, NEURON 9.0.2, numpy 2.5, scipy 1.18, pandas 3.0,
  h5py 3.16, matplotlib 3.11. Repo at `/mnt/d/CPGvalidation`.
- `data/zenodo/DatasetMotionCaputure/Rat{A..E}/…` = marker CSVs
  (`IC/GT/KN/EA/MH` + origins, L/R) → derivable joint angles.
  `data/zenodo/DatasetStatisticalAnalysis/StatisticalDataTable.csv` = duty
  factor, double/single support, step/stride length, synergy contributions.

## 4. Architecture and responsibilities

```
data loader (mocap→joint angles) ──▶ target limb trajectory
                                             │
persona/  Registry ◀── t2xcpg/ (CPG: populations+connections, all registered)
   ▲   ▲                     │  update(CPGInput)   step()→CPGOutput
   │   │                     ▼
 Agent Fisher            model/ MuJoCo  ◀── CPGOutput (SOL/TA drives, L&R)
   │   │                     │
   └───┴── NoiseRecorder ◀───┘  (sensory rates + STDP Δw)
                    personalize.py orchestrates the closed loop
```

- `persona/` is the single source of truth for optimizable parameters and the
  update rule. Nothing outside it mutates a parameter mean/sigma directly.
- `t2xcpg/` owns circuit topology; it may only create NEST objects through
  `Registry` helpers so PERSONA sees them.
- `model/` owns physics and sensory models; it never touches PERSONA internals.
- `personalize.py` owns the loop, stop criteria and logging.

## 5. Detailed design

### 5.1 PERSONA (`persona/`)

**Objective.** Minimize a scalarized loss derived from the vector error.
- `output` = `[ankle_L, ankle_R]` (the only measured outputs; there is no EMG /
  muscle target). Global locomotion statistics are used for validation only,
  never for optimization.
- `error_vector E = |target - output|` (component-wise).
- `loss = mean(E)` (see `TODO` below).
- Per-phase baseline `b` (moving average, `PhaseBaselines`), **loss** domain:
  `advantage = b - loss` (positive when better than the phase average).
- Parameter step: natural-gradient **ascent** on `advantage`
  `Δμ = η_mu · F⁻¹ · (s ⊙ advantage)`, `s_i = (a_i-μ_i)/σ_ctrl,i²`.
- `TODO(advantage)`: replace scalarized `loss`/`advantage` with the
  component-wise vector form (human partner will specify). Leave this comment
  at every place error/advantage is computed (`Agent.step`,
  `FisherMatrix.compute_advantage`/`compute_natural_gradient_step`).
- **Per-leg phase (D11):** the two ankle components are contralateral, so the
  advantage baseline is keyed to each leg's own phase, taken from that leg's
  CUT contact (stance = contact). `Agent.step`/`PhaseBaselines` gain a
  per-component phase API; Stage 1 (static pose) uses a single neutral baseline
  until locomotion starts.

**Fisher / Strategy.** `F = ` empirical **correlation** matrix over a rolling
window of the feature-perturbation history. Features = parameters + noise tags.
Unit unification: each feature column is standardized (zero mean, unit std)
before `XTX`, so heterogeneous units (Hz, pA, probabilities, rad) enter the
correlation on equal footing; the diag is 1 and a ridge `λI` regularizes.
Record the standardization scale per feature so the step can be mapped back to
physical units losslessly. `TODO(fisher)`: confirm correlation vs covariance
with the human partner.
*Well-posedness:* the registry exposes ~250–280 features while any practical
window is smaller, so `F` is rank-deficient and `F⁻¹` is ridge-dominated
(≈ plain SGD). Resolution for the MVP: keep the ridge (asymmetric `F⁻¹` is
exact for this rank), and treat the *effective* optimization subspace as those
features that actually vary; a `TODO(fisher)` marks where a recursive /
Sherman–Morrison inverse or a pruned subspace can replace this.

**Dual stddev.**
- `σ_ctrl`: samples the exploratory mean, drives exploration/exploitation, and
  is updated by the policy-gradient variance rule.
- `σ_param`: re-samples per-neuron/ per-synapse biological dispersion via
  `Population.sample_matrix` and `PersonaParam.sample_population`; updated
  separately (`update_parametric_variance`).

**Noise recorders.** `NoiseRecorder` observes, each step:
- Poisson generator rates (BS-E/F, CUT, Ia-E/F, Ib-E/F) — deltas;
- STDP synaptic weight changes — subscribed via a NEST `weight_recorder` on
  each plastic projection, sampled per step.
Their deltas enter the perturbation feature vector, so the Fisher couples them
to reward automatically. Noise features have **no** `score` term (they are not
sampled actions): they influence the step only through `F⁻¹`, exactly as
specified ("strategy considering noise as an ordinary parameter").

**Agent API (kept stable).** `sample_actions()`, `apply` via registry,
`step(output, target, phase) → dict`, `history` logs. Add `loss` to history.

### 5.2 CPG (`t2xcpg/`)

Circuit per Diagram 2, verified against `memcpg/leg.py` and
`tinycpg/cpg_2legs_fast.py`:

- RGE↔RGF reciprocal inhibition through InE/InF (asymmetric E/F gains from
  `cpg_2legs_fast` constants).
- Reflex arc / Renshaw: `M-E → R-E → M-E` (recurrent inhibition),
  `M-F → R-F → M-F`, `R-E ⊣ Ia-E`, `R-F ⊣ Ia-F`, `Ia-E ⊣ M-F`,
  `Ia-F ⊣ M-E`.
- **Ia** (separate): driven by muscle length/velocity; excites homonymous MN and
  ipsilateral RG, drives the Ia inhibitory interneuron that inhibits the
  antagonist MN. `Ia-e_in → {m_e, ia_e, rg_e}`, `Ia-f_in → {m_f, ia_f, rg_f}`.
- **Ib** (separate): driven by muscle activation/force; **inhibits the
  homonymous MN** (autogenic inhibition): `Ib-e_in → m_e ⊣`, `Ib-f_in → m_f ⊣`.
- CUT: plantar contact → RGE (reinforce stance) and InE.
- Commissural: `RGF → V2a → contralateral V0v → contralateral In1 ⊣ RGF`;
  `RGF ⊣ contralateral V0d ⊣ RGF`; `RGF → V3-F → contralateral RGF`.
- Brainstem: `BS_in_e → RGE`, `BS_in_f → RGF`.
- All weights/probabilities are `PersonaParam` subjects to optimization;
  bio-plausible seeds taken from the reference constants.
- **Plasticity (D12):** STDP is used on the sensory input pathways
  (`CUT→RGE`, `Ia→RG`), mirroring `cpg_2legs_fast`'s plastic CUT/Ia→RG
  projections; all other projections are static synapses (weights still
  Persona-optimized). Each STDP projection gets a NEST `weight_recorder`.
- **Authority on the circuit** is `t2xcpg` fixed to the Diagram-2 scheme the
  human partner described; `memcpg`/`tinycpg` are used only as sources of
  bio-plausible constants, never as wiring authority (they disagree in
  topology and units).

`L2CPG.update(CPGInput)` routes live afferent rates to the Poisson generators
(no timers); `L2CPG.step() → CPGOutput` reads muscle drives.

### 5.3 Adapter / MuJoCo (`model/`)

- **Bilateral rig (D1):** extend the MuJoCo model so both hindlimbs are
  controlled and both have a target. Recommended: four limbs —
  controlled-L, controlled-R, target-L, target-R (`*_m_L/_m_R`,
  `*_ref_L/_ref_R`), mirroring the existing `_m`/`_ref` pair. The CPG's L leg
  drives controlled-L, R leg drives controlled-R; target limbs are position
  servos commanded by the mocap trajectory. *This is the largest single
  change and is the main item to confirm.*
- **Ankle MVP (D9):** controlled-L/R `hip`/`knee`(`/toe`) driven by position
  servos following the target trajectory, so only `ankle` is free and moved by
  `sol_m`/`ta_m`.
- `CPGInput` extended to carry **both** legs' full sensory state, including Ia
  `length` + `velocity` and Ib `activation` + `force` (already mostly present;
  fill R from the real R limb instead of copying L).
- `Leg.step()` accepts an external target ankle trajectory (from mocap) and
  returns `CPGInput` for both legs.
- Afferent models (`cut.py`) reused; Ia uses `k_l·(l-l0)⁺ + k_v·(v)⁺`,
  Ib uses `k_force·force + k_act·activation` (inhibition handled by the CPG
  sign).

### 5.4 Data (`loader/`)

Source: Fujiki et al. Zenodo 2025 (split-belt treadmill, rat hindlimb).
Markers per side: `IC` (iliac crest), `GT` (greater trochanter), `KN` (knee),
`EA` (external ankle), `MH` (4th metatarsal head), plus corrected-knee markers.
Motion capture 500 Hz, low-pass 20 Hz; 5 Wistar rats (RatA..RatE).

- **Filename decode** `RatXX_xYY_ZZfast_CC##.csv`: `YY` = speed ratio encoded
  as `1d50/1d75/2d00` (= 1.5 / 1.75 / 2.0), `ZZ` = fast belt (`L`/`R`),
  `CC` = condition (`T` tied, `S` split-belt, `W` wash-out), `##` = trial.
  This drives the **treadmill configuration**: slow belt 12 m/min =
  0.2 m/s; tied = both belts 0.2 m/s; split-belt fast = ratio × 12 m/min
  (0.3 / 0.35 / 0.4 m/s). The MuJoCo treadmill `surfacevel` is set accordingly
  (one plane for tied; two half-planes for split-belt, deferred beyond MVP).
- **Marker → angle (D10):** planar sagittal angles from marker triples —
  hip = angle(IC→GT vs GT→KN), knee = angle(GT→KN vs KN→EA),
  ankle = angle(KN→EA vs EA→MH); sign/zero aligned to the sim rest pose by
  subtracting the rest-pose offset. Produces per-side hip/knee/ankle rad
  trajectories resampled to the sim `dt`.
- **MVP target:** RatA, one clean trial (default: a `T` tied trial); the L/R
  ankle trajectories drive the two target limbs.
- Stats: read `StatisticalDataTable.csv` (duty factor, double support, step/
  stride length, synergy contributions) and the synergy tables for validation
  against the 5-rat distribution.

### 5.5 Personalization (`personalize.py`)

Two-stage, loss-tracked:
- **Stage 0 — build:** Registry, bilateral `t2xcpg.L2CPG`, bilateral MuJoCo rig
  (`model.Model`), `Agent`, mocap target driver.
- **Stage 1 — initial pose:** hold the target at the bout's first frame; run the
  closed loop until both ankle errors `|ankle_model - ankle_target| < eps_pose`
  (L and R) hold for a sustained window. This is ordinary PERSONA optimization
  (not a special mode).
- **Stage 2 — tracking:** play the mocap target trajectory; keep optimizing
  until the mean loss over a window falls below `loss_stop` (early stop) or the
  step budget is reached.
- Logs for plotting; deterministic seeding.

### 5.6 Validation figures

- Ankle tracking (model vs target, L & R) over time.
- Muscle activation (`mus_E`, `mus_F`) with stance/swing shading.
- Afferent rates (CUT, Ia, Ib) over time.
- Loss curve + phase-separated baselines/advantages.
- Parameter `μ`/`σ_ctrl`/`σ_param` trajectories and Fisher/influence weights.
- **Biological plausibility (acceptance):**
  - *Global parameters* (duty factor, double support, step length, stride
    length, synergy contributions): the model's point must fall inside the
    distribution of the 5 Zenodo rats — a scatter/cluster plot; the model is
    acceptable if it lands in one of the clusters (or the single unified cluster
    if the rats are homogeneous).
  - *Angles:* the model's joint angle must stay within a **5% margin for 70% of
    the locomotion phase** (thresholds configurable later).

## 6. Interfaces (stable contracts)

- `PersonaParam(mean, sigma_param, sigma_ctrl, min, max)`;
  `.sample_exploration()`, `.sample_population(n, center)`,
  `.update_mean`, `.update_controlling_variance`, `.update_parametric_variance`.
- `Registry.create_population(Population)`, `.create_connection(Connection)`,
  `.get_all_optimizable_params()`, `.sample_all_actions()`,
  `.apply_explored_parameters(dict)`, noise recorders.
- `CPGInput` / `CPGOutput` dataclasses (add bilateral fields).
- `CPGAdapter.update(CPGInput)` / `.step() → CPGOutput`.
- `Agent.sample_actions()` / `.step(output, target, phase) → dict`.

## 7. Constraints and conventions

- OOP, matching the existing class style; no comments except the requested
  `TODO(...)` hooks.
- Do not modify `tinycpg/`, `memcpg/`, or `persona.py` (reference only).
- Tests run in the WSL venv; add `tests/` with pytest.
- Keep the step `dt` consistent between NEST resolution, MuJoCo substeps and the
  mocap resample.

## 8. Testing / verification

- Unit: `PersonaParam` sampling/clipping/updates; `PhaseBaselines` per phase;
  `FisherMatrix` correlation shape, symmetry, unit diagonal, regularization;
  `Agent.step` loss decreases on a synthetic convex target; `NoiseRecorder`
  deltas and STDP capture.
- Integration: `t2xcpg` builds with all populations/connections registered;
  `Registry.get_all_optimizable_params()` is non-empty and stable; GID/reference
  integrity.
- Adapter: `CPGInput` fields populated from a stepped MuJoCo rig for both legs;
  ankle is the only free joint under the MVP config.
- End-to-end smoke: a short two-stage personalization on a simple periodic
  target reaches Stage-1 and reduces Stage-2 loss.

## 9. Decisions taken by the builder (check these)

- **D1** Bilateral rig = four limbs (controlled L/R + target L/R) in the MuJoCo
  XML. *Highest-impact call; confirm.*
- **D2** `F` is the **correlation** matrix (standardized columns, ridge), not
  raw covariance.
- **D3** Objective is **loss** (minimize); `advantage = baseline_loss - loss`.
- **D4** Vector error/advantage **scalarized** for the MVP, with `TODO` hooks.
- **D5** Two-stage run (initial-pose dwell → tracking) with loss early-stop.
- **D6** Output/reward vector = `[ankle_L, ankle_R]` only (no EMG/ muscle
  target data exists). Statistical locomotion parameters are validation-only.
- **D7** Target limbs are driven by mocap-derived joint angles.
- **D8** Noise features = Poisson rate deltas + STDP weight deltas.
- **D9** Non-ankle joints held to the target trajectory (ankle-only MVP).
- **D10** Marker→angle = planar marker-triple sagittal angles, sign/zero aligned
  to the sim rest pose.
- **D11** Phase is per-leg, from each leg's CUT contact; Stage 1 uses a neutral
  baseline.
- **D12** STDP on sensory input pathways (`CUT→RGE`, `Ia→RG`); all other
  synapses static (weights still optimized).

## 10. Resolved questions (human partner, 2026-10-14)

- **D1 confirmed** — 2 legs, 4 limbs; reuse the existing limb and mirror it.
- **D2 confirmed** — correlation matrix (single unit for mixed-unit variables).
- **D3–D5, D7–D9 confirmed.**
- **D6 revised** — reward vector is `[ankle_L, ankle_R]`; no muscle target.
- **D1 usage** — Stage 1 catches the *two* ankle angles of both target limbs.
- **Marker→angle (D10)** — planar marker-triple sagittal angles, aligned to the
  sim rest pose.
- **Phase (D11)** — per-leg, from each leg's CUT; neutral baseline in Stage 1.
- **Target trajectory** — MVP uses RatA; treadmill config from the filename
  (`xYY` speed ratio, `ZZ` fast belt, `CC` condition).

### Still to settle before Stage 2 runs
- Starting `eps_pose`, tracking margin, and `loss_stop` (plausible defaults
  proposed in the plan; tunable).
