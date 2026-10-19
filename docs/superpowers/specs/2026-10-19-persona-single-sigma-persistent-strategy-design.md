# Design (revision): single σ per site, per-site noise recorders, persisted two-type strategy (metric + trace + credit)

Date: 2026-10-19
Status: approved (chat, 2026-10-19)
Supersedes the PERSONA-strategy and σ sections (2026-10-15 spec §5.1, §5.2, §5.5, §5.6),
the noise-recorder interface, and the Registry-interface requirements of
`2026-10-15-persona-variantB-cpg-mvp-design.md`; supersedes V1–V5, V13 and the
`agent/` part of V17 there. Everything else — CPG topology §5.3, reference values
§5.4, render §5.7, V6–V12, V14–V16, V18–V21 — still holds.
Visual companion for the per-step walk-through: `docs/persona_process_review.html`.

## 1. Intent

Four coupled defects in the current PERSONA module:

1. **Dual σ loses the applied perturbation.** The sample is `μ + σ_ctrl·ε`, but
   each subject also gets biological dispersion `σ_param`. The recorded/scored
   perturbation matches only `σ_ctrl`, while the plant receives both — the
   score-advantage product is attenuated by a mismatch factor and `σ_param`
   never adapts. Fix: **one σ per site**, used for exploration, dispersion,
   recording, and scoring; the recorded sample *is* the applied sample.
2. **Noise recorders are per leg, not per site.** Tags such as `noise_cut_rate`
   collide between L/R (R overwrites L); several producers (STDP, Poisson) are
   only partially represented; `_step_deltas` is never cleared, so stale deltas
   leak into the metric and step length every step. Fix: per-site and
   per-producer recorders with unique keys, read-and-clear freshness, and
   β-decayed statistics.
3. **Credit assignment is lagged and windowed.** `Cov(Δθ, A)` inside one step
   cannot credit a parameter change whose effect appears several dt later
   (1-dt sensory latency + multi-step parameter effect), and rolling windows
   forget. Fix: a **persisted two-type strategy** — a metric `F`, a sample
   trace `X̄`, and a credit `G`, all merged with forgetting factor β every dt,
   never pruned.
4. **Integration gaps.** Some NEST `Create`/`Connect` calls bypass the Registry
   (muscle parrots, Poisson→parrot relays in `leg.py`/`bs.py`/`utils.py`), the
   Agent snapshots the parameter set once at construction, and
   `apply_explored_parameters` swallows exceptions silently. Fix: single
   Registry route, dynamic parameter sync, and surfaced apply errors.

## 2. Goals / non-goals / anti-goals

### Goals
- **G1 single σ per site.** `σ_s` drives exploration *and* per-subject
  dispersion; σ is updated online by its own rule (log-variance, variant A);
  the σ channel participates in the strategy through the covariance matrix.
- **G2 recorded = applied.** Every site's attached recorder logs `(x̂, σ)` each
  time the site is (re)sample-applied; the strategy consumes these records.
- **G3 late credit.** `X̄_t = βX̄_{t−1} + (1−β)a_t` (PERSONA actions),
  `G_t = βG_{t−1} + (1−β)X̄_tA_tᵀ`, `Δμ = η_μ(F_t+λI)⁻¹(G_t·1₂)` — every dt.
- **G4 noise accounting.** STDP projections and sensory Poisson generators get
  producer recorders; their deltas enter `F` with a declared fixed reference
  scale `R` (`s = δ/R²`; STDP: initial `Wmax`, Poisson: nominal rate, CUT:
  10 Hz); they never enter the trace/credit; they change step length through
  `F`.
- **G5 NEST integration audit.** All `nest.Create`/`nest.Connect` go through
  `Registry` helpers; site recorder keys are unique by construction; apply
  failures are recorded and logged, never swallowed.

### Non-goals
- No change to CPG topology or reference values (V6/V7 stand); no change to
  `tinycpg/`, `memcpg/`, `persona.py`.
- No per-phase strategy state: only the *advantage baselines* differ per phase.
- Population size `N` is not optimized — it is the realization substrate.
- Keep the `TODO(advantage)` / `TODO(fisher)` comment hooks.

### Anti-goals
- Any channel in the trace/credit that PERSONA did not produce (STDP weight
  changes, Poisson rates) — external noise stays in the metric only.
- A silent apply failure, a duplicated recorder key, or a stale delta swap.

## 3. Notation

- Site `s ∈ S`: every `PersonaParam` (population parameters: Izh `a,b,c,d`,
  rates; connection parameters: static weights, STDP hyper-parameters).
  State `(μ_s, σ_s)`; bounds `[θmin_s, θmax_s]`, `[σmin_s, σmax_s]`; own RNG.
- Step `t` at `dt = 10 ms`. `ε_s(t) ~ N(0,1)` i.i.d. per site and step.
  Noiseless target is unaffected; all randomness is the site's own stream.
- Sample: `x̂_s(t) = clip(μ_s(t) + σ_s(t)·ε_s(t))`; perturbation
  `Δθ_s(t) = x̂_s(t) − μ_s(t)`; unit perturbation `z_s = Δθ_s/σ_s`.
- Realization (the plant): per subject `x_j = clip(x̂_s + σ_s·ξ_j)`, `ξ_j`
  i.i.d.; the marginal per-subject width is `√2σ_s` (common sample + own
  draw) — accepted; `N` is a substrate, not a tuned quantity.
- Advantage `A(t) ∈ R²` (one per leg), `A_i = b_i(phase_i) − E_i`, `b` the
  per-component, per-phase moving-average baselines (unchanged, α = 0.90).
- Noise producer `p`: each STDP projection and each sensory Poisson group;
  per-step delta `δ_p(t)`; declared fixed reference scale `R_p` (STDP: initial
  `Wmax`; Poisson: nominal group rate; CUT: 10 Hz), set at registration. No
  estimation and no warm-up.
- Forgetting factor `β = 1 − dt/τ_mem`; `τ_mem` default 2.0 s (≈ a few
  locomotion cycles) → `β = 0.995` at `dt = 10 ms`.

## 4. Per-dt algorithm (normative)

State per site: `μ_s, σ_s`. Strategy state (persisted for the whole run):
`F ∈ R^{n×n}`, `X̄ ∈ R^{n}`, `G ∈ R^{n×2}`.

Channel layout (`n = 2·|S| + |P|`): sites `[μ-channels; σ-channels]`, then noise
producers. For each site `s` (sampling-time values):

    s^μ_s = Δθ_s / σ_s²            # μ-score of a Gaussian policy
    s^σ_s = z_s² − 1               # log-σ-score
    a_s   = Δθ_s / σ_s²            # trace row, μ part — score units (see below)
    b_s   = Δlogσ_s(t)             # trace row, σ part (σ change applied at t)

For each producer `p`:

    s^N_p = δ_p / R_p²             # declared scale; δ never enters the trace
    a row = 0 (excluded from X̄ and G)

Every step `t` (after the first `warmup = 3` steps, updates are applied; the
three warm-up steps only accumulate `F`, `X̄`, `G`):

    F_t    = β·F_{t−1} + (1−β)·s_t s_tᵀ            # s_t = [s^μ; s^σ; s^N]
    X̄_t    = β·X̄_{t−1} + (1−β)·[a; b; 0]           # trace, PERSONA only
    G_t    = β·G_{t−1} + (1−β)·X̄_t · A_tᵀ
    g_t    = [G_t·1₂ ; 0_noise]                    # zero rows for noise
    Δμ_t   = η_μ · (F_t + λI)⁻¹ · g_t              # μ rows applied only
    Δlogσ_s = η_σ · clip((z_s²−1)·mean(A_t), −0.25, +0.25)
    σ_s    ← clip(σ_s · exp(Δlogσ_s), σmin_s, σmax_s)

**Units (normative).** The trace's μ rows carry the μ-score (`Δθ/σ²`), not the
raw perturbation, so `g` and `F` live in the same space and
`Δμ = η(F+λI)⁻¹g` is exactly the score-paired natural gradient with a
multi-lag gradient estimate. With raw `Δθ` rows each site's gain acquires a
spurious `σ_s²` factor (~10⁶ spread over the current site scales: Izh `a` at
0.005 vs `Wmax` at 12), and the update fails unit invariance (`θ→cθ, σ→cσ`
would scale the step `c³` instead of `c`). The σ trace row stays in log-σ
units (the σ score `z²−1` is consumed directly by the σ rule; the cross terms
are second-order); noise producers have no trace rows.

Order within one loop iteration (normative; unchanged loop, changed contents):

1. **Sample** (at refresh step, see §5.2): for each site `x̂ = clip(μ+σε)`;
   write `(t, x̂, σ)` into the site's recorder. The agent keeps a per-site
   snapshot of `(μ_s, σ_s)` used for this sample.
2. **Apply/realize** `x̂` per site (populations per subject, static synapses
   per synapse, STDP hyper-parameters scalar).
3. **NEST dt** (`cpg.update` → `cpg.step`) and **MuJoCo dt** (`rig.step`):
   producer recorders receive this step's STDP deltas and Poisson rate deltas.
4. **Loss / advantage** `E_t`, `A_t` (per-leg phase baselines).
5. **Strategy** — channels, `F`, `X̄`, `G` above; `Δlogσ_s` for this step is
   computed here (agent) from the snapshot and `mean(A_t)`.
6. **Apply updates**: `param.update_mean(Δμ_s)`; `param.update_sigma(...)`
   (variant A), both computed from the **sampling-time** snapshot — never from
   a mean that was already moved this step (fixes the update-order bug).
7. **Log** (see §5.6).

The σ update is variant **A**: the current advantage with the phase MA
baselines, no extra EWMA. `TODO(sigma-ewma)` stays in the code to A/B-test
variant B (`Ā_t = βĀ_{t−1} + (1−β)A_t`) on the same run.

## 5. Detailed design

### 5.1 Single σ per site — `persona/parameter.py`

`PersonaParam` keeps one σ:

```python
mean: float
sigma: float = 0.1          # explore == dispersion
min: float = -1e6
max: float = 1e6
sigma_min: float = 1e-4
sigma_max: float | None = None   # None → max(2.0, sigma) at construction
rng: np.random.Generator
```

- `sample_exploration()` → `x̂ = clip(μ + σ·ε)`; `sample_population(n, center)`
  → realization draws `clip(center + σ·ξ, min, max)` (used by the registry's
  apply path with `center = x̂`).
- `update_sigma(sample, mu_at_sample, sigma_at_sample, advantage, lr_sigma)`:
  `z² = ((sample − mu_at_sample)/sigma_at_sample)²`; step
  `clip(lr_sigma·(z²−1)·advantage, ±0.25)`; `σ ← clip(σ_at_sample·exp(step),
  sigma_min, sigma_max)`.
- **σ bounds.** `sigma_min = 1e-4`; `sigma_max` defaults to `max(2.0, σ_initial)`
  — a site never starts out of bounds, and σ can shrink but not grow beyond its
  starting scale. Sites that previously had `σ_ctrl > 2` (Izh `c` 5.0, `Wmax`
  12.0, large weights, BS/BASE rates 5.0) keep their initial σ unclipped. Any
  site may override either bound explicitly.
- Remove `sigma_param`, `sigma_ctrl`, the `sigma` property,
  `log_sigma_param`/`set_log_sigma_param`, `update_controlling_variance`,
  `update_parametric_variance`.
- Initial σ per site = the site's **current σ_ctrl magnitude**: `_param(...)`
  and `_stdp_params(...)` pass `sigma=σ_ctrl_value`; `_izhi` per family
  (a 0.005, b 0.04, c 5.0, d 2.0); BS/BaseDrive rates σ = 5.0.

Per-site σ bounds are fields, so a site can override `[σmin, σmax]`.

### 5.2 Realization and cadence — `persona/registry.py`, `persona/population.py`

- Populations: `x_j = p.sample_population(n, center=x̂)` — one draw per subject,
  clipped to the site's `[min, max]`; applied with one `node_collection.set`.
  (The existing loop over params does this; `Population.sample_matrix` reads
  `param.sigma`.)
- Static synapses: one array realization per projection,
  `w_j = clip(x̂ + σ·ξ_j)`; apply with a single
  `synapses(key).set({"weight": array})` (verified supported in NEST 3.10).
  `N = len(synapses(key))` cached per projection. `TODO(perf)` stands — the
  apply step remains the cost hot spot.
- Device-only parameters (Poisson `rate` of registered device populations,
  e.g. BS-E/F, BASE) cannot take an array in one `set`: realize per device in
  a loop (`node.set({...})` per node). The silent mean-of-draws fallback is
  removed; a failed set is recorded in `last_apply_errors` (§5.5).
- STDP hyper-parameters: scalar `x̂` (shared by the projection). The STDP
  weight itself is never written by PERSONA (owned by plasticity, see §5.5).
- **Cadence** `realize_every_k` (default **1** = every dt): sample + realize +
  record every `k` steps; the strategy still updates every dt using the
  currently applied (held) sample from the site recorders. `k > 1` trades
  learning noise for wall-clock (the apply dominates the step cost).
- **Recorded = applied** holds by construction: the strategy reads site
  recorders, not a fresh draw.

### 5.3 Noise recorders — `persona/noise_recorder.py`, `persona/registry.py`

One recorder instance per **site** and per **producer**, registered in the
Registry under unique names:

    site:pop_<name>_<param>            e.g. site:pop_rg_e_L_a
    site:conn_<conn_key>_<param>       e.g. site:conn_cut_rge_L_weight_tau_plus
    producer:stdp:<conn_key>           e.g. producer:stdp:cut_rge_L
    producer:poisson:<group>           e.g. producer:poisson:cut_L

API (replaces `observe`/`observe_weight`/`record_direct`/`flush_distribution`/
`get_latest_step_deltas`/`samples`):

```python
class NoiseRecorder:
    def __init__(self, name, beta=0.995, scale=None): ...
    def record_sample(self, t, sample, sigma):      # site recorders
    def record_delta(self, t, delta):               # producer recorders
    def latest_sample(self) -> tuple[float, float] | None   # (x̂, σ), never cleared
    def take_delta(self) -> float | None            # fresh since last take; read-and-clear
    def stats(self) -> dict                         # diagnostics only
```

- **Declared producer scale.** Every producer recorder carries a fixed
  reference scale `R` supplied at registration (never estimated, never
  updated): STDP → the projection's initial `Wmax`; Poisson → the nominal
  group rate. CUT rests at 0 Hz, so it declares `R = 10 Hz` explicitly. The
  strategy scores a producer as `s = δ/R²`; `R = 0`/absent means no influence,
  and a step with no fresh delta uses `δ = 0` (score 0). There is no warm-up
  and no "n ≥ 2" gate: the first declared delta counts at full weight.
- β-EWMA statistics (`m`, `q`, `std`, `t`, `n`) remain available for
  diagnostics with constant memory, but are **not** used for scoring.
- **Freshness**: `take_delta` clears; a step with no new record yields `None`
  (treated as `δ = 0.0`); a stale value can never be re-read. (Fixes the
  `_step_deltas`-never-cleared bug.)
- Registry creates the site recorder for every optimizable parameter at
  `create_population`/`create_connection` time and exposes
  `site_recorder(full_key)` / `producer_recorder(name)`. Producer recorders:
  one per STDP projection (`create_connection` with a `weight_recorder`) and
  one per sensory Poisson group (`create_relay(noise_name=...)`).
- Producer set for the MVP: one recorder per plastic projection — 6 in the
  current network (`cut_rge_L/R`, `ia_rge_L/R`, `ia_rgf_L/R`) — plus 10 sensory
  groups (CUT, Ia-E, Ia-F, Ib-E, Ib-F × L/R). Muscle drives and RG rates are
  readouts/logs, not noise channels.

### 5.4 Strategy — `persona/strategy.py`

The class name `FisherMatrix` is kept (imports stable); it is now the
persisted two-type strategy. No windows, no per-phase reset:

```python
FisherMatrix(param_names, noise_names=None, lr_mu=0.02, lr_sigma=0.005,
             lambda_reg=1e-3, beta=0.995, warmup=3, baseline_alpha=0.90)
    .ensure_names(param_names, noise_names)   # zero-pad F/X̄/G when sites appear
    .compute_advantage(loss_vector, phase)    # PhaseBaselines, unchanged
    .step(sites, noise, advantage) -> dict    # sites: key -> (Δθ, σ, Δlogσ)
                                              # noise: name -> (δ, R); returns Δμ per site
    .influence()                              # normalised |G| row norms
    .F, .Xbar, .G, .noise_scales              # exposed for logging/tests
```

- F = β-EWMA of `s sᵀ` over `[μ-scores; σ-scores; noise-scores]` with
  `s_noise = δ/R²`; ridge `+λI` only inside the solve (keep the raw F).
- Solve with `np.linalg.solve`, fallback `pinv`. `g` has zero rows for noise;
  the σ rows of `F⁻¹g` are **not** applied (σ has its own rule) but σ and
  noise still shape the μ step through the covariance (they are columns of F).
- `influence()` over the `2·|S|` site rows.
- Warm-up: `F/X̄/G` accumulate from step 1; `Δμ` and `Δlogσ` are applied from
  step `warmup+1`. No other gating.

Housekeeping this replaces: `perturbation_history`/`loss_history`/
`phase_history`, `response_dtheta`/`response_da` windows, `_score_fisher`,
`_covariances`, `record_response`, `record_step`, `compute_metric`,
`compute_gradient`, `compute_correlation_matrix`, `update_feature_names`
(→ `ensure_names`). `feature_scales()` returns `sqrt(diag(F))` (kept for
h5/figures).

### 5.5 NEST integration — `persona/registry.py`, `t2xcpg/*`

- **Single route.** New helpers so no module calls `nest.Create`/`nest.Connect`
  directly:
  - `register_nodes(name, nodes)` — readout parrots (`mus_e/f`), relay parrots.
  - `create_relay(name, n, *, rate=0.0, rate_param=None, noise=False)` —
    `poisson_generator` + `parrot_neuron` + `one_to_one`, registered; when
    `rate_param` is given the generator is a registered optimizable
    `Population` (BS-E/F, BASE); when `noise=True` it gets a producer recorder.
  - `set_relay_rate(name, rate)` — sets the rate and, for `noise=True` groups,
    records the per-step delta into the producer recorder.
  - `create_spike_recorder(target)` / `create_weight_recorder()` — devices;
    `t2xcpg/utils.py` keeps only `nest.CopyModel`-based helpers
    (`copy_syn_model`, `create_stdp_synapse`).
  **Scope:** *everything* that creates or connects NEST nodes goes through the
  Registry; the only sanctioned raw calls outside `persona/registry.py` are
  `nest.CopyModel` in `t2xcpg/utils.py`. Enforced by a source-scan test (§8).
  Call-site rewrites: `t2xcpg/leg.py` (mus parrots, CUT/Ia/Ib relays, spike
  recorders, `update_sensory_feedback` → `set_relay_rate`), `t2xcpg/bs.py`
  (BS and base), `t2xcpg/utils.py` (`poisson_to_parrot` and
  `create_izhi_population` removed or rerouted).
- **Dynamic parameter sync.** `Agent` re-reads
  `registry.get_all_optimizable_params()` each step (no construction-time
  snapshot) and calls `strategy.ensure_names(...)` when keys appear; new sites
  start with zero history. (Removed sites are an error, logged loudly.)
- **No silent failures.** `create_population`'s `set()` and
  `apply_explored_parameters` collect failures into
  `registry.last_apply_errors: list[tuple[key, str]]` (cleared each apply); the
  loop logs each new failure once. Tests assert errors surface.
- **STDP delta redefined.** `sample_plastic_weight_deltas` drains the weight
  recorder (events carry `times, weights, senders, targets` — verified in NEST
  3.10) and computes the **per-synapse mean weight increment**:

      Δw = w_event − last_w[(key, sender, target)]     # first sighting → seed weight
      δ_key(t) = Σ Δw / N_syn(key)

  `last_w` is per projection and per synapse; `N_syn` is cached at
  `finalize_connections`. The drained batch is still shared across both limbs
  within one step (V18). This replaces the two-batch-mean difference.
- `finalize_connections` (bug fix from the plan) stays: collections are
  captured once the network is complete.

### 5.6 Loop and logging — `persona/personalization.py`, `persona/h5io.py`, `persona/validation.py`

- `Personalizer.run` keeps its order; it now (a) passes `t_ms` into
  `agent.sample_actions/step`, (b) gates sample+apply on `realize_every_k`,
  (c) logs `registry.last_apply_errors`.
- `Agent.history` groups: `means`, `sigmas` (**single**; replaces
  `sigmas_ctrl` + `sigmas_param`), `influence`; keep vectors.
- `run.h5`: `agent/` stores `means`, `sigmas`, `influence`, the final `F`
  (`feature_scales` unchanged) and adds `trace_final` (`X̄`) and
  `credit_final` (`G`). `stdp/` unchanged except the redefined delta semantics.
- `plot_param_shift` reads `sigmas`; `plot_metric` unchanged.

## 6. Interfaces (delta)

- `PersonaParam`: single `sigma`; `update_sigma(sample, mu_at_sample,
  sigma_at_sample, advantage, lr_sigma)`; no `sigma_ctrl/param` methods.
- `NoiseRecorder`: `record_sample/latest_sample/record_delta/take_delta/stats`.
- `Registry`: `site_recorder(key)`, `producer_recorder(name)`,
  `register_nodes`, `create_relay`, `set_relay_rate`, `last_apply_errors`;
  `sample_plastic_weight_deltas` per §5.5.
- `FisherMatrix`: `ensure_names`, `step(sites, noise, advantage)`,
  `influence`, `F/Xbar/G`; windows and covariance helpers removed.
- `Agent`: `sample_actions(t_ms=None)`, `step(output, target, phase,
  t_ms=None)`, history `sigmas`; `warmup`, `tau_mem_s`/`beta` parameters.
- `Personalizer.run`: `realize_every_k`; `personalize.py` gains
  `--tau-mem-s` (default 2.0) and `--realize-every-k` (default 1).

## 7. Constraints

- OOP, existing style; keep `TODO(advantage)`/`TODO(fisher)` hooks; add
  `TODO(sigma-ewma)` and keep `TODO(perf)`.
- Do not modify `tinycpg/`, `memcpg/`, `persona.py`.
- WSL venv `/home/t0kgw/.virtualenvs/KinematicsCPG`; repo `/mnt/d/CPGvalidation`;
  `OPENBLAS_NUM_THREADS=1` (+ OMP/MKL/NUMEXPR) before numpy import.

## 8. Testing / verification

- `PersonaParam`: single σ drives both sample and realization; σ update uses
  the sample-time snapshot (order-independent); cap ±0.25; bounds respected.
- `NoiseRecorder`: `take_delta` freshness (stale value cannot be re-read);
  β-EWMA stats converge to mean/var of a known signal; per-site keys unique
  across L/R (`cut` collision regression test).
- `FisherMatrix`:
  - `F/X̄/G` recursion equals a brute-force EWMA on random data;
  - **delayed credit**: synthetic plant whose advantage depends on a site's
    sample `k` steps earlier — after burn-in, that site's `Δμ` is the largest
    and non-coupled sites ≈ 0 (this is the test for G2/G3);
  - **no hidden σ² gain**: comparing two sites with σ differing ×100 at equal
    trace credit, their physical steps scale with the credit, not σ²; unit
    invariance `θ→cθ, σ→cσ` holds with a small ridge λ (=1e-6);
  - noise channels: zero-mean noise creates no systematic `Δμ` but changes the
    step through `F`; noise never appears in `X̄`;
  - warm-up finite; `ensure_names` pads correctly.
- `Registry`/NEST: every optimizable site has exactly one recorder; relays,
  parrots and generators are all registry-registered; **source scan** of
  `persona/`+`t2xcpg/`+root loop scripts finds no `nest.Create`/`nest.Connect`
  outside `persona/registry.py` (only `nest.CopyModel` allowed in
  `t2xcpg/utils.py`); `last_apply_errors` surfaces an injected failure,
  including the device-loop path (Poisson rates); STDP delta = Σ(Δw)/N on
  synthetic events; `sigma_max` default keeps initial σ > 2 unclipped.
- Integration: tiny two-leg run — finite F/G, σ bounded, means move every dt,
  `realize_every_k=2` keeps recorded = applied.
- Existing suites updated where the API changed (`test_variantb_*`,
  `test_noise`, `test_stdp`, `test_review_fixes`, `test_persona_objective`,
  `test_phase`, `test_personalization`, `test_fisher`).

## 9. Decisions (check these)

- **V22** One σ per site, used for explore and dispersion; recorded = applied;
  σ_min = 1e-4 and σ_max = max(2.0, σ_initial) per site; per-step
  `|Δlogσ| ≤ 0.25`; initial σ = previous `σ_ctrl` magnitudes; per-subject
  marginal variance `2σ²` accepted (`N` is a realization substrate, not
  optimized). *(Supersedes V5; amends V4: σ is now an updated quantity in the
  optimized set.)*
- **V23** Per-site and per-producer noise recorders with unique keys
  (`site:*`, `producer:stdp:*`, `producer:poisson:*`), read-and-clear
  freshness, declared fixed producer scale `R` for scoring (`δ/R²`; no
  estimation/gate); β-EWMA stats kept for diagnostics only. *(Fixes tag
  collisions and stale deltas.)*
- **V24** Persisted two-type strategy, updated every dt, never pruned:
  `F = βF + (1−β)s sᵀ` over `[μ-score; σ-score; noise-score]` channels;
  `X̄ = βX̄ + (1−β)a` over `[Δθ/σ²; Δlogσ]` (μ rows in score units, §4 Units);
  `G = βG + (1−β)X̄Aᵀ`; `Δμ = η_μ(F+λI)⁻¹(G·1₂)` with per-step η; warm-up 3
  steps. No windows, no per-phase strategy state. *(Supersedes V1/V2 windowed
  estimators; keeps V2's *influence* as |G| row norms.)*
- **V25** σ update variant A: `Δlogσ = η_σ·clip((z²−1)·mean(A), ±0.25)` with
  the phase-MA baselines; EWMA variant B deferred behind `TODO(sigma-ewma)`.
  *(Supersedes V13 — same math, single σ, sample-time snapshot.)*
- **V26** Noise is in `F` only: excluded from `X̄`/`G`; STDP δ = Σ(Δw)/N_syn
  per dt; Poisson δ = rate delta; STDP hyper-parameters realized scalar.
- **V27** Registry is the single NEST route — relays, readouts, devices, and
  producer recorders (only `nest.CopyModel` stays in `t2xcpg/utils.py`),
  dynamic agent sync, surfaced apply errors (incl. the per-device Poisson
  path), enforced by a source-scan test. *(Fixes the audit gaps.)*
- **V28** Realization cadence is decoupled from the recorded σ cadence via
  `realize_every_k` (default 1); strategies, recording and updates stay per dt
  when `k > 1`.
- **V29** β = 1 − dt/τ_mem, `τ_mem` default 2.0 s (β = 0.995 at 10 ms);
  advantage baselines keep α = 0.90.
- **V30** `run.h5` `agent/` gains `sigmas` (single), `trace_final`,
  `credit_final`; `stdp/` delta semantics per V26. *(Amends V17's `agent/`.)*
