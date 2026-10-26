**High**

* Ankle inversion/stuck gate: upper stop lowered from `1.76` to `1.10`
  (sole-up fixed; observed safe minimum 1.125); controlled ankle tone added
  (`stiffness=0.05`, `springref=1.0`); reciprocal phase-gated E/F and
  first-class PhaseClock added; TA swing force raised `2.8 → 5.6`; **1/5 gate
  PASSES** (7 cycles, period CV `0.196`, E/F `-3.033`, L/R drift `0.159`,
  angle corr `0.679`, no stuck/sole-up, render present); plan
  `docs/superpowers/plans/2026-10-25-ankle-inversion-stuck-fix.md`
* 1/2 and full-scale equivalence of the green 1/5 pattern; **1/2 FAIL**
  (period CV `0.326`, E/F `+0.920` std `0.976`, angle `0.360`; raw contacts
  ~9 but clock 14; raw CPG E/F corr `+0.69/+0.81`) and **full FAIL** (CV
  `0.206`, L/R drift `0.361`, angle `0.230`; right hip 0.34 rad short in swing,
  right foot 1.9 mm clearance, contact 5.69 N in swing; raw E/F corr
  `+0.61/+0.31`). The 1/5 pass is not an equivalence result; fix scale-invariant
  clock/CPG and right-leg swing clearance first. Shared-frequency diagnostic:
  full period CV `0.150` and L/R drift `0.286` pass, only angle `0.331` fails;
  right foot swing z `-0.0033 m`, contact `8.74 N`, right hip/knee ac
  `0.153/0.175` vs left `0.603/0.595`. Right-only gain/bias additions fail;
  next is a physical right-leg clearance/symmetry fix
* Separate game-selected validation arm (`game=None` is the gate arm)
* CPG drives both E/F to 1.0 and parks the ankle at a physical stop; make the
  E/F command reciprocal and audit control authority (2 s post-fix check)
* Phase/frequency entrainment and bilateral symmetry (R phase agreement ~0.12)
* Per-limb/phase credit assignment (stop summing anti-correlated advantages)
* 120 s game-on validation with `--realize-every-k 1` and a seed sweep
  (`k=10` is rejected for learning: 2 s last10 `0.276` vs `1.001`)

**Medium**

* Production run modes: `full`, `1/2`, `1/5` as first-class modes across
  `run_2min.py`, `run_validation.py`, and `personalize.py`; orthogonal to
  `--persona/--no-persona`, `--game`, seed, `realize_every_k`, and render;
  central size/provenance config. Current gaps: `run_2min` hardcodes full,
  `run_validation` has `--debug-small`, `personalize` has small/normal, and
  the phase gate has its own LITE 1/5. Return later.
* Full conceptual model test (**with NoiseRecorders/without**, Game/no game...)
* Longer multi-seed, multi-reference game validation; per-player resource
  distributions, not just active counts
* Identifiability audit: freeze structural constants, group sites into
  functional players
* PERSONA-Evo (typed covariance improves player redistribution). Redistributing orphaned segments via typed covariance — structural (from registry links) + functional (from Fisher/advantage), explicitly excluding global (shared hyperparameters/baseline) and noise (Poisson/PERSONA, via Noise Recorder) covariance — yields more stable Nash equilibria, better overlap with manual physiological segmentation, and fewer spurious captures than redistribution by raw covariance, with sensitivity analysis over thresholds. Design: `docs/superpowers/specs/2026-10-26-persona-evo-design.md`.

Measured 2026-10-26 (fast CPG, 141 segments/273 sites, seed 0, 3 generations,
400 steps): territory P=24 beats random on ARI (0.154 vs 0.0008), NMI (0.564
vs 0.412) and purity (0.454 vs 0.270); final 21 players. P=137 territory/random
ARI 0.054/0.038. Partition evaluation at 400 steps: evo-territory loss 0.148,
reference 0.199, random 0.176, manual 141 0.263, no violations; phase gate
fails for all arms and bio acceptance is false, so this is a partition result,
not a bio-plausibility result. Global-drive hypothesis is not supported in this
run: `pop_BS_E`/`pop_BS_F`/`pop_BASE_pg` land in players 0/3/3 and their
projections span players 0-5 (`other`). Report:
`docs/reports/persona_evo.md`.

**Strategy history / noise findings (2026-10-26)**

Synthetic `FisherMatrix` probe: rebuilding `F/Xbar/G` from exact per-step
`(sample, mu_at, sigma_at, dlogsigma, A)` is exact; rebuilding from the saved
post-step mean/sigma traces is ~99% wrong; dropping `Xbar` changes `G` and the
natural-gradient step by orders of magnitude. `F` and `Xbar` merge exactly only
with decay-aware weights; `G` is a bilinear `Xbar*A` EWMA and cannot be merged
exactly from per-segment `G` alone (35% error in a 3-way split). Current
`merge` is a normalized ensemble average, not an exact continuation;
`step_batch` and `merge` differ by 49% in the existing test. Parameter traces
are diagnostic, not sufficient statistics.

Noise recorders: producers are 10 Poisson sensory relays (CUT/Ia/Ib per side)
and 6 STDP projections. BS/BASE rates and site recorders are not producers.
Open issues: producer `scale` is fixed at registration (`Wmax`/rate can move);
`record_delta` overwrites pending (multiple writes per step lose all but the
last); the score `delta/R^2` uses `R` as a value, not a noise std (dimension
check needed); a site recorder stores the policy action, not the realized
per-neuron/per-synapse dispersion; zero-synapse projections still have sites.
Add tests for all producer configurations (all/none/Poisson-only/STDP-only/
per-relay), stale/duplicate/missing-scale deltas, and the fast-CPG A/B.
Design: `docs/superpowers/specs/2026-10-26-persona-strategy-history-noise-design.md`.

Measured 2026-10-26 (`bench_strategy_history.py`, 120 synthetic steps):
A-trace rebuild 0; P-trace ~1.00; no-Xbar ~4.24; merge F/Xbar/G
0.45/0.35/0.72. Fast-CPG 400 steps seed 0: configs all/none/poisson/stdp/
one-relay give n_noise 16/0/10/6/1, G-noise rows 0, A-trace error 0; wall
171-225 s per config. 100-step `all`: `noise_F_rel ≈ 5e-20` (absolute
`noise_F_norm` is scale-dominated). Report:
`docs/reports/strategy_history_noise.md`.

**Low**

* Variance reduction for advantage estimates

**Cosmetics**

* Add timestamps to logs
* Change the log to graphics processing script to make_clear_figures

**Done**

* MuJoCo ankle geometry/range: SOL insertion moved to the calcaneus;
  physical `[0.19, 1.76]` range; target clipping; regression tests
* Single-CPG runtime: replica path removed; NEST threads measured slower; keep
  `--nest-threads 1`
