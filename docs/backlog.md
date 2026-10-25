**High**

* Ankle inversion/stuck gate: upper stop lowered from `1.76` to `1.10`
  (sole-up fixed; observed safe minimum 1.125); controlled ankle tone added
  (`stiffness=0.05`, `springref=1.0`); reciprocal phase-gated E/F and
  first-class PhaseClock added; TA swing force raised `2.8 → 5.6`; **1/5 gate
  PASSES** (7 cycles, period CV `0.196`, E/F `-3.033`, L/R drift `0.159`,
  angle corr `0.679`, no stuck/sole-up, render present); plan
  `docs/superpowers/plans/2026-10-25-ankle-inversion-stuck-fix.md`
* 1/2 and full-scale equivalence of the green 1/5 pattern; the pass is close
  to the period-CV limit (`0.196` vs `0.20`), so do not assume equivalence
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
