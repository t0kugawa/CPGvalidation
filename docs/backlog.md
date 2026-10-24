**High**

* CPG drives both E/F to 1.0 and parks the ankle at a physical stop; make the
  E/F command reciprocal and audit control authority (2 s post-fix check)
* Phase/frequency entrainment and bilateral symmetry (R phase agreement ~0.12)
* Per-limb/phase credit assignment (stop summing anti-correlated advantages)
* 120 s game-on validation with `--realize-every-k 1` and a seed sweep
  (`k=10` is rejected for learning: 2 s last10 `0.276` vs `1.001`)

**Medium**

* Full conceptual model test (**with NoiseRecorders/without**, Game/no game...)
* Longer multi-seed, multi-reference game validation; per-player resource
  distributions, not just active counts
* Identifiability audit: freeze structural constants, group sites into
  functional players

**Low**

* Variance reduction for advantage estimates

**Cosmetics**

* Add timestamps to logs

**Done**

* MuJoCo ankle geometry/range: SOL insertion moved to the calcaneus;
  physical `[0.19, 1.76]` range; target clipping; regression tests
* Single-CPG runtime: replica path removed; NEST threads measured slower; keep
  `--nest-threads 1`
