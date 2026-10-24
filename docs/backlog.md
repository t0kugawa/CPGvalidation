**High**

* Phase/frequency entrainment and bilateral symmetry (R phase agreement ~0.12)
* Per-limb/phase credit assignment (stop summing anti-correlated advantages)
* 120 s game-on validation with `--realize-every-k 10` and a seed sweep

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

* Single-CPG runtime: replica path removed; NEST threads measured slower; keep
  `--nest-threads 1`
