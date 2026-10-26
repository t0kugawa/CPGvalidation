# PERSONA-Evo: registry segment matrix and evolutionary player selection — design

**Status:** decisions approved 2026-10-26. Opt-in training mode; no production
default change. It does not replace the current registry-derived players.

**Spec index:** `docs/superpowers/specs/2026-10-26-persona-evo-design.md`

## 1. Intent

The registry already knows every personalizable object: each population and
each projection is a physiological unit with its own parameters. Today the
conserved-stack game uses one such unit per player (`PhysicalMap.from_registry`,
137 players on the fast CPG). PERSONA-Evo asks a different question:

> Which groups of registry segments behave as one player under the game's
> fraction currency, and can those groups be discovered instead of hand-picked?

Evo therefore treats **segments as territory**. It starts from `P` randomly
initialized players built from whole segments, runs the normal game-on
personalization, removes players whose fraction reaches zero, redistributes
each orphan segment to its nearest surviving player, and repeats from the same
seeds until the partition stops changing (or the game's replicator residual is
small enough). The final partition is a reusable player map; the manual mode
remains the default.

Success is both structural and behavioral:

1. recovered players are close to a physiologically meaningful reference;
2. the recovered partition performs no worse than the manual 137-segment map
   on the fast-CPG benchmark and the bio-plausibility checks;
3. the result is deterministic from saved seeds and initial configurations.

## 2. Settled constraints

- **One registry object with optimizable parameters = one segment.** A
  population is a segment; a projection is a segment. The registry builds the
  segment matrix by design; Evo never invents a segment outside the registry.
- A **player is a set of whole segments**. Sites are never split from their
  segment and never dropped.
- The only player currency is the **fraction** `s_i`, `Σ s_i = 1`, exactly as in
  `StackGame`. Players do not see the global advantage.
- **Evo removes `share_floor`.** A fraction may reach zero; that is the prune
  signal. No parameter is left untouched because every orphan segment is
  redistributed.
- **Per-segment reassignment** is the only redistribution rule: each segment
  of a dead player goes to its nearest surviving player. Whole-player merger is
  not the default.
- **Restart from the same seed and initial configuration every generation.**
  The partition is the only variable. Seeds and every generation's partition
  are saved.
- **Both stop rules are tested:** partition fixed point and replicator
  residual. Neither is declared correct in advance.
- Evo requires the game on. H and Evo are mutually exclusive in v1.
- No change to `F`, `Xbar`, `G`, the sigma rule, advantage, credit, noise
  scoring, the recursive inverse, or the CPG/MuJoCo model.

## 3. Definitions

- **Segment** — one registry object with at least one optimizable
  `PersonaParam`; either a population (`pop_<name>`) or a projection
  (`conn_<key>`). Its `sites` are its canonical parameter keys.
- **Segment matrix** — the `n_segments × n_segments` typed adjacency plus a
  precomputed shortest-path distance matrix. It encodes physiological
  proximity, not parameter similarity.
- **Player** — a named, non-empty set of segment ids.
- **Partition** — a list of players that covers every segment exactly once.
- **Fraction** — a player's stack share `s_i`; initialized uniform.
- **Impact score** — the player's fraction at the end of a generation (mean
  over the final window is reported separately). A zero fraction means death.
- **Replicator residual** — `max_i |Δs_i|` over the final window, where
  `Δs_i` is the game's pre-projection transfer. It is the distance from the
  game's fraction fixed point.
- **Generation** — one full personalization run with a fixed partition, a
  fixed network/controller seed, and a fixed Evo seed.
- **Stable partition** — a generation that produces no zero-fraction player;
  because the restart is deterministic, running it again would give the same
  partition.
- **Replicator-stable partition** — stable partition and final-window
  `residual <= nash_tol`. This is the operational meaning of "Nash
  equilibrium" in this design. It is not a formal game-theoretic proof.

## 4. Registry segment matrix

### 4.1 Extraction

`persona/segments.py` builds the matrix from a live `Registry` after the
network is complete:

```
SegmentMatrix.from_registry(registry, *, include_shared=True) -> SegmentMatrix
```

A segment exists for:

- every `registry.population_cfgs[name]` with `pop.params` non-empty;
- every `registry.connection_cfgs[key]` with `conn.parameters` non-empty.

The segment list is sorted by canonical name for deterministic JSON and
reports. Each segment carries:

```
Segment(
    id: int,
    name: str,                 # "pop_rg_e_L" or "con_rg_e_L_m_e_L"
    kind: str,                 # "population" | "projection"
    param_names: tuple[str, ...],
    size: int,                 # neurons or synapses; 0 allowed
    ptype: str,                # exc | inh | poisson | static_exc | ...
    source: str | None,
    target: str | None,
    model: str | None,
    metadata: dict,            # zero_synapse, sign, limb hints, ...
)
```

Relay/parrot nodes without optimizable parameters (`BS_in_*`, `cut_in_*`,
`ia_*_in`, `ib_*_in`, `mus_*`) are not segments. A projection whose source or
target is such a relay still exists; its visible endpoint is the segment that
does exist (normally the target population).

`create_relay` must record the link `in_key -> (pg_key, pop_name)` in a new
registry mapping `relay_links` (and expose it as `registry.relay_links()`).
This is required because the BS/BASE rate segments are Poisson populations
whose efferents are registered from the parrot, not from the population
segment; without the link, `pop_BS_E`, `pop_BS_F` and `pop_BASE_pg` would have
no edge to anything and the global-drive hypothesis would be untestable.

A connection with `size == 0` is still a segment and keeps its sites; it is
flagged `metadata["zero_synapse"] = True` and reported separately. It is never
silently dropped.

### 4.2 Edges

The matrix starts empty and adds typed, undirected edges. A pair records the
set of types that proposed it; the stored weight is the maximum type weight.

| type | proposal | weight |
|---|---|---:|
| `incidence` | a projection and its source or target population segment | 1.00 |
| `antagonism` | two incoming projections to one target with opposite sign | 0.70 |
| `reciprocity` | projections `A→B` and `B→A` between the same pair of populations | 0.60 |
| `relay_drive` | a rate population segment and a projection whose source is its relay parrot (`registry.relay_links()`) | 1.00 |
| `shared_source` | two projections with the same source population (or relay) | 0.40 |
| `shared_target` | two projections with the same target population (or relay) | 0.40 |

`include_shared=True` adds the last two. They are the physiological fan-out
signal: two brainstem projections to different limbs are related through their
shared drive, and two sensory projections onto the same motor pool are related
through that pool. `relay_drive` is what connects `pop_BS_E`, `pop_BS_F` and
`pop_BASE_pg` to their four efferent projections each; it is always included.

### 4.3 Distance and proximity

`distance` is the unweighted shortest-path length over the edge set
(`scipy.sparse.csgraph.shortest_path(..., unweighted=True)`), computed once.
`proximity[i, j] = 1 / (1 + distance[i, j])`; disconnected pairs have distance
`inf` and proximity `0`.

The API exposes:

```
matrix.n_segments -> int
matrix.n_sites -> int
matrix.segment_of_site -> dict[str, int]
matrix.site_names -> tuple[str, ...]
matrix.segments -> tuple[Segment, ...]
matrix.adjacency -> np.ndarray            # (n, n) float
matrix.distance -> np.ndarray             # (n, n) float
matrix.nearest_segment(segment_id, candidates) -> int
matrix.nearest_player(segment_id, player_segments) -> int
matrix.validate_partition(partition) -> None
matrix.territory(seeds, rng) -> PlayerPartition
```

`nearest_player` uses `min(distance[segment_id, s] for s in player_segments)`;
ties are broken by higher summed edge weight, then by the Evo RNG.

Disconnected segments are handled structurally, never by string distance:

- **Territory initialization** computes the connected components of the matrix
  and allocates seeds per component proportionally to component size, with at
  least one seed per component. No segment starts without a reachable seed.
- **Redistribution** first tries `nearest_player`; if every surviving player is
  at infinite distance, the segment goes to the surviving player with the
  highest summed incident edge weight to the segment (shared source/target/
  antagonism/relay), then to the largest fraction, and the assignment is
  flagged `disconnected_fallback` in the redistribution log.
- If an entire component has no survivor, the same fallback is used; it is
  reported, not silently hidden.

### 4.4 Coverage validation

`validate_partition` raises if:

- a segment appears in two players or in none;
- a player is empty;
- a registry site is not mapped to exactly one segment;
- a partition refers to an unknown segment id or name.

The Evo trainer validates after initialization, after every redistribution, and
before every generation.

## 5. Player partition

`persona/partition.py` defines `PlayerPartition` (re-exported by
`persona/evo.py` so the public import is stable):

```
@dataclass(frozen=True)
class PlayerPartition:
    groups: tuple[tuple[int, ...], ...]   # segment ids, sorted inside a group
    def validate(self, matrix) -> None
    def to_json(self) -> dict
    @classmethod
    def from_json(cls, data) -> "PlayerPartition"
    def site_groups(self, matrix) -> list[tuple[str, ...]]
    def physical_map(self, matrix, *, name_prefix="player") -> PhysicalMap
```

JSON schema (version 1):

```json
{
  "version": 1,
  "n_segments": 137,
  "segment_names": ["con_BASE_in_rg_e_L", "..."],
  "groups": [
    {"name": "player_000", "segments": [3, 17, 42]},
    {"name": "player_001", "segments": [0, 1, 5]}
  ],
  "seed": 0,
  "init": "territory"
}
```

`physical_map` builds one `PlayerSpec` per group:

```
PlayerSpec(
    name="player_000",
    kind="segment_group",
    param_names=tuple(all sites of the group's segments),
    size=sum of segment sizes,
    ptype="mixed",
)
```

The resulting `PhysicalMap.player_of` maps every site to exactly one player;
`StackGame` consumes it unchanged. Group names are canonical (`player_000`,
`player_001`, …) after sorting groups by their smallest segment id.

## 6. Initialization

Both modes are deterministic from `(seed, P)` and are saved.

### 6.1 Territory growth (default)

1. Draw `P` distinct segment ids uniformly without replacement using
   `np.random.default_rng(seed ^ 0xE0)`.
2. Every seed owns itself.
3. Assign every remaining segment to its nearest seed by `matrix.distance`;
   ties are broken uniformly with the same RNG.
4. If a seed receives no other segment it remains a singleton player; this is
   allowed.
5. Validate coverage.

This is the requested "randomly initialized players" with territories grown on
the registry matrix.

### 6.2 Pure random (ablation)

Assign each segment a uniform random player id in `[0, P)`. If any player is
empty, move one segment from the largest group to each empty player, choosing
the moved segment uniformly, until every group is non-empty. Validate coverage.

`P > n_segments` is an error; `P == n_segments` is the one-segment-per-player
control.

## 7. One generation

```
run_generation(partition, config, generation, seed) -> GenerationResult
```

A production generation runs in a **fresh process** (`ProcessPoolExecutor`
with one worker, the `bench_game._run_fresh` pattern). Tests may use
`fresh_process=False`.

The run uses the normal closed loop:

- `build_personalizer(..., player_map=partition, game=evo_game_config,
  max_steps=config.steps_per_generation, seed=config.seed, **config.sizes)`;
- `GameConfig.current_best()` with `share_floor=0.0`;
- `eps_pose=0.10`, `pose_hold_steps=3`, `loss_stop=-1.0`, `loss_window=20`,
  `ref_hz=700.0`, `resolution_ms=0.2`, `belt_mps=0.2`,
  `belt_start_step=50`, `belt_ramp_steps=50`, `realize_every_k=1`,
  `pose_max_steps=100` (the fast-CPG benchmark settings);
- the same initial network parameters and seed as every other generation.

`GenerationResult` records:

- `partition`, `generation`, `seed`;
- per-step and final-window fractions;
- per-player `fraction_mean`, `fraction_final`, `fraction_min`,
  `zero_steps`, `contribution`, `a_hat`, `transfer`, `throttle`,
  `fisher_length`;
- `replicator_residual = max_i |mean_t transfer_i|` over the final window —
  the distance from the game's fraction fixed point;
- `movement = mean_t max_i |transfer_i|` over the final window — the per-step
  share movement, a noise diagnostic, **not** the fixed-point distance;
- `loss_eval`, `loss_full`, `r_p90`, `at_bound`, `q_gt_1`, `violations`,
  `active_players`, `stack_entropy`, `wall_s`, `step_ms`;
- the raw game diagnostics needed for the report.

The generation's **impact score** is `fraction_final`; `fraction_mean` is
reported for stability. A player is dead when
`fraction_final <= max(config.impact_eps, 1e-12)`.

### 7.1 Zero fraction and the simplex

Evo sets `GameConfig.share_floor = 0.0`. `GameConfig.__post_init__` accepts
`share_floor >= 0.0`; manual mode keeps `1e-12`. After the transfer,
`StackGame` projects with `max(s, 0)` and normalizes; if the sum is not
positive (all players zero), it restores the previous simplex and reports an
error in `last` instead of dividing by zero. No site is removed by this guard.

## 8. Pruning and per-segment redistribution

For each dead player, in increasing player id:

1. Remove the player.
2. For each of its segments, assign the segment to the surviving player with
   the smallest `matrix.nearest_player` distance. Ties use summed edge weight,
   then the Evo RNG.
3. Record `(segment, from_player, to_player, distance, edge_types)` in the
   redistribution log.
4. If no surviving player remains, keep the dead player with the largest
   fraction as `player_000` and skip redistribution; report `all_dead`.
5. Renumber and sort groups, validate coverage.

No segment is dropped and no site is left untouched. A redistribution that
splits a dead player across several survivors is expected and correct.

## 9. Stop rules

`config.stop` is one of `partition`, `nash`, `both`.

- **`partition`** — stop after `patience` consecutive generations with no dead
  player (default `patience=2`). With deterministic restart this is already a
  fixed point; patience guards numerical drift.
- **`nash`** — stop when no player is dead and
  `replicator_residual <= config.nash_tol` (default `1e-3`) in the final
  window. If the run ends without reaching the tolerance, the generation is
  marked `partition_stable_nash_failed`, and the trainer may extend the run by
  `config.extension_steps` (default `0`) before deciding.
- **`both`** (default) — require both conditions.
- **`min_players` floor** — if pruning would leave fewer than
  `config.min_players` survivors, stop **before** pruning with
  `reason="min_players"`, keep the last valid partition, and set
  `floor_hit=true`. The trainer never fabricates a survivor by keeping a
  zero-fraction player as a "survivor"; the report says the floor was hit.
- **cycle guard** — hash the canonical partition. Because pruning is
  monotone (players are removed, never added or split), a repeated partition
  is normally a stable partition; the cycle guard is a safety net for future
  non-monotone rules. It is checked only after the stability/min-floor checks.
- **max generations** — always stops with `reason="max_generations"`.

Stop-reason precedence, evaluated after each generation:

1. `all_dead` (the all-dead guard fired);
2. `min_players` (the next prune would go below the floor);
3. `partition` / `nash` / `both` according to `config.stop`;
4. `partition_stable_nash_failed` (no deaths, partition stable, but
   `stop` includes `nash` and the residual is above tolerance after
   `extension_steps`);
5. `cycle`;
6. `max_generations`.

The trainer never claims a formal Nash equilibrium. The report always prints
`replicator_residual`, `movement`, `fraction_stability`, `partition_changed`,
`floor_hit`, and the stop reason.

## 10. Integration

### 10.1 `Agent`

`Agent(..., player_map: PlayerPartition | None = None)`. When the game is
enabled:

- `player_map is None` → current `PhysicalMap.from_registry(registry)`;
- `player_map` → `player_map.physical_map(matrix)` where `matrix` is built from
  the same registry.

The strategy is unchanged: it still owns site-level `F/Xbar/G` and site
sampling. The partition only changes `StackGame`'s `player_of` and
`_player_sites`.

Evo + `homeostasis` raises `ValueError` in v1.

### 10.2 `personalize.py`

`build_personalizer(..., player_map=None)` passes the map to `Agent`.
`run_personalization(..., training="manual", player_map=None, evo=None)`:

- `training="manual"` reproduces today's behavior exactly;
- `training="evo"` builds the segment matrix after the CPG is constructed,
  runs `EvoTrainer`, writes `evo/`, optionally runs the winning partition with
  the normal `--steps` horizon and `training="manual"` semantics;
- `--player-map PATH` loads a saved partition and runs it in manual mode.

### 10.3 CLI

`persona/cli.py` adds:

```
--training {manual,evo}          default manual
--player-map PATH                default None
--evo-players INT                default 24
--evo-generations INT            default 6
--evo-steps INT                  default 400
--evo-eval-window INT            default 50
--evo-init {territory,random}    default territory
--evo-stop {partition,nash,both} default both
--evo-patience INT               default 2
--evo-nash-tol FLOAT             default 1e-3
--evo-impact-eps FLOAT           default 0.0
--evo-min-players INT            default 2
--evo-seed INT                   default None (use --seed)
--evo-out PATH                   default <run out>/evo
--evo-fresh-process / --no-evo-fresh-process   default fresh
--evo-final-run / --no-evo-final-run          default on
```

`--training evo` with `--game none` is an error. `--player-map` with
`--training evo` is an error (Evo builds its own map; use manual mode to reuse
one). `--evo-players` must be in `[2, n_segments]` (or `1` for a degenerate
control). `--evo-min-players` is the stop floor described in §9: it is not a
promise that a dead player survives; when the floor is hit the run stops and
reports `floor_hit=true`.

### 10.4 Outputs

`evo/` contains:

- `evo_config.json` — all config, seed, sizes, game config, git commit;
- `initial_partition.json` — the exact initial territory/random partition;
- `gen_000.json` … — per-generation partition, fractions, metrics, residual,
  redistribution log, wall time;
- `final_partition.json` — reusable map for `--player-map`;
- `history.csv` / `history.json` — one row per generation;
- `report.md` — human-readable summary with the reference comparison and the
  global-drive verdict;
- `evo_metrics.json` — machine-readable final metrics.

### 10.5 Benchmark harness

`bench_evo.py` mirrors `bench_game.py`: `--smoke` uses one seed, 400 steps,
`P=24`, one generation; `--compare` runs the full matrix (init × stop × P ×
seeds) with fresh processes and writes `benchmarks/evo/`. It never changes
production defaults.

## 11. Physiological reference and evaluation

The reference is **evaluation only**; it never trains Evo. It is a hierarchy,
so a partition can be compared at the granularity it actually discovered.

### 11.1 Level 0 — segment

The 137 registry segments. This is the current manual map.

### 11.2 Level 1 — population + efferents (~37)

Rule: a population segment and every projection whose **source** is that
population form one module. Projections whose source is a relay/parrot
(`BS_in_*`, `BASE_in`, `cut_in_*`, `ib_*_in`) are assigned by their afferent
system:

- `BS_in_*`, `BASE_in` → `global_drive`;
- `cut_in_*` → the limb's `cutaneous`;
- `ib_*_in` → the limb's `ib`.

Populations with no efferents remain singleton modules. Every segment gets
exactly one Level-1 label.

### 11.3 Level 2 — functional modules (~19)

| label | members |
|---|---|
| `global_drive` | BS_E, BS_F, BASE_pg and all BS/BASE projections to both limbs |
| `<side>_rg_ext` | RG-E population, its self-projection, RG-E efferents |
| `<side>_rg_flex` | RG-F population, its self-projection, RG-F efferents |
| `<side>_recip_inh` | In-E, In-F and their efferents |
| `<side>_motor` | M-E, M-F and their efferents |
| `<side>_renshaw` | R-E, R-F and their efferents |
| `<side>_ia` | Ia-E, Ia-F, Ia-E/F input interneurons and their efferents |
| `<side>_ib` | In-Ib-E, In-Ib-F, relay-sourced Ib projections, and efferents |
| `<side>_cutaneous` | CUT→RGE STDP and CUT→In-E |
| `<side>_commissural` | V2a, V0v, V3f, V0d, In1 and their efferents |

`<side>` is `L` or `R`. The exact member list is generated by
`persona/physo_reference.py` from the registry names and printed in the report;
the rule above is the contract. The global drive is deliberately one module:
its rate parameters are shared across limbs, while its projection weights are
limb-specific. The report tracks both the population and projection fate.

### 11.4 Metrics

- Partition agreement: ARI, NMI, V-measure (homogeneity and completeness),
  per-Evo-player purity, per-reference-module coverage, pairwise F1.
  Implemented in pure numpy; no scikit-learn dependency.
- Combined with the performance metrics below on the fast CPG.

### 11.5 Performance comparison

On the same fast-CPG benchmark, same seed and steps:

| arm | partition |
|---|---|
| `manual_segments` | 137 registry segments |
| `reference_functional` | Level-2 reference (~19) |
| `evo_territory` | Evo final partition (territory init) |
| `evo_random` | Evo final partition (random init) |

Report `loss_eval`, `r_p90`, `at_bound`, `q_gt_1`, `violations`,
`active_players`, `stack_entropy`, phase-gate metrics, `biological_acceptance`,
and `LocomotionMetrics`.

Acceptance is a report, not a hidden pass/fail:

- Evo vs reference: ARI/NMI above a permutation baseline; each Evo player has
  a dominant reference label (purity reported).
- Global drive: classify per seed as `same_player`, `split_by_limb`,
  `rate_global_projections_limb`, `absorbed`, or `other`; report the hypothesis
  verdict honestly.
- Performance: no seed loses more than 10% `loss_eval` vs `manual_segments`,
  no type/sign violations, and p90 does not worsen systematically. If Evo
  fails this, say so.
- Determinism: same seed and config reproduce the same partition and metrics.

## 12. Tests

Fast, simulator-free unit tests:

- `tests/test_segment_matrix.py` — segment extraction counts, one segment per
  object, site coverage, edge types/weights, distance symmetry, zero-synapse
  flag, deterministic order.
- `tests/test_player_partition.py` — validation (empty, overlap, missing,
  unknown), JSON round-trip, `physical_map`, site groups.
- `tests/test_evo_init.py` — territory determinism, every segment assigned,
  random init coverage, no empty groups, `P > n_segments` error.
- `tests/test_evo_prune.py` — zero fraction prunes, per-segment nearest
  reassignment, tie-break, coverage after redistribution, all-dead guard,
  no site lost.
- `tests/test_evo_stop.py` — partition stop, Nash stop, both, cycle guard,
  max generations, `partition_stable_nash_failed`.
- `tests/test_evo_game.py` — `StackGame` with a partition, `share_floor=0`,
  zero fraction, normalization guard, no H.
- `tests/test_evo_cli.py` — manual default, evo args, game-none error,
  player-map error, H error.
- `tests/test_physo_reference.py` — Level-1 and Level-2 labels cover all 137
  segments exactly once; known memberships (BS/BASE global, per-limb groups).

Simulator smoke (WSL CPGV, marked slow):

- `tests/test_evo_integration.py` — tiny CPG, one generation, territory init,
  outputs written, coverage valid, deterministic partition from the same seed.

Empirical:

- `python bench_evo.py --smoke` (P=24, 400 steps, 1 seed, 1 generation).
- `python bench_evo.py --compare` after the smoke is green; full tables in
  `benchmarks/evo/`.

## 13. Non-goals and risks

- Not a replacement for manual players; `manual` remains the default.
- No change to the CPG, plant, PERSONA update, sigma, advantage, credit, noise,
  or Fisher math.
- No H in Evo v1; no freezing; no per-site reassignment.
- No formal Nash proof. "Nash" means the replicator residual plus partition
  stability, both reported.
- Risks: exact zero fractions may be rare for small `P`; `P` sensitivity; the
  global-drive segments may not admit a unique physiological owner; zero-
  synapse segments are structurally dead; fresh-process runs add overhead;
  noisy online contributions may keep the residual above tolerance.
- Functional (Fisher/advantage) affinity is deliberately excluded from
  redistribution in v1. The backlog's typed-covariance idea is a later
  ablation, not part of this design.

## 14. Decisions left to the builder

- Exact edge weights and the distance algorithm are fixed above; a builder may
  add a type without changing the contract.
- JSON field names and report formatting are fixed by the plan.
- `patience=2`, `nash_tol=1e-3`, `impact_eps=0.0`, `min_players=2` are the
  starting values; the experiment sweeps them rather than hard-coding a
  winner. `replicator_residual` is the fixed-point distance
  `max_i |mean_t transfer_i|`; `movement` is the separate noise diagnostic.
- The final run's horizon (`--steps`) and the generation horizon
  (`--evo-steps`) are separate; Evo never silently changes the manual horizon.
- Snapshots/partitions are archival documents, not a promise of a resumable
  Agent: the Agent/phase/game/H/RNG state is outside the Evo partition file.

## 15. Implementation order

1. Segment matrix and coverage tests.
2. Partition object, JSON, physical map.
3. Evo initialization (territory + random) and determinism tests.
4. Pruning, redistribution, stop rules, and tests.
5. `GameConfig.share_floor=0`, `Agent`/`StackGame` partition wiring, CLI.
6. Physiological reference hierarchy and metrics.
7. `bench_evo.py` smoke, then the scaled experiment.
8. Report with the reference comparison and the global-drive verdict.
