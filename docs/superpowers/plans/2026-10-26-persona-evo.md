# PERSONA-Evo implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--training evo` mode that discovers player groups of registry segments with the conserved-stack game, and compare the recovered partitions with the registry/physiological reference on the fast CPG.

**Architecture:** A registry-built segment matrix supplies typed proximity; `PlayerPartition` groups whole segments; `EvoTrainer` restarts the normal game-on personalization from the same seed each generation, prunes zero-fraction players, and reassigns each orphan segment to its nearest surviving player. The existing site-level PERSONA update is untouched.

**Tech Stack:** Python 3.12, NumPy, SciPy, pytest, NEST/MuJoCo for the integration and fast-CPG runs (WSL `CPGV`).

**Spec:** `docs/superpowers/specs/2026-10-26-persona-evo-design.md`

## Global Constraints

- Manual mode is the default and must be bit-for-bit unchanged when `training="manual"` and `player_map=None`.
- One registry object with optimizable parameters = one segment; players are whole segments; no site is ever dropped or split from its segment.
- Evo removes `share_floor` only in Evo mode; manual `GameConfig` keeps `1e-12`.
- Restart from the same network/controller seed and Evo seed every generation; save the initial partition and every generation.
- Per-segment reassignment only; no whole-player merger as the default.
- Evo requires the game on and H off.
- Do not change `F`, `Xbar`, `G`, sigma, advantage, credit, noise, or the CPG/MuJoCo model.
- The workspace has uncommitted user work (`model/leg.py`, `model/cut.py`, `t2xcpg/*`, `persona/phase_metrics.py`, `personalize.py`, deleted phase-clock files). Make additive edits only; never reset them.
- Fast tests on Windows Python; NEST tests with `/home/t0kgw/miniconda3/envs/CPGV/bin/python` under WSL.

## Review Focus

- `pop_BS_E`, `pop_BS_F`, `pop_BASE_pg` must not be isolated in the matrix; the relay-drive edge is the fix, and a test must show it.
- A partition with a zero-fraction player must never leave a segment unassigned after redistribution; Task 5.
- Restarting with the same seed must produce the same initial partition; Task 4.
- `share_floor=0` must not divide by zero when every fraction collapses; Task 7.
- A saved `--player-map` must round-trip through JSON and reconstruct the same `StackGame` players; Task 3/9.

---

### Task 1: Registry relay links

**Files:**
- Modify: `persona/adapter.py` (default `relay_links`)
- Modify: `persona/nest_adapter.py:39-56, 408-473`
- Modify: `persona/neuron_adapter.py:69-195, 473-548`
- Test: `tests/test_relay_links.py` (new)

**Interfaces:**
- Produces: `registry.relay_links() -> dict[str, tuple[str, str]]`, mapping
  `in_key -> (pg_key, pop_name)`; `pop_name` may be `""`.

- [ ] **Step 1: Write the failing test**

```python
from t2xcpg.bs import BrainStem, BaseDrive

def test_bs_and_base_relays_are_linked(nest_kernel):
    from persona.registry import Registry
    reg = Registry()
    BrainStem(registry=reg, n_bs=4)
    BaseDrive(registry=reg, n_bs=4)
    links = reg.relay_links()
    assert links["BS_in_e"] == ("bs_pg_e", "BS_E")
    assert links["BS_in_f"] == ("bs_pg_f", "BS_F")
    assert links["BASE_in"] == ("base_pg", "BASE_pg")
```

- [ ] **Step 2: Run test to verify it fails**

Run (WSL CPGV): `python -m pytest tests/test_relay_links.py -q`
Expected: FAIL with `AttributeError: 'Registry' object has no attribute 'relay_links'`.

- [ ] **Step 3: Implement the link**

Add `self.relay_links: dict[str, tuple[str, str]] = {}` to both adapters.
In `create_relay`, after `register_nodes(in_key, parrot)`, set
`self.relay_links[in_key] = (pg_key, name or "")`. Add
`def relay_links(self) -> dict[str, tuple[str, str]]: return dict(self.relay_links)`
to `Adapter` as a concrete default returning `{}`; override in both adapters.

- [ ] **Step 4: Run the test**

Run (WSL CPGV): `python -m pytest tests/test_relay_links.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/adapter.py persona/nest_adapter.py persona/neuron_adapter.py tests/test_relay_links.py
git commit -m "feat(persona): expose relay-to-population links"
```

---

### Task 2: Segment matrix

**Files:**
- Create: `persona/segments.py`
- Test: `tests/test_segment_matrix.py` (new)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class Segment:
    id: int
    name: str
    kind: str                 # "population" | "projection"
    param_names: tuple[str, ...]
    size: int
    ptype: str
    source: str | None = None
    target: str | None = None
    model: str | None = None
    metadata: dict = field(default_factory=dict, compare=False)

@dataclass
class SegmentMatrix:
    segments: tuple[Segment, ...]
    segment_of_site: dict[str, int]
    site_names: tuple[str, ...]
    adjacency: np.ndarray
    edge_types: dict[tuple[int, int], frozenset[str]]
    distance: np.ndarray
    @classmethod
    def from_registry(cls, registry, *, include_shared=True) -> "SegmentMatrix"
    def nearest_segment(self, segment_id, candidates) -> int
    def nearest_player(self, segment_id, player_segments) -> int
    def validate_partition(self, partition) -> None
```

- [ ] **Step 1: Write the failing test**

Use a small fake registry (populations A, B, C and connections A→B, B→C,
A→C with opposite signs) and assert:

```python
m = SegmentMatrix.from_registry(fake)
assert [s.name for s in m.segments] == ["con_A_B", "con_A_C", "con_B_C",
                                         "pop_A", "pop_B", "pop_C"]
assert m.segment_of_site["pop_A_a"] == 3
assert ("incidence" in m.edge_types[(0, 3)])
assert np.allclose(m.distance, m.distance.T)
assert np.all(np.diag(m.distance) == 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_segment_matrix.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'persona.segments'`.

- [ ] **Step 3: Implement extraction**

Build one segment per parameterized population and projection. Sort by name.
Use `_connection_sign`, `_population_type`, `_synapse_type` from
`persona.homeostasis` for `ptype`/sign; import them rather than copying.
Record `metadata["zero_synapse"] = size == 0` when the registry can report
`size` (projections with no synapses).

- [ ] **Step 4: Implement edges and distance**

Add undirected edges with the types/weights from the spec (§4.2), including
`relay_drive` via `registry.relay_links()`. Keep the maximum weight per pair
and the set of types. Build a SciPy sparse matrix and run
`scipy.sparse.csgraph.shortest_path(..., unweighted=True)`; keep `inf` for
disconnected pairs. Implement `nearest_segment`/`nearest_player` with
deterministic tie-breaks (summed incident weight, then segment id).

- [ ] **Step 5: Add the NEST inventory test**

On the fast CPG (small sizes), assert 137 segments, 269 sites, every site in
exactly one segment, `pop_BS_E`/`pop_BS_F`/`pop_BASE_pg` have finite distance to
their efferent projections, and the two new `in_e→m_f`/`in_f→m_e` projections
are present. Mark the test `phase_gate` or `slow` consistent with the repo.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_segment_matrix.py -q`
Run (WSL CPGV): the NEST inventory test.
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add persona/segments.py tests/test_segment_matrix.py
git commit -m "feat(persona): add registry segment matrix"
```

---

### Task 3: `PlayerPartition` and physical map

**Files:**
- Create: `persona/partition.py`
- Test: `tests/test_player_partition.py` (new)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class PlayerPartition:
    groups: tuple[tuple[int, ...], ...]
    def validate(self, matrix) -> None
    def to_json(self) -> dict
    @classmethod
    def from_json(cls, data) -> "PlayerPartition"
    def site_groups(self, matrix) -> list[tuple[str, ...]]
    def physical_map(self, matrix, *, name_prefix="player") -> PhysicalMap
```

- [ ] **Step 1: Write the failing test**

```python
p = PlayerPartition(groups=((0, 2), (1, 3)))
p.validate(matrix)
assert p.to_json()["groups"][0]["segments"] == [0, 2]
assert PlayerPartition.from_json(p.to_json()) == p
pm = p.physical_map(matrix)
assert pm.n_players == 2
assert pm.player_of["pop_A_a"] in (0, 1)
with pytest.raises(ValueError):
    PlayerPartition(groups=((0, 1), (1, 2))).validate(matrix)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_partition.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement partition**

Validate: non-empty groups, disjoint ids, every segment covered exactly once,
known ids. Canonicalize by sorting groups by their smallest id and sorting ids
inside a group. JSON schema version 1 from the spec. `physical_map` builds
`PlayerSpec(name=f"{name_prefix}_{i:03d}", kind="segment_group",
param_names=all sites, size=sum sizes, ptype="mixed")` and returns
`PhysicalMap(specs)`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_player_partition.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/partition.py tests/test_player_partition.py
git commit -m "feat(persona): add player partition and physical map"
```

---

### Task 4: Territory and random initialization

**Files:**
- Create: `persona/evo.py`
- Test: `tests/test_evo_init.py` (new)

**Interfaces:**
- Produces:

```python
def territory_partition(matrix, n_players: int, seed: int) -> PlayerPartition
def random_partition(matrix, n_players: int, seed: int) -> PlayerPartition
```

- [ ] **Step 1: Write the failing test**

```python
m = SegmentMatrix.from_registry(fake)
a = territory_partition(m, 3, seed=0)
b = territory_partition(m, 3, seed=0)
assert a == b
a.validate(m)
assert len(a.groups) == 3 and all(a.groups)
c = random_partition(m, 3, seed=0)
c.validate(m)
assert len(c.groups) == 3 and all(c.groups)
with pytest.raises(ValueError):
    territory_partition(m, m.n_segments + 1, seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evo_init.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement territory growth**

Use `np.random.default_rng(seed ^ 0xE0)`. Compute connected components. If
`n_players < n_components`, raise `ValueError`. Allocate seeds per component
proportionally to component size, at least one each. Draw distinct seeds
within each component. Assign the rest to the nearest seed by
`matrix.distance`; ties uniformly with the RNG. Return a canonical
`PlayerPartition`.

- [ ] **Step 4: Implement pure random**

Assign each segment a uniform id; fill empty groups by moving a uniformly
chosen segment from the largest group. Validate and canonicalize.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_evo_init.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add persona/evo.py tests/test_evo_init.py
git commit -m "feat(persona): add territory and random Evo initialization"
```

---

### Task 5: Pruning and per-segment redistribution

**Files:**
- Modify: `persona/evo.py` (add `prune_and_redistribute`)
- Test: `tests/test_evo_prune.py` (new)

**Interfaces:**
- Produces:

```python
@dataclass
class Redistribution:
    segment: int
    from_player: int
    to_player: int
    distance: float
    edge_types: tuple[str, ...]
    fallback: bool = False

def prune_and_redistribute(partition, fractions, matrix, *, seed,
                           impact_eps=0.0, min_players=2
                           ) -> tuple[PlayerPartition, list[Redistribution], str]
```

- [ ] **Step 1: Write the failing test**

```python
p = PlayerPartition(groups=((0,), (1,), (2,)))
new, log, reason = prune_and_redistribute(
    p, fractions=np.array([1.0, 0.0, 0.0]), matrix=m,
    seed=0, impact_eps=0.0, min_players=1)
assert reason == "pruned"
assert len(new.groups) == 1
assert sorted(s for g in new.groups for s in g) == [0, 1, 2]
assert {r.segment for r in log} == {1, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evo_prune.py -q`
Expected: FAIL with `NameError`.

- [ ] **Step 3: Implement pruning**

Dead if `fraction <= max(impact_eps, 1e-12)`. If all are dead, keep the
largest-fraction player as `player_000`, return `"all_dead"`. If survivors
would be fewer than `min_players`, return the input unchanged with
`"min_players"`. Otherwise remove dead players, reassign each dead segment to
the nearest surviving player, use the disconnected fallback, renumber, validate
coverage, and return `"pruned"`.

- [ ] **Step 4: Add tie-break and fallback tests**

Use a matrix with two equally distant survivors and assert the deterministic
tie-break; use a disconnected segment and assert `fallback=True` and that the
segment is still assigned.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_evo_prune.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add persona/evo.py tests/test_evo_prune.py
git commit -m "feat(persona): prune zero-fraction players by segment proximity"
```

---

### Task 6: Stop rules and `EvoTrainer`

**Files:**
- Modify: `persona/evo.py` (config, results, trainer)
- Create: `persona/evo_runner.py` (production runner, lazy NEST import)
- Test: `tests/test_evo_stop.py` (new)

**Interfaces:**
- Produces:

```python
@dataclass
class EvoConfig:
    n_players: int = 24
    generations: int = 6
    steps_per_generation: int = 400
    eval_window: int = 50
    init: str = "territory"          # territory | random
    stop: str = "both"               # partition | nash | both
    patience: int = 2
    nash_tol: float = 1e-3
    impact_eps: float = 0.0
    min_players: int = 2
    seed: int = 0
    extension_steps: int = 0
    fresh_process: bool = True

@dataclass
class GenerationResult:
    partition: PlayerPartition
    generation: int
    seed: int
    fractions: np.ndarray
    fraction_mean: np.ndarray
    contribution: np.ndarray
    transfer: np.ndarray
    replicator_residual: float
    movement: float
    metrics: dict
    wall_s: float

@dataclass
class EvoResult:
    final_partition: PlayerPartition
    history: list[dict]
    stop_reason: str
    equilibrium: dict

class EvoTrainer:
    def __init__(self, matrix, config, runner): ...
    def run(self) -> EvoResult: ...
```

- [ ] **Step 1: Write the failing test with a fake runner**

```python
def test_partition_stop_when_no_deaths():
    calls = []
    def runner(partition, generation, seed):
        return GenerationResult(partition=partition, generation=generation,
                                seed=seed, fractions=np.ones(len(partition.groups)),
                                fraction_mean=np.ones(len(partition.groups)),
                                contribution=np.ones(len(partition.groups)),
                                transfer=np.zeros(len(partition.groups)),
                                replicator_residual=0.0, movement=0.0,
                                metrics={}, wall_s=0.0)
    trainer = EvoTrainer(m, EvoConfig(n_players=2, stop="partition",
                                      patience=2, seed=0), runner)
    result = trainer.run()
    assert result.stop_reason == "partition"
    assert len(calls) == 2
```

Use a `calls` list inside the runner; assert the same initial partition is
passed every generation.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evo_stop.py -q`
Expected: FAIL with `NameError`.

- [ ] **Step 3: Implement the trainer state machine**

Follow the spec's precedence: `all_dead`, `min_players`, `partition`/`nash`/
`both`, `partition_stable_nash_failed`, `cycle`, `max_generations`. The trainer
builds the initial partition from `config.init`, calls `runner` once per
generation, records `GenerationResult`, and calls `prune_and_redistribute`.
`replicator_residual` is the runner-provided fixed-point distance;
`movement` is separate. Save `history` as plain dicts.

- [ ] **Step 4: Add stop-rule tests**

Cover: nash stop, both, patience not yet reached, partition_stable_nash_failed,
min_players floor, all_dead, max_generations, and a fake cycle. Each test
asserts the exact `stop_reason`.

- [ ] **Step 5: Add the production runner**

`persona/evo_runner.py`:

```python
def run_generation(partition, config, generation, seed) -> GenerationResult
def run_generation_fresh(partition, config, generation, seed) -> GenerationResult
```

`run_generation` imports `personalize.build_personalizer` lazily, builds the
personalizer with `player_map=partition`, `game=GameConfig.current_best()` with
`share_floor=0.0`, runs with the fast-CPG settings, and reads
`agent.history["game"]` for fractions/contribution/transfer. Compute
`replicator_residual = max_i |mean_t transfer_i|` and
`movement = mean_t max_i |transfer_i|` over the final window.
`run_generation_fresh` wraps `run_generation` in a one-worker
`ProcessPoolExecutor` (the `bench_game._run_fresh` pattern).

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_evo_stop.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add persona/evo.py persona/evo_runner.py tests/test_evo_stop.py
git commit -m "feat(persona): add Evo stop rules and trainer"
```

---

### Task 7: `share_floor=0` and simplex guard

**Files:**
- Modify: `persona/game.py:24-112, 289-342`
- Test: `tests/test_game.py` (append) or `tests/test_evo_game.py` (new)

**Interfaces:**
- Produces: `GameConfig(share_floor=0.0)` is valid for Evo; `StackGame.apply`
  never divides by zero.

- [ ] **Step 1: Write the failing test**

```python
cfg = GameConfig(enabled=True, share_floor=0.0)
game = StackGame(map_two_players, cfg, site_names=["a", "b"])
out = game.apply(np.array([1.0, -1.0]), np.array([0.0, 0.0]),
                 steps=1000, sigmas=np.array([0.1, 0.1]))
assert np.all(np.isfinite(game.stacks))
assert np.isclose(game.stacks.sum(), 1.0)
```

Also force an all-zero transfer and assert the previous simplex is restored
and `game.last.get("error") == "all_zero"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evo_game.py -q`
Expected: FAIL with `ValueError: share_floor must be positive`.

- [ ] **Step 3: Implement**

Change the validation to `share_floor >= 0.0`. In `apply`, after
`new = np.maximum(new, share_floor)`, compute `total = float(new.sum())`; if
`total <= eps`, restore `self.stacks` from before the transfer, set
`error="all_zero"`, and skip normalization. Manual defaults unchanged.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_evo_game.py tests/test_game.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/game.py tests/test_evo_game.py tests/test_game.py
git commit -m "feat(persona): allow zero share floor with a simplex guard"
```

---

### Task 8: Agent and `build_personalizer` player map

**Files:**
- Modify: `persona/agent.py:27-112`
- Modify: `personalize.py:61-217`
- Test: `tests/test_evo_agent.py` (new)

**Interfaces:**
- Produces: `Agent(..., player_map: PlayerPartition | None = None)` and
  `build_personalizer(..., player_map=None)`.

- [ ] **Step 1: Write the failing test**

```python
partition = PlayerPartition(groups=((0,), (1,)))
agent = Agent(registry=reg, game=GameConfig.current_best(),
              player_map=partition)
assert agent.game.n_players == 2
assert set(agent.game._site_names) == set(agent.strategy.param_names)
```

Use a fake registry with the segment matrix built from it. Also assert
`Agent(..., game=..., homeostasis=HomeostaticConfig.current_best(),
player_map=partition)` raises `ValueError`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evo_agent.py -q`
Expected: FAIL with `TypeError`.

- [ ] **Step 3: Implement**

In `Agent.__init__`, when `game` is enabled:

```python
from persona.segments import SegmentMatrix
matrix = SegmentMatrix.from_registry(registry)
physical_map = (player_map.physical_map(matrix) if player_map is not None
                else PhysicalMap.from_registry(registry))
```

Use `physical_map` for `StackGame`. If `homeostasis` is enabled and
`player_map` is not None, raise `ValueError("Evo and H are mutually
exclusive")`. `build_personalizer` passes `player_map` to `Agent`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_evo_agent.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/agent.py personalize.py tests/test_evo_agent.py
git commit -m "feat(persona): wire player partitions into the agent"
```

---

### Task 9: CLI, training mode, outputs

**Files:**
- Modify: `persona/cli.py:7-44`
- Modify: `personalize.py:220-366`
- Test: `tests/test_evo_cli.py` (new)

**Interfaces:**
- Produces: `--training {manual,evo}`, `--player-map PATH`, and the `--evo-*`
  flags from the spec; `run_personalization(..., training="manual",
  player_map=None, evo: EvoConfig | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
def test_training_defaults_to_manual():
    args = build_parser().parse_args([])
    assert args.training == "manual"
    assert args.evo_players == 24

def test_evo_rejects_game_none():
    args = build_parser().parse_args(["--training", "evo", "--game", "none"])
    with pytest.raises(SystemExit):
        validate_args(args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evo_cli.py -q`
Expected: FAIL with unrecognized arguments.

- [ ] **Step 3: Add the flags and validation**

Add all `--evo-*` flags from the spec. `validate_args` (new helper in
`persona/cli.py`) rejects evo + game none, player-map + evo, and evo + H (H is
not a CLI flag today, so the H check lives in `Agent`).

- [ ] **Step 4: Implement `training="evo"`**

In `run_personalization`, when `training="evo"`:

1. build the personalizer once to obtain the registry (or add a
   `build_registry_only` helper);
2. build `SegmentMatrix` and `EvoTrainer` with the fresh-process runner;
3. write `evo_config.json`, `initial_partition.json`, `gen_*.json`,
   `final_partition.json`, `history.csv/json`, `report.md`;
4. if `--evo-final-run`, run the winning partition with the normal
   `--steps` horizon and manual semantics.

`--player-map` loads a `PlayerPartition` and passes it in manual mode.

- [ ] **Step 5: Add output and round-trip tests**

Assert `final_partition.json` loads with `PlayerPartition.from_json`, covers all
segments, and `initial_partition.json` is identical for two runs with the same
seed. Use a fake runner for the unit test; the NEST integration is separate.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_evo_cli.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add persona/cli.py personalize.py tests/test_evo_cli.py
git commit -m "feat(persona): add evo training mode and player-map CLI"
```

---

### Task 10: Physiological reference and partition metrics

**Files:**
- Create: `persona/physo_reference.py`
- Test: `tests/test_physo_reference.py` (new)

**Interfaces:**
- Produces:

```python
def level1_labels(matrix) -> dict[int, str]
def level2_labels(matrix) -> dict[int, str]
def partition_metrics(partition, labels) -> dict
def global_drive_fate(partition, matrix) -> dict
```

`partition_metrics` returns `ari`, `nmi`, `homogeneity`, `completeness`,
`v_measure`, `purity`, `coverage`, `pairwise_f1`.

- [ ] **Step 1: Write the failing test**

```python
labels = level2_labels(matrix)
assert set(labels) == set(range(matrix.n_segments))
assert len(set(labels.values())) == 19
assert labels[matrix.segment_of_site["pop_BS_E_rate"]] == "global_drive"
# the two reciprocal inhibitions are in the limb's reciprocal module
assert labels[segment_id("con_in_e_L_m_f_L")] == "L_recip_inh"
```

Also test ARI/NMI on a known synthetic partition.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physo_reference.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the hierarchy**

Level 1: population + all projections whose source is that population;
relay-sourced projections go to `global_drive` (BS/BASE) or the limb's
`cutaneous`/`ib` (CUT/Ib). Level 2: group Level 1 by the functional table in
the spec. Implement ARI/NMI/V-measure/purity/coverage in numpy (no
scikit-learn). `global_drive_fate` classifies the three rate segments and the
eight BS/BASE projections as `same_player`, `split_by_limb`,
`rate_global_projections_limb`, `absorbed`, or `other`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_physo_reference.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/physo_reference.py tests/test_physo_reference.py
git commit -m "feat(persona): add physiological reference and partition metrics"
```

---

### Task 11: `bench_evo.py` and the smoke/scaled experiments

**Files:**
- Create: `bench_evo.py`
- Test: `tests/test_bench_evo.py` (new, fake runner)
- Report: `docs/reports/persona_evo.md`

**Interfaces:**
- Produces: `run_compare(out_dir, *, seeds, players, inits, stops, steps,
  generations, jobs, prefix) -> dict`; `--smoke`, `--compare`.

- [ ] **Step 1: Write the failing smoke test**

```python
def test_bench_evo_summary_writes(tmp_path):
    from bench_evo import run_compare
    result = run_compare(tmp_path, seeds=(0,), players=(4,),
                         inits=("territory",), stops=("partition",),
                         steps=10, generations=1, jobs=1,
                         prefix="smoke", runner=fake_runner)
    assert (tmp_path / "smoke_summary.json").exists()
    assert result["summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_evo.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the harness**

Mirror `bench_game.py`: one fresh process per arm, `ProcessPoolExecutor` for
jobs, CSV/JSON/MD outputs. `--smoke` = P=24, 400 steps, 1 seed, 1 generation,
territory + random, partition stop. `--compare` = P=24/137, 2 seeds, 800 steps,
6 generations, both inits, both stops. Write `benchmarks/evo/`.

- [ ] **Step 4: Run the smoke on the fast CPG**

Run (WSL CPGV): `python bench_evo.py --smoke`
Expected: `benchmarks/evo/smoke/` with valid partitions, no lost sites, no
errors.

- [ ] **Step 5: Run the scaled experiment**

Run (WSL CPGV): `python bench_evo.py --compare --jobs 2`
Expected: `benchmarks/evo/` tables; record wall time and failures.

- [ ] **Step 6: Compare to the reference and write the report**

Run the manual 137, reference Level-2, evo-territory and evo-random arms on the
fast CPG; write `docs/reports/persona_evo.md` with ARI/NMI/purity, performance
(loss, p90, at_bound, q>1, violations, entropy), phase/bio metrics, the
global-drive verdict, and the exact per-seed picks for the user to grade.
Update the backlog with the verdict.

- [ ] **Step 7: Commit**

```bash
git add bench_evo.py tests/test_bench_evo.py docs/reports/persona_evo.md docs/backlog.md
git commit -m "feat(bench): add PERSONA-Evo harness and report"
```
