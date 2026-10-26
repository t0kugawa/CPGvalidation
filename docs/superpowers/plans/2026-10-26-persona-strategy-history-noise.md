# PERSONA strategy history, noise recorders, and trace sufficiency — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Characterize and test the strategy history/noise-recorder paths, add an opt-in applied-sample trace + strategy snapshots, and prove whether parameter traces or `F/Xbar/G` are sufficient for replay and merge.

**Architecture:** Keep the production `F/Xbar/G` update and noise scoring unchanged. Add a construction-time producer filter for tests, a float64 applied-sample trace in `Agent`, a self-contained `FisherMatrix` snapshot, an exact offline rebuild, and HDF5/CLI plumbing. The fast-CPG A/B is the production-path confirmation; the synthetic probe is the primary logic evidence.

**Tech Stack:** Python 3.12, NumPy, SciPy, h5py, pytest; NEST only for the fast-CPG confirmation (WSL `CPGV`).

**Spec:** `docs/superpowers/specs/2026-10-26-persona-strategy-history-noise-design.md`

## Global Constraints

- Do not change `F`, `Xbar`, `G`, the sigma rule, advantage, credit, the recursive inverse, or the noise scoring formula.
- Do not change the default `merge` semantics; document and test it.
- Defaults: `trace_strategy=False`, `trace_float32=False`, `snapshot_strategy=0`, `noise_config="all"`.
- `run_personalization` must not start writing `run.h5`; `run_validation.py`/`run_2min.py` remain the HDF5 path.
- The workspace has uncommitted user work (`model/leg.py`, `model/cut.py`, `t2xcpg/*`, `persona/phase_metrics.py`, `personalize.py`, deleted `model/phase_clock.py`, `tests/test_phase_clock.py`, `tests/test_reciprocal_drive.py`). Make additive edits only; never `git restore`/reset these files.
- Fast tests run on Windows Python; NEST tests run with `/home/t0kgw/miniconda3/envs/CPGV/bin/python` under WSL.
- No new dependency.

## Review Focus

- A trace run with a site added mid-run must raise, not silently corrupt `F`; the owning task is Task 4.
- A snapshot taken between refreshes must preserve `P`/`_steps_since_refresh`, or the next `Δμ` differs; Task 3.
- A float32 trace must not be described or tested as exact; Task 4.
- A producer delta must be consumed exactly once, and duplicate writes must be documented as last-wins; Task 2.
- Loading a `run.h5` without trace/checkpoint groups must not crash; Task 6.

---

### Task 1: Producer selection filter

**Files:**
- Modify: `persona/noise_recorder.py`
- Modify: `persona/agent.py:17-112` (constructor + strategy construction)
- Modify: `personalize.py:61-217` (`build_personalizer` signature/pass-through)
- Test: `tests/test_noise_configurations.py`

**Interfaces:**
- Consumes: `registry.producer_recorders() -> dict[str, NoiseRecorder]`.
- Produces: `select_producer_names(names: Iterable[str], config: str) -> list[str]` and `Agent(..., noise_config: str = "all")`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from persona.noise_recorder import select_producer_names

NAMES = [
    "producer:poisson:cut_L", "producer:poisson:ia_e_L",
    "producer:stdp:cut_rge_L", "producer:stdp:ia_rge_L",
]

def test_select_producer_names_all_none_poisson_stdp():
    assert select_producer_names(NAMES, "all") == NAMES
    assert select_producer_names(NAMES, "none") == []
    assert select_producer_names(NAMES, "poisson") == NAMES[:2]
    assert select_producer_names(NAMES, "stdp") == NAMES[2:]

def test_select_producer_names_one_relay_accepts_prefix_or_plain():
    assert select_producer_names(NAMES, "one-relay:cut_L") == [NAMES[0]]
    assert select_producer_names(
        NAMES, "one-relay:producer:poisson:cut_L") == [NAMES[0]]

def test_select_producer_names_rejects_unknown():
    with pytest.raises(ValueError):
        select_producer_names(NAMES, "banana")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_noise_configurations.py -q`
Expected: FAIL with `ImportError: cannot import name 'select_producer_names'`.

- [ ] **Step 3: Implement `select_producer_names` in `persona/noise_recorder.py`**

Accept `all`, `none`, `poisson`, `stdp`, `one-relay:<name>` (strip an optional
`producer:` prefix). Preserve the input order. Raise `ValueError` for anything
else or for an unknown relay name.

- [ ] **Step 4: Wire `noise_config` into `Agent`**

In `Agent.__init__`, filter only the strategy's producer names:

```python
names = select_producer_names(sorted(self.registry.producer_recorders()),
                              noise_config)
```

Use `names` for the initial `FisherMatrix(noise_names=...)`. In
`Agent.refresh_params`, keep the same filter when new producers appear; do not
read filtered-out producers in `Agent.step`. `build_personalizer` accepts
`noise_config: str = "all"` and passes it to `Agent`.

- [ ] **Step 5: Run the filter tests**

Run: `python -m pytest tests/test_noise_configurations.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add persona/noise_recorder.py persona/agent.py personalize.py tests/test_noise_configurations.py
git commit -m "feat(persona): add construction-time producer selection"
```

---

### Task 2: Recorder semantics characterization

**Files:**
- Test: `tests/test_noise_configurations.py` (append)
- Test: `tests/test_noise_recorder.py` (extend only if a shared helper is useful)

**Interfaces:**
- Consumes: `NoiseRecorder.record_delta`, `take_delta`, `record_sample`,
  `latest_sample`, `stats`.
- Produces: tests only; no production behavior change.

- [ ] **Step 1: Write the characterization tests**

```python
import numpy as np
from persona.noise_recorder import NoiseRecorder

def test_duplicate_delta_is_last_wins_and_read_clears():
    rec = NoiseRecorder("producer:x", scale=2.0)
    rec.record_delta(0.0, 1.0)
    rec.record_delta(0.0, 3.0)
    assert rec.take_delta() == 3.0
    assert rec.take_delta() is None

def test_missing_scale_records_stats_but_strategy_score_is_zero():
    rec = NoiseRecorder("producer:x")
    rec.record_delta(0.0, 2.0)
    assert rec.scale is None
    assert rec.take_delta() == 2.0
    assert rec.stats()["n"] == 1

def test_site_recorder_keeps_last_sample_and_is_not_a_producer():
    rec = NoiseRecorder("site:pop_A_a")
    rec.record_sample(0.0, 1.0, 0.2)
    rec.record_sample(10.0, 1.5, 0.1)
    assert rec.latest_sample() == (1.5, 0.1)
    assert rec.take_delta() is None
```

- [ ] **Step 2: Run tests to verify they pass on current behavior**

Run: `python -m pytest tests/test_noise_configurations.py -q`
Expected: PASS (characterization). If a test fails, fix the test to match the
documented current behavior and record the discrepancy in the report; do not
change the recorder in this task.

- [ ] **Step 3: Add the Agent-level parity test**

Using the `FakeRegistry` pattern from `tests/test_game_agent.py`, create a
registry with three producer recorders and assert:

- `noise_config="none"` gives `strategy.noise_names == []` and `F.shape == (2n, 2n)`;
- `noise_config="poisson"` gives only the Poisson names;
- a delta consumed at step 1 is not seen at step 2;
- the noise rows of `G` are exactly zero.

- [ ] **Step 4: Run the Agent-level tests**

Run: `python -m pytest tests/test_noise_configurations.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_noise_configurations.py
git commit -m "test(persona): characterize producer and site recorder semantics"
```

---

### Task 3: `FisherMatrix` snapshot

**Files:**
- Modify: `persona/strategy.py:75-179` (add snapshot/from_snapshot)
- Test: `tests/test_strategy_snapshot.py` (new)

**Interfaces:**
- Produces:

```python
def snapshot(self) -> dict:
    """Self-contained strategy checkpoint (F, Xbar, G, P, steps, ...)."""

@classmethod
def from_snapshot(cls, data: dict) -> "FisherMatrix":
    ...
```

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from persona.strategy import FisherMatrix

def _fm(**kw):
    opts = dict(param_names=["a", "b"], noise_names=["n"],
                beta=0.9, lambda_reg=0.5, warmup=0, refresh_every=10)
    opts.update(kw)
    return FisherMatrix(**opts)

def test_snapshot_restores_mid_refresh_exactly():
    fm = _fm()
    for t in range(3):
        fm.step({"a": (0.1, 0.5, 0.0), "b": (-0.2, 0.4, 0.01)},
                {"n": (0.05, 1.0)}, np.array([0.3, -0.1]))
    snap = fm.snapshot()
    clone = FisherMatrix.from_snapshot(snap)
    assert np.array_equal(clone.F, fm.F)
    assert np.array_equal(clone.Xbar, fm.Xbar)
    assert np.array_equal(clone.G, fm.G)
    assert np.array_equal(clone.P, fm.P)
    assert clone._steps_since_refresh == fm._steps_since_refresh
    # next proposal is identical
    sites = {"a": (0.05, 0.5, 0.0), "b": (0.03, 0.4, 0.0)}
    noise = {"n": (0.01, 1.0)}
    A = np.array([0.2, 0.1])
    d1 = fm.step(sites, noise, A)
    d2 = clone.step(sites, noise, A)
    assert d1.keys() == d2.keys()
    for k in d1:
        assert np.isclose(d1[k], d2[k])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strategy_snapshot.py -q`
Expected: FAIL with `AttributeError: 'FisherMatrix' object has no attribute 'snapshot'`.

- [ ] **Step 3: Implement snapshot/from_snapshot**

Include `F`, `Xbar`, `G`, `P`, `steps`, `_steps_since_refresh`, `beta`,
`lambda_reg`, `warmup`, `refresh_every`, `lr_mu`, `dt_s`, `param_names`,
`noise_names`, `noise_scales`, `homeostasis`/`game` = `None` (not pickled).
`from_snapshot` must not rebuild `P`; it restores the exact inverse and
`_P_valid=True`. Document that a snapshot is a strategy checkpoint, not a full
Agent resume.

- [ ] **Step 4: Run the snapshot tests**

Run: `python -m pytest tests/test_strategy_snapshot.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/strategy.py tests/test_strategy_snapshot.py
git commit -m "feat(persona): add exact FisherMatrix snapshots"
```

---

### Task 4: Applied-sample trace and exact rebuild

**Files:**
- Modify: `persona/agent.py:27-170, 207-358`
- Create: `persona/strategy_trace.py`
- Test: `tests/test_strategy_trace.py` (new)

**Interfaces:**
- Produces:

```python
Agent(trace_strategy: bool = False, trace_float32: bool = False)
# history["strategy_trace"] = {
#   "dtheta": (T,S), "sigma": (T,S), "dlogsigma": (T,S),
#   "advantage": (T,2), "noise_delta": (T,N), "noise_scale": (T,N),
#   "param_names": [...], "noise_names": [...],
#   "beta": float, "lambda_reg": float, "warmup": int,
#   "refresh_every": int, "dtype": "float64"|"float32",
# }

def rebuild_strategy_from_trace(trace: dict,
                                FisherMatrix_cls=FisherMatrix) -> FisherMatrix
```

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from persona.strategy import FisherMatrix
from persona.strategy_trace import rebuild_strategy_from_trace

def _trace(T=20, S=3, N=2):
    rng = np.random.default_rng(0)
    return {
        "dtheta": rng.normal(0, 0.1, (T, S)),
        "sigma": rng.uniform(0.2, 0.8, (T, S)),
        "dlogsigma": rng.normal(0, 0.01, (T, S)),
        "advantage": rng.normal(0, 0.3, (T, 2)),
        "noise_delta": rng.normal(0, 0.05, (T, N)),
        "noise_scale": np.ones((T, N)),
        "param_names": [f"p{i}" for i in range(S)],
        "noise_names": [f"n{i}" for i in range(N)],
        "beta": 0.9, "lambda_reg": 0.5, "warmup": 0,
        "refresh_every": 10, "dtype": "float64",
    }

def test_rebuild_from_trace_is_exact():
    trace = _trace()
    fm = FisherMatrix(param_names=trace["param_names"],
                      noise_names=trace["noise_names"], beta=0.9,
                      lambda_reg=0.5, warmup=0, refresh_every=10)
    for t in range(len(trace["dtheta"])):
        sites = {n: (trace["dtheta"][t, i], trace["sigma"][t, i],
                     trace["dlogsigma"][t, i])
                 for i, n in enumerate(trace["param_names"])}
        noise = {n: (trace["noise_delta"][t, j],
                     trace["noise_scale"][t, j])
                 for j, n in enumerate(trace["noise_names"])}
        fm.step(sites, noise, trace["advantage"][t], accumulate_only=True)
    fm._refresh_inverse()
    rb = rebuild_strategy_from_trace(trace)
    assert np.array_equal(rb.F, fm.F)
    assert np.array_equal(rb.Xbar, fm.Xbar)
    assert np.array_equal(rb.G, fm.G)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strategy_trace.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'persona.strategy_trace'`.

- [ ] **Step 3: Implement the rebuild**

`rebuild_strategy_from_trace` creates a `FisherMatrix` from the trace metadata,
replays `_score_vectors` + `_accumulate` in order, calls `_refresh_inverse`,
and returns it. Do not use `step` (it would apply the policy); use
`accumulate_only=True` or `_accumulate` directly.

- [ ] **Step 4: Add the Agent trace**

In `Agent.__init__`, store `trace_strategy`, `trace_float32`, and
`_trace_site_names = None`. In `refresh_params`, when tracing:

- first call: freeze `_trace_site_names = tuple(params)`;
- later call: if `tuple(params) != _trace_site_names`, raise `RuntimeError`
  naming the new site.

In `step`, after `sites` and `noise` are built, append the record. Cast to
`float32` only when `trace_float32=True` and set `dtype` accordingly.

- [ ] **Step 5: Run the trace tests**

Run: `python -m pytest tests/test_strategy_trace.py -q`
Expected: PASS. Add a test that a new site with tracing on raises
`RuntimeError`, and that `--trace-float32` replay is approximate only.

- [ ] **Step 6: Commit**

```bash
git add persona/agent.py persona/strategy_trace.py tests/test_strategy_trace.py
git commit -m "feat(persona): add opt-in applied-sample trace and exact rebuild"
```

---

### Task 5: Checkpoint cadence in `Agent`

**Files:**
- Modify: `persona/agent.py:27-112, 207-358`
- Test: `tests/test_strategy_snapshot.py` (append)

**Interfaces:**
- Produces: `Agent(snapshot_strategy: int = 0)`; when `>0`, append
  `strategy.snapshot()` to `history["strategy_checkpoints"]` every N steps.

- [ ] **Step 1: Write the failing test**

```python
def test_agent_snapshots_every_n_steps(fake_registry):
    agent = Agent(registry=fake_registry, snapshot_strategy=2)
    _run(agent, steps=5)
    steps = [s["steps"] for s in agent.history["strategy_checkpoints"]]
    assert steps == [2, 4]
```

Use the existing `FakeRegistry`/`_run` pattern from `tests/test_game_agent.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strategy_snapshot.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'snapshot_strategy'`.

- [ ] **Step 3: Implement the cadence**

In `Agent.step`, after `strategy.step`, if
`snapshot_strategy > 0 and current_step % snapshot_strategy == 0`, append
`strategy.snapshot()`. Do not force a refresh. Default 0 = no snapshots.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_strategy_snapshot.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/agent.py tests/test_strategy_snapshot.py
git commit -m "feat(persona): add optional strategy checkpoint cadence"
```

---

### Task 6: HDF5 trace and checkpoints

**Files:**
- Modify: `persona/h5io.py:21-116`
- Test: `tests/test_h5io_trace.py` (new; h5py is installed on Windows)

**Interfaces:**
- Produces: `save_run` writes `agent/strategy_trace/*` when present and
  `strategy_checkpoints/<step>/*` when present; `load_run` reads them; missing
  groups are simply absent.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from persona.h5io import save_run, load_run
from persona.personalization import RunLog

class _Agent:
    history = {
        "means": {"p": [0.0, 0.1]},
        "sigmas": {"p": [0.2, 0.2]},
        "strategy_trace": {
            "dtheta": np.zeros((2, 1)), "sigma": np.ones((2, 1)),
            "dlogsigma": np.zeros((2, 1)), "advantage": np.zeros((2, 2)),
            "noise_delta": np.zeros((2, 0)), "noise_scale": np.zeros((2, 0)),
            "param_names": ["p"], "noise_names": [],
            "beta": 0.995, "lambda_reg": 1e-3, "warmup": 3,
            "refresh_every": 10, "dtype": "float64",
        },
        "strategy_checkpoints": [{"steps": 2, "F": np.eye(2)}],
    }

def test_save_load_trace_and_checkpoints(tmp_path):
    path = tmp_path / "run.h5"
    save_run(str(path), RunLog(), _Agent())
    data, _ = load_run(str(path))
    assert data["agent"]["strategy_trace"]["beta"] == 0.995
    assert data["agent"]["strategy_checkpoints"][0]["steps"] == 2
    assert np.array_equal(data["agent"]["strategy_checkpoints"][0]["F"],
                          np.eye(2))

def test_missing_trace_groups_do_not_crash(tmp_path):
    class _Plain:
        history = {"means": {}, "sigmas": {}}
    path = tmp_path / "plain.h5"
    save_run(str(path), RunLog(), _Plain())
    data, _ = load_run(str(path))
    assert "strategy_trace" not in data["agent"]
    assert "strategy_checkpoints" not in data["agent"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_h5io_trace.py -q`
Expected: FAIL with `KeyError`/missing dataset.

- [ ] **Step 3: Implement the writer/reader**

Write trace arrays as float64 (or the stored dtype), the metadata as attrs, and
each checkpoint as a group. `load_run` must not fail when the groups are absent.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_h5io_trace.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/h5io.py tests/test_h5io_trace.py
git commit -m "feat(persona): persist strategy traces and checkpoints"
```

---

### Task 7: CLI and personalizer plumbing

**Files:**
- Modify: `persona/cli.py:7-44`
- Modify: `personalize.py:61-217` (signature/pass-through only)
- Modify: `run_validation.py:103-217` (pass-through)
- Test: `tests/test_cli_trace.py` (new)

**Interfaces:**
- Produces: `--trace-strategy`, `--trace-float32`, `--snapshot-strategy N`
  on `persona/cli.py`; `build_personalizer(..., trace_strategy=False,
  trace_float32=False, snapshot_strategy=0, noise_config="all")`.

- [ ] **Step 1: Write the failing test**

```python
from persona.cli import build_parser

def test_trace_flags_parse():
    args = build_parser().parse_args(
        ["--trace-strategy", "--trace-float32", "--snapshot-strategy", "500"])
    assert args.trace_strategy is True
    assert args.trace_float32 is True
    assert args.snapshot_strategy == 500

def test_trace_flags_default_off():
    args = build_parser().parse_args([])
    assert args.trace_strategy is False
    assert args.trace_float32 is False
    assert args.snapshot_strategy == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_trace.py -q`
Expected: FAIL with `SystemExit`/unrecognized arguments.

- [ ] **Step 3: Implement the plumbing**

Add the flags to `build_parser`; pass them through `build_personalizer` to
`Agent`; `run_validation.py` and `run_2min.py` pass their parsed values.
`personalize.py` writes `strategy_trace.npz` only when `trace_strategy` is on
and `output_dir` is set. Do not add `save_run` to `run_personalization`.

- [ ] **Step 4: Run the CLI tests and the existing suite**

Run: `python -m pytest tests/test_cli_trace.py tests/test_cli_lr.py tests/test_cli_threads.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/cli.py personalize.py run_validation.py run_2min.py tests/test_cli_trace.py
git commit -m "feat(persona): expose trace and snapshot flags"
```

---

### Task 8: Strategy-history benchmark

**Files:**
- Create: `bench_strategy_history.py`
- Test: `tests/test_bench_strategy_history.py` (new, smoke only)

**Interfaces:**
- Produces: `run_synthetic(...) -> dict`, `run_fast_cpg(...) -> dict`,
  `--quick`, `--noise-config`, `--fast-cpg`, `--steps`, `--seeds`.

- [ ] **Step 1: Write the failing smoke test**

```python
def test_synthetic_quick_writes_files(tmp_path):
    from bench_strategy_history import run_synthetic
    result = run_synthetic(out_dir=tmp_path, steps=20, seed=0)
    assert (tmp_path / "synthetic_summary.json").exists()
    assert result["trace_rebuild_error"] < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_strategy_history.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the synthetic mode**

Generate the same synthetic stream used in the spec probe, run a full
`FisherMatrix`, rebuild from the A-trace, and report `F/Xbar/G` errors,
P-trace errors, merge errors, and noise-config durations. Write CSV/JSON/MD.

- [ ] **Step 4: Add fast-CPG mode**

Use `build_personalizer` with `noise_config`, `trace_strategy=True`,
`FAST_SIZES`, `game=None`, `homeostasis=None`, 400 steps. Report loss,
noise-channel `F` norms, `G` noise rows, recorder counts, and trace error.
Mark the test slow; the plan's verification may run it under WSL CPGV.

- [ ] **Step 5: Run the smoke and the fast-CPG confirmation**

Run: `python -m pytest tests/test_bench_strategy_history.py -q`
Run (WSL CPGV): `python bench_strategy_history.py --fast-cpg --steps 400 --seeds 0 1`
Expected: files in `benchmarks/strategy_history/`; no NaN; `G` noise rows zero.

- [ ] **Step 6: Commit**

```bash
git add bench_strategy_history.py tests/test_bench_strategy_history.py
git commit -m "feat(bench): add strategy-history and noise-config benchmark"
```

---

### Task 9: Report and backlog

**Files:**
- Create: `docs/reports/strategy_history_noise.md`
- Modify: `docs/backlog.md` (already has the 2026-10-26 findings; add measured numbers)

- [ ] **Step 1: Write the report**

Include the code map, the external/internal recorder definitions, the probe
table, the merge explanation, the noise-config results, the fast-CPG
confirmation, and the recommendation. State the parameter-trace update
(trace float64, fixed site set, snapshots archival, `merge` ensemble-only).

- [ ] **Step 2: Update the backlog**

Add the measured numbers and any failed configuration; keep the existing
2026-10-26 entry.

- [ ] **Step 3: Run the full fast suite**

Run: `python -m pytest -q`
Expected: PASS (NEST tests may skip on Windows; run WSL for the NEST subset).

- [ ] **Step 4: Commit**

```bash
git add docs/reports/strategy_history_noise.md docs/backlog.md
git commit -m "docs: record strategy-history and noise findings"
```
