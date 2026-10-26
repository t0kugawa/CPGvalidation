# PERSONA algorithm comparison implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair toy-CPG comparison of updated PERSONA, diagonal/block natural gradient, Optuna CMA-ES, and a numpy surrogate-gradient BPTT, then run the same PERSONA/NG/CMA-ES family on the NEST fast CPG with bio-plausibility metrics.

**Architecture:** A benchmark-only 14-parameter spiking toy is the common testbed; every arm gets 20,000 environment steps and a common held-out tail-window evaluation. The CPG comparison reuses the existing fast path and metrics, with H off and the Evo arm supplied as a saved partition. No production default changes.

**Tech Stack:** Python 3.12, NumPy, SciPy, Optuna + `cmaes`, pytest; NEST/MuJoCo for the CPG arms (WSL `CPGV`).

**Spec:** `docs/superpowers/specs/2026-10-26-persona-algorithm-comparison-design.md`

## Global Constraints

- Benchmark-only; do not change the PERSONA model, the CPG, MuJoCo, or production defaults.
- The toy surrogate gradient is pure numpy; no torch/jax/scikit-learn.
- All arms get 20,000 toy environment steps (200 episodes × 100 steps) and are scored on a common held-out tail-window episode.
- H is off in every arm. `persona_diag`/`persona_block` are game-off. `persona_evo` requires `--player-map`; the full Evo loop is not run here.
- CMA-ES on the CPG uses a reduced top-16 subspace from a seed-0 pilot; the pilot cost is reported separately. `cmaes_full` gets 16 trials.
- Sole-up is reported as "not measured (not logged)"; do not add production logging.
- Do not revert uncommitted user work in `model/`, `t2xcpg/`, `persona/phase_metrics.py`, or `personalize.py`.
- Fast tests on Windows Python; NEST tests with `/home/t0kgw/miniconda3/envs/CPGV/bin/python` under WSL.

## Review Focus

- The toy's `L_vec` must be `[L_E, L_F]`; a scalar duplicated/padded is a different update and must be caught by the toy test.
- The surrogate gradient must pass a finite-difference check; otherwise the arm is not a surrogate-gradient result.
- Every arm must use the same initial vector per seed and the same held-out evaluation; Task 5.
- `persona_diag`/`persona_block` must be game-off and H-off; Task 3.
- A CPG run with `sole-up` absent must say "not measured", not pass by omission; Task 6.

---

### Task 1: Toy spiking CPG model

**Files:**
- Create: `benchmarks/compare/__init__.py`
- Create: `benchmarks/compare/toy_cpg.py`
- Test: `tests/test_toy_cpg.py` (new)

**Interfaces:**
- Produces:

```python
PARAM_NAMES: tuple[str, ...]  # the 14 names, fixed order
BOUNDS: dict[str, tuple[float, float]]
INITIAL: dict[str, float]

@dataclass(frozen=True)
class ToyConfig:
    dt: float = 0.01
    steps: int = 100
    n_rg: int = 8
    noise_std: float = 0.0
    seed: int = 0

class ToyCPG:
    def __init__(self, config: ToyConfig)
    def reset(self, seed: int | None = None) -> None
    def step(self, params: dict[str, float]) -> dict
    def rollout(self, params: dict[str, float], *,
                record: bool = False) -> dict
    def held_out_loss(self, params: dict[str, float]) -> float
```

`rollout` returns `loss_vec` (`np.array([L_E, L_F])`), `loss_total`,
`tail_loss`, `q_model`, `q_target`, `rate_e`, `rate_f`, `tau`, and `phase`.
`held_out_loss` uses a fixed evaluation episode (seed 10_000) and the tail 50%.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from benchmarks.compare.toy_cpg import PARAM_NAMES, INITIAL, BOUNDS, ToyCPG, ToyConfig

def test_toy_forward_is_deterministic_and_bounded():
    toy = ToyCPG(ToyConfig(seed=0, noise_std=0.0))
    out1 = toy.rollout(INITIAL)
    out2 = toy.rollout(INITIAL)
    assert np.array_equal(out1["q_model"], out2["q_model"])
    assert out1["loss_vec"].shape == (2,)
    assert np.isfinite(out1["loss_total"])
    assert np.isfinite(out1["tail_loss"])
    for name in PARAM_NAMES:
        lo, hi = BOUNDS[name]
        assert lo <= INITIAL[name] <= hi

def test_toy_tail_and_held_out_are_deterministic():
    toy = ToyCPG(ToyConfig(seed=0))
    a = toy.rollout(INITIAL)["tail_loss"]
    b = toy.rollout(INITIAL)["tail_loss"]
    assert a == b
    assert toy.held_out_loss(INITIAL) == toy.held_out_loss(INITIAL)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_toy_cpg.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks'`.

- [ ] **Step 3: Implement the toy**

Two LIF RG populations (`N_rg=8`), reciprocal inhibition, tonic drive, two MN
rate readouts, one agonist/antagonist muscle pair, and a 1-DOF plant. Use the
14 parameters in the spec. The phase of each step is `"E"` or `"F"` by the
larger RG rate. `L_E`/`L_F` are the mean squared tracking errors over the
E-/F-dominated steps; if a phase has no steps, use the overall mean. The
tail window is the last 50% of the episode. Bounds and initial values are
fixed constants. No external dependencies beyond numpy.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_toy_cpg.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/compare/__init__.py benchmarks/compare/toy_cpg.py tests/test_toy_cpg.py
git commit -m "feat(bench): add differentiable toy spiking CPG"
```

---

### Task 2: Surrogate-gradient BPTT

**Files:**
- Create: `benchmarks/compare/surrogate.py`
- Test: `tests/test_surrogate_gradient.py` (new)

**Interfaces:**
- Produces:

```python
def surrogate_loss_and_grad(params: dict[str, float],
                            config: ToyConfig,
                            alpha: float = 10.0) -> tuple[float, np.ndarray]
def finite_difference_grad(params, config, eps=1e-5, idx=(0, 1, 2)) -> np.ndarray
```

- [ ] **Step 1: Write the failing gradient-check test**

```python
import numpy as np
from benchmarks.compare.toy_cpg import INITIAL, ToyConfig
from benchmarks.compare.surrogate import surrogate_loss_and_grad, finite_difference_grad

def test_surrogate_matches_finite_difference():
    cfg = ToyConfig(steps=30, seed=0)
    loss, grad = surrogate_loss_and_grad(INITIAL, cfg)
    fd = finite_difference_grad(INITIAL, cfg, eps=1e-5, idx=(0, 1, 2))
    rel = np.linalg.norm(grad[:3] - fd) / max(np.linalg.norm(fd), 1e-12)
    assert rel < 1e-4
    assert np.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_surrogate_gradient.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement forward/backward**

Forward the same LIF/plant equations as the toy. Backward with the fast-sigmoid
surrogate `d spike/d v = alpha / (1 + alpha*|v|)^2`. Backprop through the
membrane, reciprocal weights, tonic drive, MN readout, muscle, and plant. Return
the tail loss and the gradient in `PARAM_NAMES` order. If the full BPTT is too
large for a first pass, implement it for `steps=30` and extend to 100 after the
check passes; the test uses 30.

- [ ] **Step 4: Run the gradient check**

Run: `python -m pytest tests/test_surrogate_gradient.py -q`
Expected: PASS. If the relative error is above `1e-4`, fix the backward pass;
do not loosen the threshold.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/compare/surrogate.py tests/test_surrogate_gradient.py
git commit -m "feat(bench): add numpy surrogate-gradient BPTT"
```

---

### Task 3: Variant Fisher preconditioners

**Files:**
- Create: `benchmarks/compare/variants.py`
- Test: `tests/test_compare_optimizers.py` (new; part 1)

**Interfaces:**
- Produces:

```python
class VariantFisherMatrix(FisherMatrix):
    def __init__(self, *args, preconditioner="full", blocks=None, **kwargs)
```

`preconditioner` is `full`, `diag`, or `block`; `blocks` is a list of index
arrays for `block`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from benchmarks.compare.variants import VariantFisherMatrix

def _fm(mode, blocks=None):
    return VariantFisherMatrix(
        param_names=["a", "b", "c"], noise_names=[],
        beta=0.9, lambda_reg=0.5, warmup=0, refresh_every=1,
        lr_mu=0.1, preconditioner=mode, blocks=blocks)

def test_block_with_one_block_equals_full():
    full = _fm("full")
    block = _fm("block", blocks=[np.array([0, 1, 2])])
    sites = {"a": (0.1, 0.5, 0.0), "b": (-0.2, 0.4, 0.0),
             "c": (0.05, 0.3, 0.0)}
    A = np.array([0.3, -0.1])
    d1 = full.step(sites, {}, A)
    d2 = block.step(sites, {}, A)
    for k in d1:
        assert np.isclose(d1[k], d2[k])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_optimizers.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Subclass `FisherMatrix` and construct it with `refresh_every=1`, so `_deltas`
calls `_solve_direct(g)`. Override `_solve_direct`:

- `full`: the parent solve;
- `diag`: `g / (diag(F) + λ)`;
- `block`: solve each block `(F_block + λ I)` and scatter the result.

This keeps `F/Xbar/G` unchanged and makes the variant preconditioner actually
used; `P` is not consulted on this path. Assert `diag` on a diagonal `F` equals
`full`, and the block with one block equals `full`.

- [ ] **Step 4: Add the small optimizer smoke**

For each of `full`, `diag`, `block`, run 5 steps on synthetic sites and assert
finite deltas and finite `F`. Assert `persona_diag`/`persona_block` are
game-off (they are constructed without `game` in the harness).

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_compare_optimizers.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/compare/variants.py tests/test_compare_optimizers.py
git commit -m "feat(bench): add diagonal and block natural-gradient variants"
```

---

### Task 4: Optuna CMA-ES arm

**Files:**
- Create: `benchmarks/compare/cmaes_arm.py`
- Modify: `requirements.txt` (add `cmaes` under a benchmark comment)
- Test: `tests/test_compare_optimizers.py` (append)

**Interfaces:**
- Produces:

```python
def run_cmaes(objective, bounds: dict[str, tuple[float, float]],
              x0: dict[str, float], n_trials: int, seed: int) -> dict
```

Returns `best_params`, `best_value`, `values`, `n_trials`, `wall_s`.

- [ ] **Step 1: Write the failing test**

```python
def test_cmaes_finds_quadratic_minimum():
    from benchmarks.compare.cmaes_arm import run_cmaes
    bounds = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}
    result = run_cmaes(lambda p: (p["x"] - 0.3) ** 2 + (p["y"] + 0.2) ** 2,
                       bounds, {"x": 0.0, "y": 0.0}, n_trials=20, seed=0)
    assert result["best_value"] < 0.1
    assert len(result["values"]) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_optimizers.py -q`
Expected: FAIL with `ImportError` or missing `cmaes`.

- [ ] **Step 3: Install `cmaes` and implement**

Run: `python -m pip install cmaes`
Add `cmaes` to `requirements.txt` with a "benchmark only" comment. Implement
with `optuna.create_study(direction="minimize",
sampler=CmaEsSampler(seed=seed, x0=x0, sigma0=0.1, n_startup_trials=0))`.
Use `trial.suggest_float(name, lo, hi)` for each parameter. Return the best
trial and the full value list.

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/test_compare_optimizers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/compare/cmaes_arm.py requirements.txt tests/test_compare_optimizers.py
git commit -m "feat(bench): add Optuna CMA-ES arm"
```

---

### Task 5: Toy harness and 5-seed run

**Files:**
- Create: `bench_persona_compare.py`
- Create: `benchmarks/compare/objectives.py` (shared tail/held-out helpers)
- Test: `tests/test_compare_harness.py` (new)
- Test: `tests/test_compare_metrics.py` (new; bio table only)

**Interfaces:**
- Produces: `run_toy(seeds, arms, episodes=200, out_dir) -> dict`; CLI
  `--smoke`, `--toy`, `--seeds`, `--arms`.

- [ ] **Step 1: Write the failing smoke test**

```python
def test_toy_smoke_writes_files(tmp_path):
    from bench_persona_compare import run_toy
    result = run_toy(seeds=(0,), arms=("persona_game", "persona_diag",
                                       "cmaes", "surrogate"),
                     episodes=2, out_dir=tmp_path)
    assert (tmp_path / "toy" / "runs.csv").exists()
    assert result["summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_harness.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the online loop**

For each seed: generate and save the initial vector. For each episode:
`toy.reset(seed=seed + episode)`; for each step: sample each
`PersonaParam`, build `sites`, step the toy with the sampled params, compute
`loss_vec`, call `strategy.step` with phase `("E", "F")`, apply the returned
`Δμ` and the sigma rule. Keep one strategy across episodes. For `persona_game`,
build four synthetic players with `PhysicalMap.from_specs` and a `StackGame`
(`GameConfig.current_best()`); `persona_nogame`/`persona_diag`/`persona_block`
have `game=None`. `persona_diag`/`persona_block` use `VariantFisherMatrix`.
`surrogate` uses the Adam BPTT; `cmaes` uses `run_cmaes` with the tail-loss
objective. Score every final vector with `toy.held_out_loss`.

- [ ] **Step 4: Write the outputs**

`toy/runs.csv` (seed, arm, episode, tail_loss, held_out_loss, wall_s),
`toy/summary.csv` (mean/std per arm), `toy/summary.json`, `toy/curves.png`
(matplotlib, Agg). The summary includes AUC and success rate at a reported
threshold.

- [ ] **Step 5: Run the smoke**

Run: `python -m pytest tests/test_compare_harness.py -q`
Then: `python bench_persona_compare.py --smoke`
Expected: files written, all losses finite, budgets printed.

- [ ] **Step 6: Run the 5-seed toy comparison**

Run: `python bench_persona_compare.py --toy --seeds 0 1 2 3 4`
Expected: `benchmarks/compare/toy/` tables; record wall time and the winner per
metric.

- [ ] **Step 7: Commit**

```bash
git add bench_persona_compare.py benchmarks/compare/objectives.py tests/test_compare_harness.py tests/test_compare_metrics.py
git commit -m "feat(bench): add toy comparison harness"
```

---

### Task 6: CPG arms and bio/phase tables

**Files:**
- Modify: `bench_persona_compare.py` (add `run_cpg`)
- Test: `tests/test_compare_cpg_smoke.py` (new, slow)
- Test: `tests/test_compare_metrics.py` (append bio table)

**Interfaces:**
- Produces: `run_cpg(seeds, arms, steps, out_dir, player_map=None,
  cmaes_trials=16, cmaes_sub_k=16) -> dict`.

- [ ] **Step 1: Write the failing bio-table test**

```python
def test_bio_table_marks_sole_up_not_measured():
    from bench_persona_compare import bio_table
    table = bio_table({"loss_eval": 0.5, "violations": 0,
                       "phase": {"passed": True}, "acceptance": {}})
    assert "not measured" in table["sole_up"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare_metrics.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the CPG arm**

Reuse `personalize.build_personalizer` with `FAST_SIZES`, `game` per arm, H off,
`player_map` for `persona_evo` (required), and `steps`. After the run compute:

- `bench_homeostasis.inflation_metrics` and `type_sign_violations`;
- `phase_metrics.phase_gate_metrics`;
- `personalize.acceptance_from_log`;
- `loader.stats.LocomotionMetrics.from_run`;
- co-contraction fraction (both applied drives > 0.99).

`persona_diag`/`persona_block` use the variant strategy; `cmaes_sub` uses the
top-16 sites from a seed-0 pilot `persona_nogame` run, then runs the CMA-ES
trials by rebuilding the personalizer per trial and setting the selected
means; `cmaes_full` is optional and gets 16 trials. Report the pilot cost
separately.

- [ ] **Step 4: Write `cpg/bio.md` and `cpg/phase.md`**

Per arm/seed: loss_eval, r_p90, at_bound, q>1, violations, active players,
entropy, phase gate pass/fail, acceptance, locomotion metrics, co-contraction,
and `sole_up: not measured (not logged)`.

- [ ] **Step 5: Run the CPG smoke**

Run (WSL CPGV): `python -m pytest tests/test_compare_cpg_smoke.py -q`
Expected: one arm, 20 steps, finite metrics, files written.

- [ ] **Step 6: Run the CPG comparison**

Run (WSL CPGV): `python bench_persona_compare.py --cpg --seeds 0 1 --steps 800 \
  --arms persona_manual persona_nogame persona_diag persona_block cmaes_sub`
Then: `python bench_persona_compare.py --cpg \
  --player-map benchmarks/evo/final_partition.json --arms persona_evo`
Expected: `benchmarks/compare/cpg/` tables; no type/sign violations; the
reduced-subspace and pilot costs are stated.

- [ ] **Step 7: Commit**

```bash
git add bench_persona_compare.py tests/test_compare_cpg_smoke.py tests/test_compare_metrics.py
git commit -m "feat(bench): add CPG comparison and bio tables"
```

---

### Task 7: Final report

**Files:**
- Create: `docs/reports/persona_algorithm_comparison.md`
- Modify: `docs/backlog.md` (record the verdict)

- [ ] **Step 1: Write the report**

State the exact budget of every arm, the toy learning curves, the held-out
results, wall time, the CPG metrics, the bio table (including failures and
sole-up as not measured), the CMA-ES subspace/pilot cost, and the caveat that
the toy is not the NEST CPG. Name the winner per metric; do not declare a
universal winner.

- [ ] **Step 2: Run the full fast suite**

Run: `python -m pytest -q`
Expected: PASS (NEST tests may skip on Windows; run the WSL subset).

- [ ] **Step 3: Commit**

```bash
git add docs/reports/persona_algorithm_comparison.md docs/backlog.md
git commit -m "docs: record algorithm comparison results"
```
