# PERSONA recursive Fisher inverse — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the per-step O(n^3) `(F + lambda*I)` solve with an O(n^2) recursive inverse, refreshed by Cholesky at a phase shift or every `K_max` steps, whichever comes first.

**Architecture:** `FisherMatrix` keeps `P ~= (F + lambda*I)^-1`, updates it with Sherman-Morrison, and refreshes it exactly at phase shifts or `K_max`. `Agent` detects a change in the normalized phase key and passes `phase_shift` to the strategy. `F`, `Xbar`, `G`, `g`, noise scoring, and sigma updates are unchanged.

**Tech Stack:** Python 3.12, numpy, scipy, pytest, NEST 3.10.

**Spec:** `docs/superpowers/specs/2026-10-20-persona-recursive-fisher-inverse-design.md`

## Global Constraints

- Runtime: WSL venv `/home/t0kgw/.virtualenvs/KinematicsCPG`; repo `/mnt/d/CPGvalidation`.
  All test commands run as
  `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest ... -v"`.
- Do NOT modify `tinycpg/`, `memcpg/`, or root `persona.py`.
- Keep existing `TODO(advantage)`, `TODO(fisher)`, `TODO(sigma-ewma)`, and `TODO(perf)` markers; only update the `TODO(perf)` text.
- Preserve the strategy semantics: `F = beta*F + (1-beta)*s s^T`,
  `Xbar = beta*Xbar + (1-beta)*a`, `G = beta*G + (1-beta)*Xbar*A^T`,
  `g = G*1` with noise rows zeroed, `dmu = lr_mu * P @ g`.
- Require `lambda_reg > 0`. Default ridge retention for `K_max` is 95%:
  `K_max = max(1, floor(log(0.95)/log(beta)))` for `0 < beta < 1`;
  `K_max = 10**9` for `beta == 1`; `K_max = 1` for `beta <= 0`.
- Refresh **after** the current score updates `F` and **before** `nat = P @ g`.
- Channel layout stays `[site mu rows; site sigma rows; noise rows]`.
- Stage explicit paths. The working tree has user changes (`.gitignore`,
  `.commitmsg`, three other plan docs, `validation_mvp_b/`); do not touch them.

## Review Focus

1. **Cold start:** the first step has no previous phase, so it must not count as a
   phase shift; `P_0 = (1/lambda)I`; `dmu` finite.
2. **Long phase:** no phase shift for many steps; `K_max` caps the ridge drift.
   The refresh restores the ridge, not `F`'s EWMA memory.
3. **Per-leg phase vector:** a change in one component (e.g. L stance -> swing,
   R swing -> swing) triggers exactly one refresh.
4. **New site mid-run (`ensure_names`):** `P` is rebuilt after `F` is resized;
   no stale inverse and no shape mismatch.
5. **Beta edges:** `beta = 1` (no ridge decay), `beta <= 0` (refresh every step),
   and a lowered beta (smaller `K_max`) must not divide by zero or hide the
   phase trigger.

---

### Task 1: Recursive inverse + K_max refresh

**Files:**
- Modify: `persona/strategy.py`
- Modify: `tests/test_strategy.py`

**Interfaces:**
- Consumes: existing `F`, `Xbar`, `G`, score vector `s`, and `g`.
- Produces:
  - `FisherMatrix.P: np.ndarray` (n x n), initialized to `(1/lambda)*I`;
  - `FisherMatrix.refresh_every: int`;
  - `FisherMatrix._steps_since_refresh: int`;
  - `FisherMatrix._rank1_update_inverse(s: np.ndarray) -> None`;
  - `FisherMatrix._refresh_inverse() -> None`;
  - `FisherMatrix.step(sites, noise, advantage)` keeps its current signature
    in this task.

- [x] **Step 1: Write the failing tests in `tests/test_strategy.py`**

```python
def _direct(fm, ridge):
    return np.linalg.inv(fm.F + ridge * np.eye(fm.n_total))


def test_recursive_inverse_tracks_the_decaying_ridge():
    beta, lam = 0.9, 0.5
    fm = FisherMatrix(param_names=["p"], beta=beta, lambda_reg=lam,
                      warmup=0, refresh_every=10**9)
    fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta), atol=1e-10)
    fm.step({"p": (-0.1, 0.4, 0.0)}, {}, np.array([0.0, 1.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta ** 2), atol=1e-10)


def test_kmax_refresh_restores_the_constant_ridge():
    beta, lam = 0.9, 0.5
    fm = FisherMatrix(param_names=["p"], beta=beta, lambda_reg=lam,
                      warmup=0, refresh_every=2)
    fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta), atol=1e-10)
    fm.step({"p": (-0.1, 0.4, 0.0)}, {}, np.array([0.0, 1.0]))
    assert np.allclose(fm.P, _direct(fm, lam), atol=1e-10)


def test_default_kmax_tracks_beta():
    assert FisherMatrix(param_names=["p"], beta=0.995).refresh_every == 10
    assert FisherMatrix(param_names=["p"], beta=0.99).refresh_every == 5
    assert FisherMatrix(param_names=["p"], beta=0.9).refresh_every == 1
    assert FisherMatrix(param_names=["p"], beta=1.0).refresh_every >= 10**6


def test_ensure_names_rebuilds_the_inverse():
    fm = FisherMatrix(param_names=["a"], beta=0.9, lambda_reg=0.5,
                      warmup=0, refresh_every=10**9)
    for _ in range(3):
        fm.step({"a": (0.1, 0.5, 0.0)}, {}, np.array([1.0, 0.0]))
    fm.ensure_names(["a", "b"], [])
    assert fm.P.shape == (4, 4)
    assert np.allclose(fm.P @ (fm.F + fm.lambda_reg * np.eye(4)), np.eye(4),
                       atol=1e-10)


def test_recursive_step_matches_the_direct_solve():
    rng = np.random.default_rng(11)
    beta, lam, kmax = 0.95, 1e-3, 4
    fm = FisherMatrix(param_names=["p"], noise_names=["n"], beta=beta,
                      lambda_reg=lam, warmup=0, refresh_every=kmax, lr_mu=0.1)
    for _ in range(40):
        d = rng.normal(0, 0.2); sig = rng.uniform(0.2, 0.8)
        nd = (rng.normal(0, 0.1), 1.0)
        out = fm.step({"p": (d, sig, 0.0)}, {"n": nd},
                      np.array([1.0, -1.0]))
        g = fm.G.sum(axis=1); g[2 * fm.n_sites:] = 0.0
        k = fm._steps_since_refresh
        ridge = lam if k == 0 else lam * beta ** k
        expected = np.linalg.solve(fm.F + ridge * np.eye(fm.n_total), g)
        assert np.allclose(fm.P @ g, expected, rtol=1e-8, atol=1e-10)
        assert np.isclose(out["p"], fm.lr_mu * expected[0],
                          rtol=1e-8, atol=1e-10)


def test_inverse_stays_symmetric_and_positive_definite():
    rng = np.random.default_rng(12)
    fm = FisherMatrix(param_names=[f"p{i}" for i in range(10)],
                      beta=0.99, warmup=0, refresh_every=5)
    for _ in range(200):
        sites = {f"p{i}": (float(rng.normal(0, 0.2)), 0.5, 0.0)
                 for i in range(10)}
        fm.step(sites, {}, rng.normal(0, 1, 2))
    assert np.all(np.isfinite(fm.P))
    assert np.allclose(fm.P, fm.P.T, atol=1e-10)
    assert np.linalg.eigvalsh(fm.P).min() > 0.0
```

- [x] **Step 2: Run the tests to verify failure**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_strategy.py -k 'recursive_inverse or kmax or default_kmax or ensure_names_rebuilds or matches_the_direct or symmetric_and_positive' -v"`

Expected: FAIL with `TypeError` (unexpected keyword `refresh_every`) or
`AttributeError` (`P` missing).

- [x] **Step 3: Implement in `persona/strategy.py`**

Add `from scipy.linalg import cho_factor, cho_solve` at the top.

Add `refresh_every: int | None = None` as the last constructor parameter.
After `self.baseline_alpha`/`self.steps` setup, add:

```python
if self.lambda_reg <= 0.0:
    raise ValueError("lambda_reg must be positive for the recursive inverse")
self.refresh_every = self._resolve_refresh_every(refresh_every, self.beta)
self._steps_since_refresh = 0
self.P = np.eye(n, dtype=np.float64) / self.lambda_reg
```

Add the resolver:

```python
@staticmethod
def _resolve_refresh_every(value, beta):
    if value is not None:
        return max(1, int(value))
    if beta >= 1.0:
        return 10**9
    if beta <= 0.0:
        return 1
    return max(1, int(np.floor(np.log(0.95) / np.log(beta))))
```

Add the two methods:

```python
def _rank1_update_inverse(self, s):
    u = self.P @ s
    v = float(s @ u)
    denom = self.beta + (1.0 - self.beta) * v
    if denom <= 0.0 or not np.isfinite(denom):
        self._refresh_inverse()
        return
    self.P = (self.P - ((1.0 - self.beta) / denom) * np.outer(u, u)) / self.beta
    self.P = 0.5 * (self.P + self.P.T)

def _refresh_inverse(self):
    n = self.n_total
    if n == 0:
        self.P = np.zeros((0, 0), dtype=np.float64)
        self._steps_since_refresh = 0
        return
    M = self.F + self.lambda_reg * np.eye(n)
    try:
        cf = cho_factor(M, lower=True, check_finite=False)
        P = cho_solve(cf, np.eye(n), check_finite=False)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(M)
    self.P = 0.5 * (P + P.T)
    self._steps_since_refresh = 0
```

In `ensure_names`, after `self.F, self.Xbar, self.G = F, X, G`, call
`self._refresh_inverse()`.

In `step`, after the `F`/`Xbar`/`G` updates and `self.steps += 1`, add:

```python
self._steps_since_refresh += 1
if self._steps_since_refresh >= self.refresh_every:
    self._refresh_inverse()
else:
    self._rank1_update_inverse(s)
```

Replace the solve block with:

```python
nat = self.P @ g
if not np.all(np.isfinite(nat)):
    self._refresh_inverse()
    nat = self.P @ g
if not np.all(np.isfinite(nat)):
    M = self.F + self.lambda_reg * np.eye(n)
    try:
        nat = np.linalg.solve(M, g)
    except np.linalg.LinAlgError:
        nat = np.linalg.pinv(M) @ g
```

Update the class docstring: `dmu = lr_mu * P @ g`, `P` initialized to
`(1/lambda)I`, rank-1 between refreshes, Cholesky refresh at `K_max`. Replace
the `TODO(perf)` text with: "the recursive inverse removes the O(n^3) solve;
`F` is still O(n^2) per step and O(n^2) memory."

- [x] **Step 4: Run the tests to verify pass**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_strategy.py -v"`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add persona/strategy.py tests/test_strategy.py
git commit -m "feat(persona): O(n^2) recursive Fisher inverse with K_max Cholesky refresh"
```

---

### Task 2: Phase-shift trigger

**Files:**
- Modify: `persona/strategy.py`
- Modify: `persona/agent.py`
- Modify: `tests/test_strategy.py`
- Modify: `tests/test_agent_strategy.py`

**Interfaces:**
- Consumes: `FisherMatrix.refresh_every`, `P`, `_refresh_inverse()` from Task 1.
- Produces:
  - `FisherMatrix.step(..., *, phase_shift: bool = False)`;
  - `Agent._prev_phase`;
  - `Agent.step` passes the flag; first step and unchanged phase do not refresh.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_strategy.py`:

```python
def test_phase_shift_refreshes_before_kmax():
    beta, lam = 0.9, 0.5
    fm = FisherMatrix(param_names=["p"], beta=beta, lambda_reg=lam,
                      warmup=0, refresh_every=100)
    fm.step({"p": (0.2, 0.5, 0.1)}, {}, np.array([1.0, 0.0]))
    fm.step({"p": (-0.1, 0.4, 0.0)}, {}, np.array([0.0, 1.0]))
    assert np.allclose(fm.P, _direct(fm, lam * beta ** 2), atol=1e-10)
    fm.step({"p": (0.05, 0.6, 0.0)}, {}, np.array([1.0, 0.0]),
            phase_shift=True)
    assert np.allclose(fm.P, _direct(fm, lam), atol=1e-10)
```

Append to `tests/test_agent_strategy.py`:

```python
def test_agent_refreshes_on_phase_shift_only(nest_kernel):
    agent, _ = _agent()
    agent.strategy.refresh_every = 1000
    calls = []
    orig = agent.strategy._refresh_inverse

    def spy():
        calls.append(agent.strategy.steps)
        return orig()

    agent.strategy._refresh_inverse = spy
    for t, phase in enumerate(["stance", "stance", "swing", "swing"]):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.zeros(2), phase=phase,
                   t_ms=(t + 1) * 10.0)
    assert calls == [3]


def test_agent_refreshes_when_one_leg_changes_phase(nest_kernel):
    agent, _ = _agent()
    agent.strategy.refresh_every = 1000
    calls = []
    orig = agent.strategy._refresh_inverse

    def spy():
        calls.append(agent.strategy.steps)
        return orig()

    agent.strategy._refresh_inverse = spy
    phases = [["stance", "swing"], ["stance", "swing"], ["swing", "swing"]]
    for t, phase in enumerate(phases):
        agent.sample_actions(t_ms=t * 10.0)
        agent.step(np.zeros(2), np.zeros(2), phase=phase,
                   t_ms=(t + 1) * 10.0)
    assert calls == [3]
```

Update the two existing spies in `tests/test_agent_strategy.py` to accept the
new keyword:

```python
def spy(sites, noise, advantage, **kwargs):
    ...
    return orig(sites, noise, advantage, **kwargs)
```

- [x] **Step 2: Run the tests to verify failure**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_strategy.py::test_phase_shift_refreshes_before_kmax tests/test_agent_strategy.py -v"`

Expected: FAIL with `TypeError` (unexpected keyword `phase_shift`) or a
missing `_prev_phase`.

- [x] **Step 3: Implement**

In `persona/strategy.py`, change the signature to:

```python
def step(self, sites, noise, advantage, *, phase_shift: bool = False):
```

Change the refresh condition to:

```python
if phase_shift or self._steps_since_refresh >= self.refresh_every:
    self._refresh_inverse()
else:
    self._rank1_update_inverse(s)
```

In `persona/agent.py`:

- In `__init__`, add `self._prev_phase = None`.
- In `step`, after `phase_key` is computed, add:

```python
phase_shift = (self._prev_phase is not None
               and phase_key != self._prev_phase)
self._prev_phase = phase_key
```

- Change the strategy call to:

```python
delta_means = self.strategy.step(
    sites, noise, advantage_vec, phase_shift=phase_shift)
```

- Update the `step` docstring: step 4 now says the strategy refreshes `P` on
  a phase shift or `K_max`.

- [x] **Step 4: Run the tests to verify pass**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_strategy.py tests/test_agent_strategy.py tests/test_phase.py -v"`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add persona/strategy.py persona/agent.py tests/test_strategy.py tests/test_agent_strategy.py
git commit -m "feat(persona): refresh the recursive inverse on stance/swing shifts"
```

---

### Task 3: Acceptance, benchmark, and design-doc status

**Files:**
- Modify: `docs/superpowers/specs/2026-10-20-persona-recursive-fisher-inverse-design.md`
- Modify: `docs/superpowers/plans/2026-10-20-persona-recursive-fisher-inverse.md`

- [x] **Step 1: Run the full suite**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest -q"`

Expected: all tests pass (143+).

- [x] **Step 2: Run the fast end-to-end and compare with `validation_mvp_b`**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python run_validation.py --fast --out /mnt/d/CPGvalidation/validation_mvp_recursive"`

Then inspect:

```bash
wsl -e bash -lc "cd /mnt/d/CPGvalidation && /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python - <<'PY'
import h5py, numpy as np
with h5py.File('validation_mvp_recursive/run.h5', 'r') as f:
    loss = np.asarray(f['agent/loss'][()], dtype=float)
    means = {k: np.asarray(v[()], dtype=float)
             for k, v in f['agent/means'].items()}
print('loss_last10', float(loss[-10:].mean()))
print('max_dmu', max(float(np.abs(np.diff(v)).max()) for v in means.values()))
PY"
```

Record the numbers. Baseline (`validation_mvp_b`): loss last-10 `0.8725`,
`max|dmu|` `0.438`. Acceptance: no NaNs; loss last-10 within 10% of the
baseline (`0.785`–`0.960`); `max|dmu|` no more than 10% above baseline
(`<= 0.482`). If worse, do not rescale `g` silently; report the delta and
investigate the ridge drift.

- [x] **Step 3: Record the microbenchmark**

Run:

```bash
wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python - <<'PY'
import time
import numpy as np
from persona.strategy import FisherMatrix

rng = np.random.default_rng(0)
names = [f'p{i}' for i in range(221)]
noise = [f'n{i}' for i in range(16)]
sites = {k: (float(rng.normal(0, 0.1)), 0.5, 0.0) for k in names}
nd = {k: (float(rng.normal(0, 0.1)), 1.0) for k in noise}
A = np.array([1.0, -1.0])

def run(refresh_every, steps=20):
    fm = FisherMatrix(param_names=names, noise_names=noise, warmup=0,
                      refresh_every=refresh_every)
    for _ in range(5):
        fm.step(sites, nd, A)
    t0 = time.perf_counter()
    for _ in range(steps):
        fm.step(sites, nd, A)
    return (time.perf_counter() - t0) / steps * 1e3, fm

no_refresh, fm = run(10**9)
k10, _ = run(10)
g = fm.G.sum(axis=1); g[2 * fm.n_sites:] = 0.0
M = fm.F + fm.lambda_reg * np.eye(fm.n_total)
t0 = time.perf_counter()
for _ in range(20):
    np.linalg.solve(M, g)
direct = (time.perf_counter() - t0) / 20 * 1e3
print(f'no-refresh {no_refresh:.3f} ms/step')
print(f'K=10       {k10:.3f} ms/step')
print(f'direct     {direct:.3f} ms/step')
PY"
```

Record the three timings in the design doc. The expected direction is
`no-refresh < K=10 < direct`; no hard timing assertion (BLAS/threading
dependent).

- [x] **Step 4: Update the design doc and plan**

In the design doc, change **Status** to
`implemented (2026-10-20)` and add a short "Results" section with the
full-suite result, the fast-e2e loss/`max|dmu|`, and the three benchmark
timings. Tick the task checkboxes in this plan that have been completed.

- [x] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-10-20-persona-recursive-fisher-inverse-design.md docs/superpowers/plans/2026-10-20-persona-recursive-fisher-inverse.md
git commit -m "docs(persona): record recursive-inverse acceptance and benchmark"
```

---

### Task 4: Performance + accuracy harness (spreadsheets + figures)

**Files:**
- Create: `bench_recursive_inverse.py`
- Create: `tests/test_bench_recursive_inverse.py`
- Modify: `docs/superpowers/specs/2026-10-20-persona-recursive-fisher-inverse-design.md`

**Interfaces:**
- Consumes: `persona.strategy.FisherMatrix` and its `P`, `F`,
  `_steps_since_refresh`, `refresh_every`.
- Produces:
  - `run_benchmark(out_dir, n_values=(64, 128, 256, 458, 512, 1024),
    seeds=(0, 1, 2), steps=30, refresh_every=10, repeats=3,
    make_figures=True, accuracy_steps=None, accuracy_n_values=None) -> dict`;
  - `main(argv=None) -> int` with `--out`, `--quick`, `--no-figures`;
  - `performance.csv`: `n, mode, seed, repeat, step_ms`;
  - `accuracy.csv`: `n, seed, step, k, rel_error_decaying,
    rel_error_constant, cosine_constant, dmu_rel_error_constant`;
  - `summary.csv`: `table, group, metric, mean, std, median, q25, q75`;
  - `performance.png`, `accuracy.png`;
  - `recursive_inverse_results.xlsx` when `openpyxl` or `xlsxwriter` is
    importable;
  - `MODES = ("direct", "recursive_no_refresh", "recursive_kmax",
    "recursive_chatter")`, with `CHATTER_EVERY = 3`; the chatter mode models
    the observed ~2.9-step phase-change interval.

- [x] **Step 1: Write the failing smoke test**

Create `tests/test_bench_recursive_inverse.py`:

```python
import pandas as pd

from bench_recursive_inverse import MODES, run_benchmark


def test_benchmark_writes_tables_and_figures(tmp_path):
    run_benchmark(tmp_path, n_values=(32,), seeds=(0,), steps=8,
                  refresh_every=4, repeats=1, make_figures=True)
    for name in ("performance.csv", "accuracy.csv", "summary.csv"):
        assert (tmp_path / name).exists()
    perf = pd.read_csv(tmp_path / "performance.csv")
    acc = pd.read_csv(tmp_path / "accuracy.csv")
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert {"n", "mode", "seed", "repeat", "step_ms"}.issubset(perf.columns)
    assert set(perf["mode"]) == set(MODES)
    assert {"n", "seed", "step", "k", "rel_error_decaying",
            "rel_error_constant", "cosine_constant",
            "dmu_rel_error_constant"}.issubset(acc.columns)
    assert {"table", "group", "metric", "mean", "std", "median",
            "q25", "q75"}.issubset(summary.columns)
    assert (perf["step_ms"] > 0.0).all()
    assert acc["rel_error_decaying"].max() < 1e-8
    assert acc.loc[acc["k"] == 0, "rel_error_constant"].max() < 1e-8
    assert acc.loc[acc["k"] > 0, "rel_error_constant"].max() > 0.0
    assert (tmp_path / "performance.png").exists()
    assert (tmp_path / "accuracy.png").exists()
    assert (tmp_path / "README.md").exists()
    try:
        import openpyxl  # noqa: F401
        has_writer = True
    except ImportError:
        try:
            import xlsxwriter  # noqa: F401
            has_writer = True
        except ImportError:
            has_writer = False
    if has_writer:
        assert (tmp_path / "recursive_inverse_results.xlsx").exists()
```

- [x] **Step 2: Run the test to verify failure**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_bench_recursive_inverse.py -v"`

Expected: FAIL with `ModuleNotFoundError: No module named
'bench_recursive_inverse'`.

- [x] **Step 3: Implement `bench_recursive_inverse.py`**

The script is pure numpy/pandas/matplotlib; it does not create a NEST
network. Use `matplotlib.use("Agg")` before importing `pyplot`.

Implement helpers:

```python
def _layout(n):
    n_noise = min(16, max(1, n // 4))
    n_sites = (n - n_noise) // 2
    return [f"p{i}" for i in range(n_sites)], [f"n{i}" for i in range(n_noise)]
```

`run_benchmark`:

1. Create `out_dir`; for each `(n, seed)`, generate one score/gradient stream
   shared by all modes and repeats, so the modes see the same data.
2. **Performance** (no `FisherMatrix`, to isolate the linear algebra):
   pre-generate `S` (`steps x n`) and `G` (`steps x n`); for each mode run:
   - `direct`: `F = beta*F + (1-beta)*outer(s,s)`;
     `nat = np.linalg.solve(F + lam*I, g)`.
   - `recursive_no_refresh`: maintain `P = (1/lam)I`; each step update
     `F`, apply the Sherman-Morrison update, `nat = P @ g`; no refresh.
   - `recursive_kmax`: same but refresh every `refresh_every` steps after
     the `F` update, using `scipy.linalg.cho_factor/cho_solve`.
   - `recursive_chatter`: same as `recursive_kmax`, but also refresh every
     `CHATTER_EVERY = 3` steps to model the observed phase-change interval.
   Time `steps` iterations with `time.perf_counter()`; append one row per
   `(n, mode, seed, repeat)`.
3. **Accuracy** (use `FisherMatrix`): for each `(n, seed)`, construct
   `FisherMatrix(param_names=names, noise_names=noise, beta=0.995,
   warmup=0, refresh_every=refresh_every, lr_mu=0.1)`. For each step,
   generate random site/noise values, call `fm.step`, then compute:
   - `g = fm.G.sum(axis=1); g[2*fm.n_sites:] = 0`;
   - `k = fm._steps_since_refresh`;
   - `ridge = lam if k == 0 else lam * beta ** k`;
   - `expected_decay = np.linalg.solve(fm.F + ridge*I, g)`;
   - `expected_const = np.linalg.solve(fm.F + lam*I, g)`;
   - `got = fm.P @ g`;
   - `rel_error_decaying = ||got - expected_decay|| / max(||expected_decay||, 1e-12)`;
   - `rel_error_constant = ||got - expected_const|| / max(||expected_const||, 1e-12)`;
   - `cosine_constant = dot(got, expected_const) / (||got||*||expected_const|| + 1e-12)`;
   - `dmu_rel_error_constant = ||got[:2*n_sites] - expected_const[:2*n_sites]|| / max(||expected_const[:2*n_sites]||, 1e-12)`.
   Append one row per step.
4. **Tables:** write `performance.csv`, `accuracy.csv`. Build `summary.csv`
   by grouping performance on `(n, mode)` and accuracy on `(n, k)`, with
   `mean`, `std`, `median`, `q25`, `q75`; emit a `table` column
   (`performance`/`accuracy`) and a readable `group` string.
5. **Figures:** `performance.png` is a log-log line plot of mean `step_ms`
   vs `n` with std error bars, one line per mode. `accuracy.png` has two
   panels: (a) mean `rel_error_decaying` vs `k` for `n=458` (log y, dashed
   1e-12 reference); (b) mean `rel_error_constant` vs `k` for each `n`
   (log y, shaded std).
6. **XLSX:** if `openpyxl` or `xlsxwriter` is importable, write
   `recursive_inverse_results.xlsx` with sheets `performance`, `accuracy`,
   `summary`; otherwise skip silently (the CSVs are the fallback).
7. Write `README.md` in `out_dir` with the run command, column meanings,
   and how to read the figures.
8. Return a dict with the three DataFrames.

`main` uses `argparse`: `--out` default
`benchmarks/recursive_inverse`, `--quick` uses `n_values=(64,)`,
`seeds=(0,)`, `steps=8`, `repeats=1`, and `--no-figures`.

- [x] **Step 4: Run the smoke test to verify pass**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_bench_recursive_inverse.py -v"`

Expected: PASS.

- [x] **Step 5: Run the full harness and inspect the artifacts**

Run:
`wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python bench_recursive_inverse.py --out benchmarks/recursive_inverse"`

Expected: `performance.csv`, `accuracy.csv`, `summary.csv`,
`performance.png`, `accuracy.png`, `recursive_inverse_results.xlsx` (when an
Excel writer is available), and `README.md` exist. Check that
`accuracy.csv` has `rel_error_decaying < 1e-8`, `rel_error_constant` is
~0 at `k=0`, and grows with `k`; check that the performance plot shows the
direct solve above the stable-phase recursive modes at large `n`, while the
chatter mode approaches or exceeds direct. Report means and medians, not one
as the other.

- [x] **Step 6: Update the design doc and commit**

Add a "Results" subsection to the design doc with the full-harness
timings and the worst `rel_error_constant` at `k = refresh_every - 1`. Tick
the Task 4 checkboxes. Commit the script, test, plan, and design doc (leave
the generated `benchmarks/recursive_inverse/` artifacts untracked unless the
user asks to commit them):

```bash
git add bench_recursive_inverse.py tests/test_bench_recursive_inverse.py docs/superpowers/specs/2026-10-20-persona-recursive-fisher-inverse-design.md docs/superpowers/plans/2026-10-20-persona-recursive-fisher-inverse.md
git commit -m "test(persona): recursive-inverse performance and accuracy harness"
```

---

## Self-Review

- **Spec coverage:** A1/A2 -> Task 1 (`test_recursive_inverse_tracks_the_decaying_ridge`,
  `test_kmax_refresh_restores_the_constant_ridge`); A3 -> Task 1
  (`test_recursive_step_matches_the_direct_solve`); A4 -> Task 1
  (`test_inverse_stays_symmetric_and_positive_definite`, asserted after every
  step); A5/A6 -> Task 2 (`test_phase_shift_refreshes_before_kmax`, the Agent
  first-step, string/tuple, first-leg, and second-leg tests); A7 -> Task 1
  (`test_ensure_names_rebuilds_the_inverse`, including the counter reset);
  A8/A9 -> Task 3; A10/A11 -> Task 4
  (`test_benchmark_writes_tables_and_figures`, which checks summary columns,
  README, and the optional XLSX); A12 -> Task 1
  (`test_k1_step_uses_direct_solve_without_building_p`,
  `test_beta_zero_forces_constant_ridge_and_stays_finite`,
  `test_invalid_beta_and_ridge_are_rejected`).
- **Step scan:** every step names the file, the test, the command, or the exact
  implementation change. No `TBD` or open-ended "handle edge cases".
- **Type consistency:** `P`, `refresh_every`, `_steps_since_refresh`,
  `_rank1_update_inverse`, `_refresh_inverse`, and `phase_shift` are used with
  the same names and types across tasks.
- **Review Focus tests:** cold start (existing `test_few_steps_singular_no_crash`
  plus `test_agent_refreshes_on_phase_shift_only`), long phase
  (`test_kmax_refresh_restores_the_constant_ridge`), per-leg phase
  (`test_agent_refreshes_when_one_leg_changes_phase` and the second-leg test),
  new site (`test_ensure_names_rebuilds_the_inverse`), beta edges
  (`test_default_kmax_tracks_beta`, `test_invalid_beta_and_ridge_are_rejected`,
  and the beta-zero/beta-one tests), non-finite repair
  (`test_nonfinite_inverse_is_repaired`).
- **Proportion:** two small code tasks plus one acceptance task; the plan
  specifies interfaces and tests, not full function bodies beyond the
  Sherman-Morrison/refresh core.

## Review follow-up (2026-10-20)

A fresh-context review found no critical issues and five important items.
The follow-up addressed them without changing the numerical semantics:

- A4 now asserts symmetry, finiteness, positive definiteness, and the refresh
  counter after every step, not only at a refresh boundary.
- `K_max=1` now solves `F + lambda*I` directly and invalidates a lazy `P`
  cache; `test_k1_step_uses_direct_solve_without_building_p` enforces that the
  full inverse is not built. Reading `P` rebuilds it correctly.
- The harness now includes `recursive_chatter` (refresh every 3 steps) to
  model the observed ~2.9-step phase-change interval, and the Results section
  reports both stable-phase and chatter timings.
- `beta` is validated as finite in `[0, 1]`; `beta=0` forces `K_max=1` and
  stays finite with an explicit refresh interval. `lambda` must be finite and
  positive.
- The design/plan now label harness statistics as means/medians, not
  medians-only, and report the actual medians and ratios.

Minor review items also fixed: phase-key canonicalization (with history
compatibility), first-step-swing and second-leg phase tests, an
`ensure_names` counter assertion, a non-finite repair test, the accuracy
figure's std band, the README's μ/σ description, summary/README/XLSX smoke
assertions, and the one-off microbenchmark's provenance. Phase debouncing is
still deliberately out of scope; the chatter benchmark quantifies the cost.
