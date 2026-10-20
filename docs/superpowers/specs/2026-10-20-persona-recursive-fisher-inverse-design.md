# PERSONA recursive Fisher inverse — design

**Status:** implemented and review-hardened (2026-10-20). This is the
normative design for the performance follow-up. It does not change PERSONA's
credit, noise, or sigma semantics.

## 1. Context

`FisherMatrix.step` currently forms `M = F + lambda*I` and calls
`np.linalg.solve(M, g)` every control step. At `n = 458` (221 sites plus 16
producers), the solve is 6.76 ms/call and about 68% of the 10 ms online step.
`F` changes by one rank-1 term per step, so its inverse can be updated in
O(n^2) instead of refactorizing in O(n^3).

## 2. Goal

Maintain `P ~= (F + lambda*I)^-1` and compute `dmu = eta * P @ g` in O(n^2) per
step, refreshing `P` exactly with a Cholesky factorization at a bounded
interval. Preserve the existing `F`, `Xbar`, `G`, `g`, noise scoring, and sigma
updates.

Non-goals:

- low-rank/block approximation of `F` (measured unsafe for the mu step);
- changing `beta`, `lambda`, the credit gain, or the noise scales;
- phase-local `F`;
- changing the paper's algorithm or the public strategy semantics.

## 3. Recursion

Let `s_t` be the score vector and

```text
F_t = beta * F_{t-1} + (1 - beta) * s_t s_t^T
```

With `P_{t-1} = F_{t-1}^-1`, Sherman-Morrison gives

```text
u = P_{t-1} @ s_t
v = s_t^T @ u
P_t = (1/beta) * (P_{t-1} - (1-beta) * u u^T / (beta + (1-beta) * v))
```

Cost: two matrix-vector products and one symmetric outer product, O(n^2).
Then

```text
nat = P_t @ g_t
```

is another O(n^2) product; no factorization.

## 4. Ridge

Initialize `P_0 = (1/lambda) * I` (`F_0 = 0`). If no refresh occurs, the
recursion maintains exactly

```text
P_t = (F_t + lambda * beta^t * I)^-1
```

Proof by induction: if `P_{t-1} = (F_{t-1} + lambda*beta^(t-1)*I)^-1`, the
update above is the inverse of

```text
beta*(F_{t-1} + lambda*beta^(t-1)*I) + (1-beta)*s_t s_t^T
= F_t + lambda*beta^t*I.
```

The prior fades with the same `beta` as the data. Refresh at a phase shift or
every `K_max` steps, whichever comes first:

```text
P <- (F + lambda*I)^-1        via Cholesky
```

This restores the constant ridge and removes accumulated round-off. `K_max` is
chosen so the ridge retains at least 95% between refreshes:

```text
K_max = max(1, floor(log(0.95) / log(beta)))   for 0 < beta < 1
K_max = 10^9                                    for beta = 1 (no decay)
K_max = 1                                       for beta = 0 (no memory)
```

`beta` must be finite and in `[0, 1]`; negative, NaN, and above-one values
are rejected. `K_max = 1` uses the direct-solve fast path described in
section 6.

| beta | K_max |
|---|---|
| 0.995 | 10 |
| 0.99 | 5 |
| 0.98 | 2 |
| 0.95 | 1 |
| 0.90 | 1 |

At a refresh step, refresh **after** updating `F` with the current score and
**before** computing `nat`. The step at the boundary then uses the exact
`(F_t + lambda*I)^-1`. Between refreshes the step uses
`(F_t + lambda*beta^k*I)^-1`; the difference is bounded by the ridge drift.

## 5. Trigger policy

- `FisherMatrix.step(..., *, phase_shift: bool = False)`.
- Refresh if `phase_shift` or `_steps_since_refresh >= K_max`.
- `Agent.step` compares a canonical per-component `phase_key` with the
  previous step's key. A string and an equivalent per-leg tuple compare equal
  (`"stance"` == `("stance", "stance")`), while a change in any component
  counts as a shift. The first step never counts as a shift. History and the
  returned trace retain the caller's string form or the canonical tuple form
  for compatibility.
- The phase shift is the semantic event; `K_max` is the safety cap for long
  phases, standing, missed phase detection, or a lowered `beta`.
- Chattering phase detectors: this design assumes a stable phase signal. A
  shift every step would refresh every step (O(n^3)). The validation path
  changes phase about every 2.9 steps, which the harness models with a
  3-step chatter mode. If that rate can grow, debounce upstream or add a
  minimum refresh interval in a follow-up.

**Important:** the refresh restores the ridge, not `F`'s memory. `F` remains
the beta-EWMA with effective memory `1/(1-beta)`; a long phase still forgets
its first updates at that rate. `K_max` does not change that. If phase-local
`F` is desired, that is a separate design (reset `F` at phase boundaries, or
maintain per-phase metrics).

## 6. Numerical policy

- Require `lambda` finite and `> 0`; require `beta` finite and in `[0, 1]`.
- Symmetrize `P` after each rank-1 update and refresh:
  `P = 0.5 * (P + P.T)`.
- Refresh with `scipy.linalg.cho_factor` / `cho_solve` on `F + lambda*I`; on
  `numpy.linalg.LinAlgError`, fall back to `np.linalg.pinv`.
- When `K_max = 1`, solve `F + lambda*I` directly for `nat` and invalidate the
  lazy `P` cache instead of forming the full inverse. `P` is rebuilt only if
  read. This keeps the short-memory (`beta ~ 0.95`) path at the old direct
  cost; `beta = 0` uses the same path.
- If `P @ g` is non-finite, refresh and retry; if still non-finite, fall back
  to `np.linalg.solve` / `pinv` on `F + lambda*I`.
- `ensure_names` rebuilds `P` after resizing `F` (new channels have zero `F`
  and the lambda ridge), and resets the refresh counter.
- Memory: `F` and `P` are each n x n; about 3.4 MB total at n = 458, growing
  O(n^2).

## 7. Interfaces

- `FisherMatrix.P: np.ndarray` (n x n; a lazy cache that is rebuilt if read
  after a `K_max = 1` step).
- `FisherMatrix.refresh_every: int`.
- `FisherMatrix.step(sites, noise, advantage, *, phase_shift: bool = False)
  -> Dict[str, float]`.
- `FisherMatrix._rank1_update_inverse(s)` and `_refresh_inverse()`.
- `Agent._prev_phase`; `Agent.step` passes `phase_shift`.

## 8. Acceptance

- A1. At a refresh, `P == inv(F + lambda*I)` (atol 1e-9).
- A2. Between refreshes, `P == inv(F + lambda*beta^k*I)` (atol 1e-9).
- A3. `P @ g` matches `solve(F + lambda*beta^k*I, g)` (rtol 1e-8); the returned
  `dmu` matches the direct solve.
- A4. `P` is symmetric, finite, and positive definite over 200 random steps.
- A5. A phase shift triggers exactly one refresh; an unchanged phase does not;
  the first step does not.
- A6. `K_max` triggers at the expected steps; a phase shift before `K_max`
  refreshes earlier.
- A7. `ensure_names` rebuilds `P`.
- A8. Full suite green; fast e2e finite; recursive loss last-10 within 10% of
  the same-network constant-ridge A/B and `max|dmu|` no worse than that A/B.
  The historical `validation_mvp_b` (221 sites) differs from the current
  217-site network, so it is a reference, not an exact A/B.
- A9. One-off strategy-step microbenchmark recorded at n = 458: old direct
  full step, recursive no-refresh, recursive `K_max=10`, and direct
  solve-only. This is separate from the harness's linear-algebra timing.
- A10. The benchmark harness writes `performance.csv`, `accuracy.csv`,
  `summary.csv` (and `.xlsx` when an Excel writer, `openpyxl` or
  `xlsxwriter`, is available), plus `performance.png` and `accuracy.png`.
- A11. A smoke test runs the harness at small n and asserts the files and
  columns exist, that the decaying-ridge error is < 1e-8, and that the
  constant-ridge error is zero at refresh steps and grows between them.
- A12. `K_max = 1` uses a direct solve and does not build the full inverse;
  reading `P` afterwards rebuilds it correctly. `beta = 0` is finite, and
  invalid `beta`/`lambda` values are rejected.

## 9. Results

Implemented and review-hardened on 2026-10-20 on `mujoco_sim`.

- **Full suite:** 173 tests pass on the current checkout. The count includes
  unrelated CPG working-tree changes; the recursive-inverse tests are in
  `tests/test_strategy.py`, `tests/test_agent_strategy.py`, and
  `tests/test_bench_recursive_inverse.py`.
- **Fast e2e (current checkout, 400 steps):** the recursive default
  (`beta=0.995`, `K_max=10`, phase shifts) gives loss last-10 `0.6803`,
  `max|dmu|` `0.4492`, and no NaNs. A same-network constant-ridge A/B
  (`refresh_every=1`, exact solve every step) gives `0.7515`/`0.4579`; the
  recursive run is -9.5% loss and -1.9% `max|dmu|`, inside A8. The direct A/B
  was produced by setting `refresh_every=1` after Agent construction; both
  runs use the same current network, including the later CPG WIP. These are
  single stochastic fast runs; the historical 221-site baseline and the
  217-site recursive run are not an exact A/B.
- **One-off strategy-step microbenchmark, n = 458** (221 sites + 16
  producers; WSL CPU; BLAS threads pinned to 1; not regenerated by the
  harness): old direct `np.linalg.solve` every step `34.96 ms`; recursive
  no-refresh `24.46 ms`; recursive `K_max=10` `30.32 ms`; direct solve-only
  `10.05 ms`. The old Cholesky-every-step `68.78 ms` is not the current
  `K_max=1` path; that path now solves directly. The fair full-step ordering
  is `no-refresh < K_max=10 < old direct`. These are strategy-step timings
  and are not comparable to the harness below, which times only the `F`
  update, inverse update, and `P @ g`.
- **Full harness** (`benchmarks/recursive_inverse`, 3 seeds x 3 repeats = 9
  samples per cell; mean / median): the WSL CPU was shared with unrelated
  load during this run, so absolute times are inflated; within-run ratios
  are the useful result. At `n=458`: direct `20.17 / 23.89 ms`, no-refresh
  `18.52 / 18.38 ms` (1.30x median), `K_max=10` `21.87 / 22.02 ms` (1.09x),
  chatter `28.42 / 27.73 ms` (0.86x, slower). At `n=1024`: direct
  `142.49 / 141.95 ms`, no-refresh `75.04 / 74.20 ms` (1.91x), `K_max=10`
  `110.44 / 112.89 ms` (1.26x), chatter `204.87 / 203.46 ms` (0.70x,
  slower). At `n=64`, the fixed refresh overhead makes `K_max=10` slower than
  direct; all times are sub-millisecond. An earlier quieter run measured
  means `7.70 / 6.78 / 5.23 ms` at `n=458` and `59.02 / 45.31 / 30.59 ms`
  at `n=1024`; the ordering agrees, but the absolute values are not directly
  comparable.
- **Accuracy:** same-ridge relative error overall max `1.43e-12` (max group
  mean `1.03e-12`, median `1.12e-12`); constant-ridge drift at a refresh max
  mean `9.65e-13`, at `k=9` max `4.61%`, inside the 5% design bound; minimum
  cosine similarity mean `0.999962`; maximum `dmu` relative error `4.61%`.
  The harness writes `performance.csv`, `accuracy.csv`, `summary.csv`,
  `recursive_inverse_results.xlsx`, `performance.png`, `accuracy.png`, and
  `README.md`.
- **Phase churn:** the validation log has 129-130 phase-key changes in 400
  steps (~2.9-step effective refresh interval), so the every-3-step chatter
  mode is the realistic planning case. It is slower than direct at `n=458`
  and `n=1024`; the `K_max` speedup is a stable-phase result. A debounce or
  minimum refresh interval remains a follow-up.

## 10. Benchmark harness

The implementation plan adds a standalone, NEST-free harness
(`bench_recursive_inverse.py`) that measures performance and accuracy and
writes spreadsheet-ready tables and figures:

- **Performance:** mean, median, and spread of ms/step for the direct
  `F`-update-plus-solve path, recursive inverse without refresh, recursive
  inverse with `K_max=10`, and a `recursive_chatter` mode that refreshes
  every 3 steps to model the observed phase-change rate, over
  `n = [64, 128, 256, 458, 512, 1024]`. This times the linear algebra, not a
  full Agent step.
- **Accuracy:** for multiple seeds, the relative error of `P @ g` against
  `solve(F + lambda*beta^k*I, g)` (should be ~machine precision) and against
  the constant-ridge `solve(F + lambda*I, g)` (the controlled ridge drift),
  with the cosine similarity of the resulting `dmu` and the relative error
  over the μ/σ blocks (`2*n_sites` rows; noise rows are excluded). The
  constant-ridge figure includes a shaded standard-deviation band.
- **Artifacts:** `performance.csv`, `accuracy.csv`, `summary.csv`,
  `performance.png`, `accuracy.png`, and `recursive_inverse_results.xlsx`
  when `openpyxl` or `xlsxwriter` is installed. The files open directly in
  Excel or any spreadsheet program.

## 11. Risks

- Ridge drift between refreshes (5% retention at `K_max=10`); measure on the
  fast e2e.
- Numerical drift in `P`; refresh, symmetrization, and the Cholesky fallback
  bound it.
- Phase chattering; the observed ~2.9-step interval is now benchmarked and
  can make the refresh path slower than direct. Debounce or impose a minimum
  interval if the rate grows.
- `K_max=1` and `beta=0` use the direct solve and a lazy `P`; a caller that
  reads `P` every step still pays the full inverse. The online `step` path
  does not.
- O(n^2) memory and the O(n^2) `F` outer product remain. If `n` grows into the
  thousands, revisit the score-window/dual formulation or structured methods.
- Phase-local `F` is out of scope; the refresh does not restore forgotten
  phase data.
